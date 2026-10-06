"""
THE GROUND -- owned by Mukul Yadav.

Everything about the surface lives here: the force law, the material sampling,
the dent that persists, and how the dent is drawn.

The contract with the environment is one method:

    forces = ground.forces(positions, velocities)

positions/velocities are (n, 3) world arrays for the contact sites; the return
is an (n, 3) array of world forces to apply at those sites. No MuJoCo types
cross that boundary except in `update_visual`.
"""
import numpy as np

# --------------------------------------------------------------- constants
D_REF = 0.02      # m, depth at which `alpha` reaches full effect
V_EPS = 1e-3      # m/s, friction smoothing width
PATCH_HALF = 0.065   # m, half-length of the ground patch each contact point
                     # represents (foot 0.26 m long, split between heel and toe)

# ===========================================================================
# PURE FORCE LAW -- no state, no MuJoCo. Testable against hand arithmetic.
# ===========================================================================
def normal_force(depth, sink_rate, k0, c, alpha, f_yield):
    """Vertical ground force at one contact point.

    depth      : penetration below the (possibly dented) surface, m. <=0 -> no contact
    sink_rate  : rate of penetration, m/s (positive = sinking)
    returns    : (force_N, extra_dent_m or None)

    Stiffness grows with depth and the force saturates at f_yield, past which
    the surface yields plastically and is permanently displaced.
    """
    if depth <= 0.0:
        return 0.0, None

    k = k0 * (1.0 + alpha * depth / D_REF)
    f = max(k * depth + c * sink_rate, 0.0)      # ground pushes, never pulls

    if f > f_yield:                              # plastic yield
        elastic_depth = f_yield / k              # depth it can still support
        # Damping can push f past f_yield while the SPRING part is still below
        # it (fast sinking, shallow depth). Then depth < elastic_depth and the
        # naive difference is negative -- which would make the surface RISE.
        # A surface can only ever dent downward, so clamp at zero.
        return f_yield, max(depth - elastic_depth, 0.0)
    return f, None


def friction_force(normal, tangential_velocity, mu):
    """Velocity-regularised Coulomb friction.

    tanh instead of a hard sign() avoids the stick-slip discontinuity that
    makes an explicit integrator chatter. Note the minus sign: friction
    OPPOSES motion.
    """
    return -mu * normal * np.tanh(tangential_velocity / V_EPS)


# ===========================================================================
# MATERIAL DISTRIBUTIONS
# Test ranges do NOT overlap training ranges. That is the anti-memorisation
# design, and tests/test_ground.py verifies it.
# ===========================================================================
GROUND_RANGES = {
    "train": dict(
        k0=[(5e3, 1e5)],
        zeta=(0.05, 0.30),                        # damping ratio, not c
        alpha=(0.0, 2.0),
        f_yield=(300.0, 3000.0),
        mu=(0.30, 0.90),
    ),
    "test": dict(
        k0=[(2e3, 5e3), (1e5, 2e5)],             # two disjoint bands
        zeta=(0.30, 0.60),                        # heavier, "dead" ground
        alpha=(2.0, 4.0),
        f_yield=(150.0, 300.0),
        mu=(0.15, 0.30),
    ),
}


# ===========================================================================
# THE GROUND MODEL -- holds the per-episode material and the dent state.
# ===========================================================================
class GroundModel:
    """One patch of ground, sampled fresh each episode.

    Each contact point draws its OWN material. If the heel lands on something
    softer than the toe, it sinks further, the foot tilts, the leg tilts with
    it, and the body is pushed toward the edge of the foot. That asymmetry is
    the mechanical link between unknown ground and falling.
    """

    def __init__(self, split="train", n_points=2, total_mass=73.5):
        if split not in GROUND_RANGES:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.split = split
        self.n = n_points
        # kg, total mass of the leg model. Converts a sampled damping ratio zeta
        # into a damping coefficient: c = 2*zeta*sqrt(k0*total_mass). Assumes one
        # contact point carries the whole body (heel strike). The environment
        # should pass the real mass from the loaded MuJoCo model; 73.5 is only
        # the default for standalone use and tests.
        self.total_mass = total_mass
        self.k0 = np.zeros(n_points)
        self.zeta = np.zeros(n_points)
        self.last_x = np.zeros(n_points)   # x of each point, for the visual
        self._cells = None                 # visual cell geom ids, found on first draw
        self.c = np.zeros(n_points)
        self.alpha = np.zeros(n_points)
        self.f_yield = np.zeros(n_points)
        self.mu = np.zeros(n_points)
        self.surface_z = np.zeros(n_points)      # dent depth (negative = dented)
        self.last_normal = 0.0

    # ------------------------------------------------------------ sampling
    def sample(self, rng):
        """Draw a new surface. `rng` must be the env's seeded np_random."""
        r = GROUND_RANGES[self.split]

        bands = r["k0"]                          # may be several disjoint bands
        pick = rng.integers(0, len(bands), self.n)
        self.k0 = np.array([
            10 ** rng.uniform(np.log10(bands[p][0]), np.log10(bands[p][1]))
            for p in pick
        ])

        # Sample the damping RATIO, then derive c from it, so damping always
        # scales with the stiffness just drawn.
        self.zeta = rng.uniform(*r["zeta"], self.n)
        self.c = 2.0 * self.zeta * np.sqrt(self.k0 * self.total_mass)
        
        self.alpha = rng.uniform(*r["alpha"], self.n)
        self.f_yield = rng.uniform(*r["f_yield"], self.n)
        self.mu = rng.uniform(*r["mu"], self.n)

        self.surface_z = np.zeros(self.n)
        self.last_normal = 0.0

    # -------------------------------------------------------------- forces
    def forces(self, positions, velocities):
        """World forces at each contact site.

        positions  : (n, 3) world positions of the contact sites
        velocities : (n, 3) world velocities of the contact sites
        returns    : (n, 3) world forces
        """
        out = np.zeros((self.n, 3))
        total = 0.0

        self.last_x = np.asarray(positions, dtype=float)[:, 0].copy()

        for i in range(self.n):
            depth = self.surface_z[i] - positions[i][2]
            Fn, extra_dent = normal_force(
                depth, -velocities[i][2],
                self.k0[i], self.c[i], self.alpha[i], self.f_yield[i],
            )
            if extra_dent is not None:
                self.surface_z[i] -= extra_dent          # permanent deformation
            if Fn <= 0.0:
                continue

            out[i] = [friction_force(Fn, velocities[i][0], self.mu[i]), 0.0, Fn]
            total += Fn

        self.last_normal = total
        return out

    def update_visual(self, model):
        """Show the dent.

        MuJoCo's floor collision is off, so without this the leg appears to
        stand on nothing and sink into empty space.

        model/leg.xml provides geoms named ground_cell*, drawn as a cutaway
        strip in front of the leg. Each cell under a contact point drops to that
        point's dented surface, so heel and toe dent separately. The strip
        follows the foot, snapped to a fixed grid so it does not shimmer.

        Note: MuJoCo never updates a plane's position at runtime, so moving the
        floor plane (the previous approach) had no visible effect. If the cells
        are missing, nothing is drawn.
        """
        if self._cells is None:
            self._cells = [i for i in range(model.ngeom)
                           if (model.geom(i).name or "").startswith("ground_cell")]
        if not self._cells:
            return

        n = len(self._cells)
        w = 2.0 * model.geom_size[self._cells[0], 0]       # cell width
        # Centre the strip on the foot, snapped to a fixed world grid so the
        # cells do not shimmer as the foot moves.
        centre = np.round(self.last_x.mean() / w) * w
        xs = centre + (np.arange(n) - (n - 1) / 2.0) * w
        for gid, x in zip(self._cells, xs):
            top = 0.0
            for i in range(self.n):
                if abs(x - self.last_x[i]) <= PATCH_HALF:
                    top = min(top, self.surface_z[i])
            half_h = model.geom_size[gid, 2]
            model.geom_pos[gid, 0] = x
            model.geom_pos[gid, 2] = top - half_h

    # ------------------------------------------------------------- logging
    def describe(self):
        return {
            "k0": self.k0.round(0),
            "mu": self.mu.round(2),
            "dent_mm": (-self.surface_z * 1000).round(1),
        }
