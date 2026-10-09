"""
This file describes the surface. 
1. How hard the ground pushes back
2. How a new ground is picked each episode
3. How the dent is drawn

The Gym environment talks to this file through:
    forces = ground.forces(positions, velocities)

positions and velocities are taken to be (n,3) arrays for the contact points.
'n' represents contact points (2 for heel and toe). '3' -> x,y,z 

"""
import numpy as np

# --------------------------------------------------------------- constants
D_REF = 0.02      # m, used for k later. D_REF (2 cm) scales how quickly the stiffness rises with depth.
V_EPS = 1e-3      # m/s, how gently friction switches direction near zero speed

# ===========================================================================
# THE FORCE LAW
# ===========================================================================
def normal_force(depth, sink_rate, k0, c, alpha, f_yield):
    """Upward push of the ground at one contact point.

    depth      : how far the point is below the ground surface, in m.
                 Zero or less means it isn't touching, so no force.
    sink_rate  : how fast that contact point is sinking, in m/s (positive = going down)
    returns    : (force in N, extra permanent dent in m, or None)

    The ground acts like a spring that gets stiffer the deeper we press.
    Once the push would exceed f_yield, the ground gives way: the force stays
    at f_yield and the surface is permanently pushed down a bit more.
    """
    if depth <= 0.0:
        return 0.0, None

    # Spring stiffness grows with depth (alpha = 0 equates to a plain linear spring)
    k = k0 * (1.0 + alpha * depth / D_REF)
    
    # Spring push plus a damping term that resists fast sinking.
    f = max(k * depth + c * sink_rate, 0.0)      # total ground pushback = spring + damping, clamped to be non-negative, since the ground can't "pull"

    if f > f_yield:                              # the ground gives way
        
        elastic_depth = f_yield / k              # depth the spring can hold at f_yield
        
        # Fast sinking can make the damping term alone push us past f_yield
        # while the spring is still shallow. Then depth < elastic_depth and the
        # difference goes negative, which would make the ground rise. Ground
        # can only get dented, never rebuilt, so we stop at zero.
        
        return f_yield, max(depth - elastic_depth, 0.0)
    return f, None


def friction_force(normal, tangential_velocity, mu):
    """Sideways friction at one contact point.

    Classic friction (mu * normal force) flips sign abruptly when the foot
    changes direction, which will make the simulation jitter. tanh makes that flip
    smooth. The minus sign is there because friction always opposes the motion.
    """
    return -mu * normal * np.tanh(tangential_velocity / V_EPS)


# ===========================================================================
# WHAT KINDS OF GROUND WE CAN DRAW
# The test ranges never overlap the training ranges, so the agent can't just
# memorise the grounds it trained on.
# ===========================================================================
GROUND_RANGES = {
    "train": dict(
        k0=[(5e3, 1e5)],                          # base stiffness, N/m
        zeta=(0.05, 0.30),                        # damping ratio (not c itself)
        alpha=(0.0, 2.0),                         # how much it stiffens when pressed
        f_yield=(300.0, 3000.0),                  # N, push at which the ground gives way
        mu=(0.30, 0.90),                          # friction coefficient
    ),
    "test": dict(
        k0=[(2e3, 5e3), (1e5, 2e5)],              # two bands: softer and harder than train
        zeta=(0.30, 0.60),                        # heavier damping, "dead" ground
        alpha=(2.0, 4.0),
        f_yield=(150.0, 300.0),
        mu=(0.15, 0.30),
    ),
}


# ===========================================================================
# THE GROUND MODEL -- this class remembers the episode's ground and its dents.
# ===========================================================================
class GroundModel:
    """One patch of ground, chosen fresh at the start of each episode.

    The heel and the toe each get their own material. If the heel lands on
    softer ground than the toe, it sinks further, the foot tilts, the leg tilts
    with it, and the body is pushed toward the edge of the foot. That mismatch
    is how unknown ground ends up making the leg fall.
    """

    def __init__(self, split="train", n_points=2, total_mass=73.5):
        if split not in GROUND_RANGES:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.split = split
        self.n = n_points                  # contact points (heel and toe)
        # Total mass of the leg model, in kg. Needed to turn a damping ratio
        # into a real damping value: c = 2 * zeta * sqrt(k0 * total_mass).
        # The environment passes the real mass from the loaded model; 73.5 is
        # only the default for standalone use and tests.
        self.total_mass = total_mass

        # Per-point material is filled in by sample(); the rest is state that
        # changes during an episode.
        self.k0 = np.zeros(n_points)
        self.zeta = np.zeros(n_points)
        self.last_x = np.zeros(n_points)   # x of each point at the last step (for drawing)
        self.last_z = np.zeros(n_points)   # z (height) of each point at the last step (for drawing)
        self._cells = None                 # the ground-cell boxes, looked up on first draw
        self._cell_dent = None             # deepest dent each cell has reached so far
        self.c = np.zeros(n_points)
        self.alpha = np.zeros(n_points)
        self.f_yield = np.zeros(n_points)
        self.mu = np.zeros(n_points)
        self.surface_z = np.zeros(n_points)      # dent depth: 0 = untouched, negative = dented
        self.last_normal = 0.0                   # total upward force at the last step

    # ------------------------------------------------------------ sampling
    def sample(self, rng):
        """Pick a new ground. `rng` should be the env's seeded random generator."""
        r = GROUND_RANGES[self.split]

        # Stiffness: choose a band for each point, then a value inside it.
        # Sampling in log space spreads values evenly across orders of
        # magnitude, so soft grounds aren't squeezed out by hard ones.
        bands = r["k0"]
        pick = rng.integers(0, len(bands), self.n)
        self.k0 = np.array([
            10 ** rng.uniform(np.log10(bands[p][0]), np.log10(bands[p][1]))
            for p in pick
        ])

        # Pick the damping ratio first and derive c from it, so the damping
        # always matches the stiffness we just drew.
        self.zeta = rng.uniform(*r["zeta"], self.n)
        self.c = 2.0 * self.zeta * np.sqrt(self.k0 * self.total_mass)

        self.alpha = rng.uniform(*r["alpha"], self.n)
        self.f_yield = rng.uniform(*r["f_yield"], self.n)
        self.mu = rng.uniform(*r["mu"], self.n)

        # Start the episode on untouched ground.
        self.surface_z = np.zeros(self.n)
        self.last_normal = 0.0
        self._cell_dent = None

    # -------------------------------------------------------------- forces
    def forces(self, positions, velocities):
        """Forces the ground applies at each contact point.

        positions  : (n, 3) world positions of the contact points
        velocities : (n, 3) world velocities of the contact points
        returns    : (n, 3) forces, as [sideways, 0, upward] per point
        """
        out = np.zeros((self.n, 3))
        total = 0.0

        # Remember where the points are, so update_visual can draw the dent.
        self.last_x = np.asarray(positions, dtype=float)[:, 0].copy()
        self.last_z = np.asarray(positions, dtype=float)[:, 2].copy()

        for i in range(self.n):
            # How far below the (possibly already dented) surface this point is.
            depth = self.surface_z[i] - positions[i][2]
            Fn, extra_dent = normal_force(
                depth, -velocities[i][2],
                self.k0[i], self.c[i], self.alpha[i], self.f_yield[i],
            )
            if extra_dent is not None:
                self.surface_z[i] -= extra_dent          # the dent stays for good
            if Fn <= 0.0:
                continue                                 # not touching: no force

            out[i] = [friction_force(Fn, velocities[i][0], self.mu[i]), 0.0, Fn]
            total += Fn

        self.last_normal = total       # the env reads this for the observation and reward
        return out

    # -------------------------------------------------------------- drawing
    def update_visual(self, model):
        """Draw the ground sinking under the foot, and the trench it leaves behind.

        MuJoCo's own floor collision is off, so without this the foot would
        look like it sinks into empty air. leg.xml contains a row of small
        boxes called ground_cell*; here we raise or lower each one. If the
        model has none, nothing is drawn.
        """
        if self._cells is None:
            self._cells = [i for i in range(model.ngeom)
                           if (model.geom(i).name or "").startswith("ground_cell")]
        if not self._cells:
            return

        cell_x = model.geom_pos[self._cells, 0]          # where each cell sits along x
        if self._cell_dent is None:
            self._cell_dent = np.zeros(len(self._cells))

        # The sole is a straight line from heel to toe, so for cells in between
        # we estimate the surface height by blending the heel and toe values.
        order = np.argsort(self.last_x)
        xs = self.last_x[order]
        under = (cell_x >= xs[0]) & (cell_x <= xs[-1])   # cells currently under the foot

        # Permanent dent at each cell, and how deep the foot is pressing right now.
        dent_now = np.interp(cell_x, xs, self.surface_z[order])
        sink_now = np.interp(cell_x, xs, np.minimum(self.surface_z, self.last_z)[order])

        # Each cell remembers its deepest dent, so the trench is still there
        # after the foot lifts off.
        self._cell_dent[under] = np.minimum(self._cell_dent[under], dent_now[under])

        # Under the foot: show whichever is lower, the trench or the current sink.
        # Everywhere else: just show the trench.
        top = np.where(under, np.minimum(self._cell_dent, sink_now), self._cell_dent)
        # geom_pos is the centre of a box, so subtract half its height to put its top at `top`.
        model.geom_pos[self._cells, 2] = top - model.geom_size[self._cells, 2]

    # ------------------------------------------------------------- logging
    def describe(self):
        """Short summary of this episode's ground, for the env's `info` output."""
        return {
            "k0": self.k0.round(0),
            "mu": self.mu.round(2),
            "dent_mm": (-self.surface_z * 1000).round(1),
        }