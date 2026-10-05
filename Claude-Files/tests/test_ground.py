"""
Closed-form checks on the ground model -- the "verified against closed-form
results" claim in the proposal. Run after every change to the force law:

    pytest tests/test_ground.py -v
"""
import numpy as np
import mujoco
import pytest

from stance_env.ground import normal_force, friction_force, GroundModel, D_REF, GROUND_RANGES


G = 9.81


# ============================================================ pure force law
def test_linear_spring_exact():
    """alpha = 0, no sink rate  ->  F must equal k*d exactly."""
    f, _ = normal_force(0.005, 0.0, k0=5e4, c=100, alpha=0.0, f_yield=1e9)
    assert abs(f - 5e4 * 0.005) < 1e-9


def test_no_contact_no_force():
    """Above the surface, nothing happens."""
    f, _ = normal_force(-0.01, 0.0, 5e4, 100, 0.0, 1e9)
    assert f == 0.0


def test_ground_never_pulls():
    """Rising fast out of a shallow dent must not suck the foot down."""
    f, _ = normal_force(0.001, -5.0, 5e4, 400, 0.0, 1e9)
    assert f >= 0.0


def test_stiffening_with_depth():
    """k(d) = k0 (1 + alpha d / D_REF): at d = D_REF with alpha = 1, k doubles."""
    f, _ = normal_force(D_REF, 0.0, k0=1e4, c=0, alpha=1.0, f_yield=1e9)
    assert abs(f - 2.0 * 1e4 * D_REF) < 1e-9


def test_force_saturates_at_yield():
    """Past yield the force is capped and the surface reports extra denting."""
    f, dent = normal_force(0.05, 0.0, k0=5e4, c=0, alpha=0.0, f_yield=300.0)
    assert abs(f - 300.0) < 1e-9
    assert dent is not None and dent > 0.0


def test_friction_opposes_motion():
    """Sliding forward -> friction backward, and never above mu * Fn."""
    assert friction_force(500.0, +0.5, 0.8) < 0
    assert friction_force(500.0, -0.5, 0.8) > 0
    assert abs(friction_force(500.0, 1.0, 0.8)) <= 0.8 * 500.0 + 1e-9


# ================================================ dynamics on a minimal model
# The full leg is an inverted pendulum: it TOPPLES during a drop test, which
# invalidates any 1-DOF closed form. So the dynamics checks use a single mass
# on a vertical slide, where mg/k and sqrt(k/m) hold exactly.
MINIMAL = """
<mujoco><option timestep="0.001" gravity="0 0 -9.81" integrator="Euler"/>
<worldbody><body name="m" pos="0 0 0.2">
  <joint name="z" type="slide" axis="0 0 1"/>
  <geom type="sphere" size="0.05" mass="50.0" contype="0" conaffinity="0"/>
  <site name="p" pos="0 0 0"/>
</body></worldbody></mujoco>
"""
MASS = 50.0


def _drop(k, c, steps=6000, z0=None):
    m = mujoco.MjModel.from_xml_string(MINIMAL)
    d = mujoco.MjData(m)
    if z0 is not None:
        d.qpos[0] = z0 - 0.2                 # body origin sits at z = 0.2
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "p")
    hist = []
    for _ in range(steps):
        z, vz = d.site_xpos[sid][2], d.qvel[0]
        f, _ = normal_force(-z, -vz, k0=k, c=c, alpha=0.0, f_yield=1e9)
        d.qfrc_applied[0] = f
        mujoco.mj_step(m, d)
        d.qfrc_applied[0] = 0.0
        hist.append(d.site_xpos[sid][2])
    return np.array(hist)


def test_static_settling_depth():
    """Mass must settle at exactly mg/k."""
    k = 5e4
    h = _drop(k, c=800.0)
    expected = MASS * G / k
    settled = -h[-500:].mean()
    assert abs(settled - expected) < 0.02 * expected, \
        f"settled {settled*1000:.3f} mm, expected {expected*1000:.3f} mm"


def test_oscillation_frequency():
    """Lightly damped ring must be at sqrt(k/m) / 2pi Hz."""
    k = 5e4
    # Start 3 mm BELOW equilibrium so it rings while staying in contact.
    # Dropping from height makes it bounce clear, a different oscillation.
    h = _drop(k, c=5.0, steps=3000, z0=-(MASS * G / k) - 0.003)
    expected_hz = np.sqrt(k / MASS) / (2 * np.pi)
    sig = h - h.mean()
    freqs = np.fft.rfftfreq(len(sig), d=0.001)
    peak = freqs[np.argmax(np.abs(np.fft.rfft(sig))[1:]) + 1]
    assert abs(peak - expected_hz) < 0.05 * expected_hz, \
        f"peak {peak:.2f} Hz, expected {expected_hz:.2f} Hz"


def test_plastic_dent_persists():
    """Load past yield, unload: the surface must stay dented."""
    surface = 0.0
    for depth in (0.02, 0.02, 0.02):
        f, extra = normal_force(depth - surface, 0.0, 2e4, 0.0, 0.0, 200.0)
        if extra is not None:
            surface -= extra
    assert surface < -1e-4, f"no permanent dent formed: {surface}"


def test_dent_is_never_negative():
    """Fast sinking at shallow depth can push force past yield while the spring
    term is still below it. The surface must not RISE as a result."""
    for rate in (0.0, 1.0, 5.0, 20.0):
        f, extra = normal_force(0.001, rate, k0=5e4, c=400, alpha=0.0, f_yield=200.0)
        if extra is not None:
            assert extra >= 0.0, f"surface rose by {extra*1000:.2f} mm at rate {rate}"


def test_ground_model_dent_only_deepens():
    """Across a whole episode, surface_z must be monotonically non-increasing."""
    g = GroundModel(split="train", n_points=2)
    g.sample(np.random.default_rng(0))
    prev = g.surface_z.copy()
    for i in range(300):
        pos = np.array([[0.0, 0.0, -0.001 * (i % 30)], [0.0, 0.0, 0.0]])
        vel = np.array([[0.2, 0.0, -3.0], [0.2, 0.0, -3.0]])
        g.forces(pos, vel)
        assert np.all(g.surface_z <= prev + 1e-12), "surface rose"
        prev = g.surface_z.copy()

# ================================================== material range design
def _bands(value):
    """k0 is a list of (lo, hi) bands; every other parameter is one (lo, hi)."""
    return value if isinstance(value, list) else [value]


@pytest.mark.parametrize("param", ["k0", "c", "alpha", "f_yield", "mu"])
def test_train_and_test_ranges_are_disjoint(param):
    """No test band may overlap any train band (touching at an edge is allowed).

    This is the anti-memorisation guarantee: the agent is evaluated only on
    grounds outside everything it trained on.
    """
    train = _bands(GROUND_RANGES["train"][param])
    test = _bands(GROUND_RANGES["test"][param])
    for tr_lo, tr_hi in train:
        for te_lo, te_hi in test:
            assert te_hi <= tr_lo or te_lo >= tr_hi, (
                f"{param}: test band ({te_lo}, {te_hi}) overlaps "
                f"train band ({tr_lo}, {tr_hi})")
