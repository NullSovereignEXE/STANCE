"""
Leg model structure checks -- owned by Person 1.

These guard the things that silently break when leg.xml is edited: the DOF
count your proposal's underactuation argument rests on, the site offsets the
reset geometry depends on, and the actuator limits.

    pytest tests/test_leg.py -v
"""
import numpy as np
import mujoco
import pytest

from stance_env.ankle_env import (
    MODEL_PATH, HEEL_OFFSET, TOE_OFFSET, ANKLE_RANGE, TAU_LIMIT,
)


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def test_five_dof_one_motor(model):
    """THE underactuation claim in the proposal. If this ever fails, the
    'five degrees of freedom, one motor' argument is no longer true."""
    assert model.nv == 5, f"expected 5 DOF, got {model.nv}"
    assert model.nu == 1, f"expected 1 actuator, got {model.nu}"


def test_joint_order_matches_env(model):
    """ankle_env indexes qpos positionally. Reordering joints in the XML
    without updating those indices would silently scramble the observation."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(model.njnt)]
    assert names == ["foot_x", "foot_z", "foot_pitch", "ankle_hinge", "pylon_slide"]


def test_site_offsets_match_constants(model):
    """HEEL_OFFSET / TOE_OFFSET in ankle_env must match the XML, or the
    reset geometry places the leg in the wrong place every episode."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)

    foot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot")
    foot_pos = data.xpos[foot_bid]

    for name, expected in (("heel_site", HEEL_OFFSET), ("toe_site", TOE_OFFSET)):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        rel = data.site_xpos[sid] - foot_pos
        assert abs(rel[0] - expected[0]) < 1e-9, f"{name} x offset"
        assert abs(rel[2] - expected[1]) < 1e-9, f"{name} z offset"


def test_heel_touches_ground_at_zero_pose(model):
    """At the default pose both contact points sit exactly on z = 0. This is
    the sanity check behind the reset formula."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name in ("heel_site", "toe_site"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        assert abs(data.site_xpos[sid][2]) < 1e-9


def test_actuator_limits(model):
    """Torque ceiling must match TAU_LIMIT -- the impedance law clips to it."""
    lo, hi = model.actuator_ctrlrange[0]
    assert abs(hi - TAU_LIMIT) < 1e-6 and abs(lo + TAU_LIMIT) < 1e-6


def test_ankle_range(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_hinge")
    lo, hi = model.jnt_range[jid]
    assert abs(lo - ANKLE_RANGE[0]) < 1e-3 and abs(hi - ANKLE_RANGE[1]) < 1e-3


def test_nothing_collides(model):
    """MuJoCo's contact solver is disabled -- every geom must have
    contype = conaffinity = 0, or MuJoCo will add forces on top of ours."""
    assert np.all(model.geom_contype == 0), "a geom still has contype != 0"
    assert np.all(model.geom_conaffinity == 0), "a geom still has conaffinity != 0"


def test_mass_is_physical(model):
    """Total mass should be a plausible human, not a typo."""
    total = model.body_subtreemass[1]
    assert 60.0 < total < 90.0, f"total mass {total:.1f} kg is implausible"


def test_falls_under_gravity(model):
    """With no forces at all the leg must fall -- nothing anchors it to the
    world. If it hangs in the air, a joint is over-constrained."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)      # xpos is stale until this runs
    z0 = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_mass")][2]
    for _ in range(500):
        mujoco.mj_step(model, data)
    z1 = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_mass")][2]
    assert z1 < z0 - 0.5, f"body only fell {z0 - z1:.3f} m in 0.5 s"
