# Finds the STANCE environment, builds it, and converts actions between real units
# (degrees, stiffness, damping) and the network's -1 to 1 range.
#
# The flow:
# 1. _add_env_path() looks for the Gym/ folder that holds stance_env/ and adds it to Python's path.
# 2. import stance_env then registers the train and test environments with Gymnasium.
# 3. make_env() builds one of them and applies the wrappers it is given.
# 4. to_norm_action() and to_physical() convert actions between real units and -1 to 1.

import os
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym

TRAIN_ID = "StanceAnkle-v0"
TEST_ID = "StanceAnkleTest-v0"
N_SENSORS = 11
SENSOR_NAMES = [
    "ankle_angle", "ankle_rate", "ankle_torque", "vertical_grf", "pylon_compression",
    "shank_tilt", "shank_tilt_rate", "vertical_accel",
    "prev_theta_d", "prev_K", "prev_B",
]


# ---------- Find the environment ----------
def _add_env_path():
    # Put the folder containing stance_env/ on sys.path: $STANCE_GYM_PATH, else a parent Gym/.
    candidates = [os.environ.get("STANCE_GYM_PATH")] + [p / "Gym" for p in Path(__file__).resolve().parents]
    for c in candidates:
        if c and (Path(c) / "stance_env" / "__init__.py").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return


_add_env_path()
import stance_env  # noqa: E402,F401  (importing registers the environment ids)


# ---------- Build the environment ----------
def make_env(split="train", wrappers=()):
    # Build one environment on the train or test ground, then apply each wrapper function in order.
    env = gym.make(TRAIN_ID if split == "train" else TEST_ID)
    for wrap in wrappers:
        env = wrap(env)
    return env


# ---------- Convert actions ----------
def _consts(env):
    # Return (ANKLE_RANGE, K_RANGE, B_RANGE) from the module that defines the env class.
    module = sys.modules[type(env.unwrapped).__module__]
    return module.ANKLE_RANGE, module.K_RANGE, module.B_RANGE


def _to_unit(x, lo, hi):
    return 2 * (x - lo) / (hi - lo) - 1


def _from_unit(u, lo, hi):
    return lo + (u + 1) / 2 * (hi - lo)


def to_norm_action(env, theta_deg, K, B):
    # Physical action (degrees, N.m/rad, N.m.s/rad) to the normalised [-1, 1] action.
    (a_lo, a_hi), (k_lo, k_hi), (b_lo, b_hi) = _consts(env)
    a = [_to_unit(np.deg2rad(theta_deg), a_lo, a_hi),
         _to_unit(np.log10(K), np.log10(k_lo), np.log10(k_hi)),
         _to_unit(np.log10(B), np.log10(b_lo), np.log10(b_hi))]
    return np.clip(np.array(a, dtype=np.float32), -1, 1)


def to_physical(env, a):
    # Normalised [-1, 1] action to (theta_d in degrees, K, B). Inverse of to_norm_action.
    (a_lo, a_hi), (k_lo, k_hi), (b_lo, b_hi) = _consts(env)
    a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
    return (float(np.rad2deg(_from_unit(a[0], a_lo, a_hi))),
            float(10 ** _from_unit(a[1], np.log10(k_lo), np.log10(k_hi))),
            float(10 ** _from_unit(a[2], np.log10(b_lo), np.log10(b_hi))))
