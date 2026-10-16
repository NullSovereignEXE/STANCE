"""Finds the STANCE environment and converts actions. Shared and frozen.

This is the only file that knows where the environment code lives. Everything
else calls `make_env()`.

How it works
    1. `_add_env_path()` looks for a folder that contains `stance_env/`, first in
       the environment variable STANCE_GYM_PATH, then in a `Gym/` folder next to
       (or above) this project.
    2. `import stance_env` runs the environment package, which registers the
       Gymnasium ids below.
    3. `make_env()` builds the environment with `gym.make`.

Uses: gymnasium, numpy, and the constants ANKLE_RANGE, K_RANGE and B_RANGE that
the environment module defines. Those constants are read from the environment
itself, so the action conversions stay correct if the ranges change.

If the ids or constants in your environment are named differently, edit
TRAIN_ID / TEST_ID below or `_consts()`. This is a day-one fix, done once by P3.
"""
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


def _add_env_path():
    """Put the folder containing `stance_env/` on sys.path.

    Tries $STANCE_GYM_PATH first, then `Gym/` in this file's parent folders.
    Does nothing if neither exists, in which case an installed `stance_env` is used.
    """
    candidates = []
    if os.environ.get("STANCE_GYM_PATH"):
        candidates.append(Path(os.environ["STANCE_GYM_PATH"]))
    candidates += [p / "Gym" for p in Path(__file__).resolve().parents]
    for c in candidates:
        if (c / "stance_env" / "__init__.py").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return


_add_env_path()
import stance_env  # noqa: E402,F401  (importing registers the environment ids)


def make_env(split="train", wrappers=()):
    """Build one environment.

    split: "train" or "test" (the held-out ground).
    wrappers: functions that each take an env and return a wrapped env, applied in
        order. The same list must be used for training and evaluation.
    """
    env = gym.make(TRAIN_ID if split == "train" else TEST_ID)
    for wrap in wrappers:
        env = wrap(env)
    return env


def _consts(env):
    """Return (ANKLE_RANGE, K_RANGE, B_RANGE) from the module that defines the env class."""
    module = sys.modules[type(env.unwrapped).__module__]
    try:
        return module.ANKLE_RANGE, module.K_RANGE, module.B_RANGE
    except AttributeError as err:
        raise RuntimeError(
            f"{module.__name__} has no ANKLE_RANGE / K_RANGE / B_RANGE; "
            "edit _consts() in common/env_setup.py to read them from where they live") from err


def to_norm_action(env, theta_deg, K, B):
    """Physical action (degrees, N.m/rad, N.m.s/rad) to the normalised [-1, 1] action.

    Mirrors the environment's own mapping: theta is linear, K and B are log-spaced.
    Used for the fixed reference action and the residual anchor.
    """
    (a_lo, a_hi), k_range, b_range = _consts(env)

    def inverse_log(r, x):
        lo, hi = np.log10(r[0]), np.log10(r[1])
        return 2 * (np.log10(x) - lo) / (hi - lo) - 1

    a = [2 * (np.deg2rad(theta_deg) - a_lo) / (a_hi - a_lo) - 1,
         inverse_log(k_range, K), inverse_log(b_range, B)]
    return np.clip(np.array(a, dtype=np.float32), -1, 1)


def to_physical(env, a):
    """Normalised [-1, 1] action to (theta_d in degrees, K, B). Inverse of `to_norm_action`.

    Used by the evaluator to report what stiffness and angle a policy actually applied.
    """
    (a_lo, a_hi), k_range, b_range = _consts(env)
    a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)

    def log_span(r, x):
        lo, hi = np.log10(r[0]), np.log10(r[1])
        return 10 ** (lo + (x + 1) / 2 * (hi - lo))

    theta = a_lo + (a[0] + 1) / 2 * (a_hi - a_lo)
    return float(np.rad2deg(theta)), float(log_span(k_range, a[1])), float(log_span(b_range, a[2]))
