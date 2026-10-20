"""The residual action wrapper. OWNER: P3.

Without it, the policy's output is the environment's action directly: three numbers in
[-1, 1] for the target angle, the stiffness K and the damping B. With `residual: True`
in config.py, the policy instead outputs an OFFSET from a fixed anchor action:

    env_action = clip(anchor + scale * policy_action, -1, 1)

An untrained policy then outputs roughly zero, so it already behaves like the anchor
(by default the best fixed action, -20 degrees / K 500 / B 10, which survives about
92.5% of episodes) and learning only has to find improvements.

Known problem to solve (P3)
    The default anchor sits at the edge of every range: -20 degrees is the minimum, and
    K 500 and B 10 are the maximums. The final `clip` cuts off every offset that points
    outward, so half of each action dimension does nothing. Options: map the output
    only inward from the anchor, move the anchor inside the range, or show with data
    that the clipping is harmless.

Uses: gymnasium.ActionWrapper and common.env_setup.to_norm_action, which converts the
anchor from physical units to the normalised action.
"""
import gymnasium as gym
import numpy as np

from common.env_setup import to_norm_action


class ResidualAction(gym.ActionWrapper):
    """Maps the policy's action to anchor + scale * action, clipped to [-1, 1].

    anchor: (degrees, N.m/rad, N.m.s/rad), converted once to the normalised action.
    scale: largest offset per dimension, in normalised units.
    """

    def __init__(self, env, anchor=(-20.0, 500.0, 10.0), scale=(0.3, 0.3, 0.3)):
        super().__init__(env)
        self.anchor = to_norm_action(env, *anchor)
        self.scale = np.asarray(scale, dtype=np.float32)

    def action(self, a):
        """Return the environment action for the policy's action `a`."""
        return np.clip(self.anchor + self.scale * np.asarray(a, dtype=np.float32), -1.0, 1.0)
