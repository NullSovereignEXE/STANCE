import gymnasium as gym
import numpy as np

from common.env_setup import to_norm_action


class ResidualAction(gym.ActionWrapper):
    """
    Maps the policy's action to anchor + scale * action, clipped to [-1, 1].
    anchor: (degrees, N.m/rad, N.m.s/rad), converted once to the normalised action.
    scale: largest offset per dimension, in normalised units. The default 0.3 is 0.3 normalised units, which is 15% of the full range (log units for K and B).
    """

    def __init__(self, env, anchor=(-20.0, 500.0, 10.0), scale=(0.3, 0.3, 0.3)):
        super().__init__(env)
        self.anchor = to_norm_action(env, *anchor)
        self.scale = np.asarray(scale, dtype=np.float32)

    def action(self, a):
        return np.clip(self.anchor + self.scale * np.asarray(a, dtype=np.float32), -1.0, 1.0)
