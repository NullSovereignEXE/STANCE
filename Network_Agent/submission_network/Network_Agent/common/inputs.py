"""Chooses which part of the observation a network sees. Shared and frozen.

The environment gives one flat vector of 110 numbers: 10 frames of 11 sensor
values, oldest frame first (see SENSOR_NAMES in env_setup.py for the 11 channels).
`SelectInputs` reshapes that vector to (frames, sensors), keeps the newest
`frames` frames, removes the sensor columns listed in `drop`, and flattens it again.

Both networks use it through their `wrappers(cfg)` function, controlled by the
config keys `frames` and `drop_sensors`:
    frames=10, drop=[]       the full observation (Network A default)
    frames=1                 newest frame only (Network B default; its memory is the LSTM)
    drop=[7]                 remove one channel from every frame (e.g. vertical acceleration)

Uses: gymnasium.ObservationWrapper, numpy.
"""
import gymnasium as gym
import numpy as np

from STANCE.Network_Agent.submission_network.Network_Agent.common.env_setup import N_SENSORS


class SelectInputs(gym.ObservationWrapper):
    """Keep the newest `frames` frames and all sensor columns except those in `drop`.

    frames: number of frames to keep (1 to the environment's history length). None keeps all.
    drop: sensor indices (0-10) to remove from every frame.
    The new observation space is the old bounds reshaped and cut the same way, so the
    wrapped environment still passes Gymnasium's checks.
    """

    def __init__(self, env, frames=None, drop=()):
        super().__init__(env)
        self.total = env.observation_space.shape[0] // N_SENSORS
        self.frames = self.total if frames is None else int(frames)
        if not 1 <= self.frames <= self.total:
            raise ValueError(f"frames must be between 1 and {self.total}, got {frames}")
        bad = [d for d in drop if not 0 <= d < N_SENSORS]
        if bad:
            raise ValueError(f"drop_sensors must be between 0 and {N_SENSORS - 1}, got {bad}")
        self.keep = [i for i in range(N_SENSORS) if i not in set(drop)]
        if not self.keep:
            raise ValueError("drop_sensors removes every sensor")
        low = self._cut(env.observation_space.low)
        high = self._cut(env.observation_space.high)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _cut(self, flat):
        """Reshape to (frames, sensors), keep the newest frames and chosen columns, flatten."""
        grid = np.asarray(flat).reshape(self.total, N_SENSORS)
        return grid[-self.frames:, self.keep].reshape(-1).astype(np.float32)

    def observation(self, obs):
        """Apply the same cut to a real observation."""
        return self._cut(obs)
