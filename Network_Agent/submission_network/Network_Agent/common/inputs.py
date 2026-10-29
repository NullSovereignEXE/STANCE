# This file decides which part of the observation a network gets to see. The environment gives
# 10 frames of 11 sensor readings, and each network can keep fewer frames or drop some sensors.

import gymnasium as gym
import numpy as np

from common.env_setup import N_SENSORS


class SelectInputs(gym.ObservationWrapper):
    # Built on Gymnasium's ObservationWrapper. Which frames and sensors to keep is our own choice.
    # This wrapper keeps the newest `frames` frames, and every sensor except the ones listed in `drop`.

    def __init__(self, env, frames=None, drop=()):
        # This works out which frames and sensors to keep, checks that the settings make sense,
        # and tells Gymnasium what shape the trimmed observation has.
        super().__init__(env)

        self.total = env.observation_space.shape[0] // N_SENSORS
        # If frames is not given we keep the whole history, which is what Network A uses.
        self.frames = self.total if frames is None else int(frames)

        self.keep = [i for i in range(N_SENSORS) if i not in drop]
        # We check frames ourselves because frames=0 would not raise an error. It would quietly
        # keep every frame instead, and we would never notice.
        if not 1 <= self.frames <= self.total:
            raise ValueError(f"frames must be between 1 and {self.total}, got {frames}")
        if not self.keep or any(not 0 <= d < N_SENSORS for d in drop):
            raise ValueError(f"drop must be sensor indices 0-{N_SENSORS - 1} and leave one sensor")

        space = env.observation_space
        # The min and max values go through observation() as well, so the declared space
        # always matches the observations we actually return.
        self.observation_space = gym.spaces.Box(
            self.observation(space.low), self.observation(space.high), dtype=np.float32)

    def observation(self, obs):
        # The frames are stored oldest first, so the newest ones are the last rows.
        grid = np.asarray(obs).reshape(self.total, N_SENSORS)
        return grid[-self.frames:, self.keep].reshape(-1).astype(np.float32)
