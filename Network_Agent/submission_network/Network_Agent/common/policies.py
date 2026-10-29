# The evaluator needs to run every controller in the same way: call reset() at the start of an
# episode, then act(obs) at every step. Each class below wraps one kind of controller so it
# offers exactly those two methods.
#
# There are only three classes because the project only compares three controllers:
#   FixedPolicy          the reference action that both networks are compared against
#   SB3Policy            Network A (MLP), where each action depends only on the current observation
#   RecurrentSB3Policy   Network B (LSTM), which also has to carry its memory from step to step
#
# Another type of controller would only need its own class with the same two methods.

import numpy as np


class FixedPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def reset(self):
        pass

    def act(self, obs):
        return self.action


class SB3Policy:
    def __init__(self, model, vecnorm=None):
        self.model, self.vecnorm = model, vecnorm

    def reset(self):
        pass

    def _normalise(self, obs):
        return obs if self.vecnorm is None else self.vecnorm.normalize_obs(obs)

    def act(self, obs):
        action, _ = self.model.predict(self._normalise(obs), deterministic=True)
        return action


class RecurrentSB3Policy(SB3Policy):
    def __init__(self, model, vecnorm=None):
        super().__init__(model, vecnorm)
        self.reset()

    def reset(self):
        self.state = None
        self.start = np.ones((1,), dtype=bool)

    def act(self, obs):
        action, self.state = self.model.predict(
            self._normalise(obs), state=self.state, episode_start=self.start, deterministic=True)
        self.start = np.zeros((1,), dtype=bool)
        return action
