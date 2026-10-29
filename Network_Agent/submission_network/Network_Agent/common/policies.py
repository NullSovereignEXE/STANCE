"""Adapters that give every policy the same two-method interface. Shared and frozen.

The evaluator (evaluate.py) only needs `reset()` at the start of an episode and
`act(obs)` once per step. Each adapter wraps one kind of policy to provide those:

    FixedPolicy          always returns the same action (the reference controller)
    SB3Policy            a feed-forward Stable-Baselines3 model (PPO)
    RecurrentSB3Policy   a recurrent model (sb3-contrib RecurrentPPO); it carries the
                         LSTM state through the episode and clears it on `reset()`

SB3 models were trained on normalised observations, so the adapters apply the
saved VecNormalize statistics before predicting. Predictions are deterministic,
meaning the mean of the action distribution with no exploration noise.
"""
import numpy as np


class FixedPolicy:
    """Always returns `action` (normalised, shape (3,))."""

    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def reset(self):
        """Nothing to clear."""

    def act(self, obs):
        """Ignore the observation and return the fixed action."""
        return self.action


class SB3Policy:
    """Wraps a feed-forward SB3 model.

    model: a trained Stable-Baselines3 model.
    vecnorm: the VecNormalize object holding the observation statistics, or None.
    """

    def __init__(self, model, vecnorm=None):
        self.model, self.vecnorm = model, vecnorm

    def reset(self):
        """Nothing to clear."""

    def act(self, obs):
        """Normalise the observation, then return the model's deterministic action."""
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class RecurrentSB3Policy(SB3Policy):
    """Wraps a recurrent SB3 model and carries its LSTM state between steps."""

    def __init__(self, model, vecnorm=None):
        super().__init__(model, vecnorm)
        self.reset()

    def reset(self):
        """Clear the LSTM state; call at the start of every episode."""
        self.state = None
        self.start = np.ones((1,), dtype=bool)

    def act(self, obs):
        """Return the deterministic action and update the LSTM state."""
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        action, self.state = self.model.predict(
            obs, state=self.state, episode_start=self.start, deterministic=True)
        self.start = np.zeros((1,), dtype=bool)
        return action
