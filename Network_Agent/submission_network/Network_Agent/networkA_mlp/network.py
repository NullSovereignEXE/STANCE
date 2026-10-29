"""Network A: a feed-forward MLP that reads a fixed window of recent sensor frames. OWNER: P1.

This file defines the network only. It was trained with PPO; demo.py loads the trained weights.
The observation is the last 10 frames of 11 sensors; `wrappers()` keeps the newest
`frames` frames and drops the sensor columns in `drop_sensors`. The actor and the critic
each get their own HistoryMLP feature extractor followed by a small head (`head`).
"""
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from common.policies import SB3Policy

import gymnasium as gym
import numpy as np

# Network configuration
CONFIG = {
    "frames": 10,
    "drop_sensors": [],
    "hidden": [256, 256],
    "activation": "tanh",
    "layer_norm": False,
    "head": [64],
    "log_std_init": 0.0,
}

NAME = "networkA_mlp"
ALGO = PPO
POLICY_NAME = "MlpPolicy"
POLICY = SB3Policy


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}

class SelectObservations(gym.ObservationWrapper):
    """Keep only the newest `frames` frames and drop the sensor columns listed in `drop_sensors`."""

    def __init__(self, env, frames=10, drop_sensors=None):

        super().__init__(env)

        self.frames = frames
        self.drop_sensors = drop_sensors or []

        self.total_sensors = 11
        self.total_frames = (
            env.observation_space.shape[0] // self.total_sensors
        )

        if not 1 <= self.frames <= self.total_frames:
            raise ValueError("Invalid history length")

        if any(i not in range(11) for i in self.drop_sensors):
            raise ValueError("Invalid sensor index")

        self.keep = [
            i for i in range(11)
            if i not in self.drop_sensors
        ]

        if not self.keep:
            raise ValueError("At least one sensor must remain")

        low = self._select(env.observation_space.low)
        high = self._select(env.observation_space.high)

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

    def _select(self, observation):
        """Reshape the flat history to (frames, sensors), cut it, and flatten it again."""

        history = np.asarray(observation).reshape(
            self.total_frames,
            self.total_sensors
        )

        selected = history[-self.frames:, self.keep]

        return selected.flatten().astype(np.float32)

    def observation(self, observation):
        """Gymnasium hook: apply the cut to every observation."""

        return self._select(observation)


def wrappers(cfg):
    """Environment wrappers for this network: select the frames and sensors it sees."""

    return [
        lambda env: SelectObservations(
            env,
            frames=cfg["frames"],
            drop_sensors=cfg["drop_sensors"]
        )
    ]

class HistoryMLP(BaseFeaturesExtractor):
    """Fully connected layers (optionally with LayerNorm) that turn the frame window into features."""

    def __init__(
        self,
        observation_space,
        hidden=(256, 256),
        activation="tanh",
        layer_norm=False
    ):

        # Output dimension of our feature extractor
        super().__init__(
            observation_space,
            features_dim=hidden[-1]
        )

        layers = []

        # Number of sensor inputs
        input_dim = observation_space.shape[0]

        # Build hidden layers
        for neurons in hidden:

            layers.append(
                nn.Linear(input_dim, neurons)
            )

            if layer_norm:
                layers.append(nn.LayerNorm(neurons))

            layers.append(
                ACTIVATIONS[activation]()
            )

            input_dim = neurons

        self.network = nn.Sequential(*layers)

    def forward(self, observations):
        """Return the features for a batch of observations."""

        return self.network(observations)

def policy_kwargs(cfg):
    """Keyword arguments that tell SB3 how to build the actor and critic from CONFIG."""

    return {
        "features_extractor_class": HistoryMLP,

        "features_extractor_kwargs": {
            "hidden": tuple(cfg["hidden"]),
            "activation": cfg["activation"],
            "layer_norm": cfg["layer_norm"],
        },

        "share_features_extractor": False,

        "net_arch": {
            "pi": list(cfg["head"]),
            "vf": list(cfg["head"]),
        },

        "log_std_init": cfg["log_std_init"],

        "ortho_init": True,
    }