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

        history = np.asarray(observation).reshape(
            self.total_frames,
            self.total_sensors
        )

        selected = history[-self.frames:, self.keep]

        return selected.flatten().astype(np.float32)

    def observation(self, observation):

        return self._select(observation)


def wrappers(cfg):

    return [
        lambda env: SelectObservations(
            env,
            frames=cfg["frames"],
            drop_sensors=cfg["drop_sensors"]
        )
    ]

class HistoryMLP(BaseFeaturesExtractor):

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

        return self.network(observations)

def policy_kwargs(cfg):

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