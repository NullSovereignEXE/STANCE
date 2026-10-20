"""Network A: a feed-forward network on the stacked sensor history. OWNER: P1.

This file defines the network only. The agent (agent/train.py) trains it.

Inputs and preprocessing
    The environment's observation is 10 frames of 11 sensor values, flattened to 110
    numbers, oldest frame first. `wrappers()` uses common.inputs.SelectInputs to keep the
    newest `frames` frames and to remove the sensor columns in `drop_sensors`. The agent
    then normalises what is left with running statistics (VecNormalize). The 11 channels
    are listed in common/env_setup.py (SENSOR_NAMES).

Data flow
    observation -> HistoryMLP (hidden layers) -> head layers -> output.
    The actor and the critic have separate copies of every layer, so the value estimate
    cannot disturb the policy's features. `policy_kwargs()` assembles this.

Outputs
    Actor: three means in [-1, 1] (target angle, stiffness K, damping B; the environment
    maps them to physical units) plus one learned spread (log standard deviation) per
    output, initialised from `log_std_init`. Actions during training are sampled around
    the means; evaluation uses the means.
    Critic: one number, the estimated value of the state.

Losses and gradients
    Not defined in this file. PPO, run by the agent, computes them: the clipped policy
    loss, the value-error loss and an entropy bonus, weighted as set in agent/config.py,
    then clips the gradients and applies them with Adam. Describe them in Report 2.

Uses: stable_baselines3 (PPO, BaseFeaturesExtractor), torch, common.inputs, common.policies.
"""
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from common.inputs import SelectInputs
from common.policies import SB3Policy

NAME = "networkA_mlp"
ALGO = PPO
POLICY_NAME = "MlpPolicy"
POLICY = SB3Policy
CONFIG = {
    "frames": 10,
    "drop_sensors": [],
    "hidden": [256, 256],
    "activation": "tanh",
    "layer_norm": False,
    "head": [64],
    "log_std_init": 0.0,
}
"""Network settings. frames: newest frames kept (1-10). drop_sensors: channel indices removed
from every frame. hidden: widths of the encoder layers. activation: "tanh", "relu" or "elu".
layer_norm: add LayerNorm after each hidden layer. head: widths of the layers after the encoder.
log_std_init: initial action spread; the standard deviation is exp(this)."""

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


class HistoryMLP(BaseFeaturesExtractor):
    """The encoder: a stack of linear layers, each followed by an activation.

    observation_space: the (already cut) observation. hidden: layer widths, for example
    (256, 256). activation: key of ACTIVATIONS. layer_norm: insert LayerNorm after each
    linear layer. The output has hidden[-1] features and feeds the head layers.
    """

    def __init__(self, observation_space, hidden=(256, 256), activation="tanh", layer_norm=False):
        super().__init__(observation_space, features_dim=hidden[-1])
        layers, width = [], observation_space.shape[0]
        for h in hidden:
            layers.append(nn.Linear(width, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(ACTIVATIONS[activation]())
            width = h
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        """Return the encoded features for a batch of observations."""
        return self.net(obs)


def wrappers(cfg):
    """Environment wrappers for this network: select the frames and sensors it sees."""
    return [lambda env: SelectInputs(env, cfg["frames"], cfg["drop_sensors"])]


def policy_kwargs(cfg):
    """Keyword arguments that tell SB3 how to build the actor and critic layers."""
    return dict(
        features_extractor_class=HistoryMLP,
        features_extractor_kwargs=dict(hidden=tuple(cfg["hidden"]), activation=cfg["activation"],
                                       layer_norm=cfg["layer_norm"]),
        share_features_extractor=False,
        net_arch=dict(pi=list(cfg["head"]), vf=list(cfg["head"])),
        log_std_init=cfg["log_std_init"],
        ortho_init=True,
    )
