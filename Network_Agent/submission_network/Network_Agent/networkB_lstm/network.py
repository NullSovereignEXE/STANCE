"""Network B: a recurrent (LSTM) network that builds its own memory of the stance. OWNER: P2.

This file defines the network only. It was trained with RecurrentPPO; demo.py loads the trained weights.

Why recurrent
    The ground's stiffness, yield point and dent history never appear in a single sensor
    frame. A recurrent network keeps a hidden state that can accumulate evidence across
    the whole 100-step stance, so it can in principle work out how the ground is behaving.
    Network A instead gives the network a fixed window of the last 10 frames.

Inputs and preprocessing
    `wrappers()` uses common.inputs.SelectInputs. The default keeps only the newest frame
    (`frames: 1`), so all memory is the LSTM's. Raise `frames` to also give it a short
    window. Sensor columns in `drop_sensors` are removed from every frame. The agent then
    normalises the result with running statistics (VecNormalize).

Data flow
    observation -> FrameEncoder (one linear layer) -> LSTM -> head layers -> output.
    The critic has its own LSTM of the same shape (`enable_critic_lstm`). The LSTM state
    is carried from step to step and cleared at the start of every episode.

Outputs
    Actor: three means in [-1, 1] (target angle, stiffness K, damping B; the environment
    maps them to physical units) plus one learned spread per output, initialised from
    `log_std_init`. Critic: one number, the estimated value of the state.

Losses and gradients
    Not defined in this file. RecurrentPPO, run by the agent, computes the same PPO loss
    as Network A, with gradients flowing back through time inside each stored sequence.
    The loss weights used (vf_coef, ent_coef, clip_range) are stored in each run's config.json.

Uses: sb3_contrib (RecurrentPPO), torch, common.inputs, common.policies.
"""
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from common.inputs import SelectInputs
from common.policies import RecurrentSB3Policy

NAME = "networkB_lstm"
ALGO = RecurrentPPO
POLICY_NAME = "MlpLstmPolicy"
POLICY = RecurrentSB3Policy
CONFIG = {
    "frames": 1,
    "drop_sensors": [],
    "frame_encoder": 64,
    "lstm_hidden": 64,
    "lstm_layers": 1,
    "head": [64],
    "log_std_init": 0.0,
}
"""Network settings. frames: newest frames fed at each step (1-10). drop_sensors: channel indices
removed from every frame. frame_encoder: width of the layer before the LSTM (0 feeds the raw
observation straight in). lstm_hidden: LSTM state size. lstm_layers: stacked LSTM layers.
head: widths of the layers after the LSTM. log_std_init: initial action spread; the standard
deviation is exp(this)."""


class FrameEncoder(BaseFeaturesExtractor):
    """One linear layer and a tanh, applied at every step before the LSTM.

    observation_space: the (already cut) observation. hidden: output width, which is the
    size of the vector the LSTM receives.
    """

    def __init__(self, observation_space, hidden=64):
        super().__init__(observation_space, features_dim=hidden)
        self.net = nn.Sequential(nn.Linear(observation_space.shape[0], hidden), nn.Tanh())

    def forward(self, obs):
        """Return the encoded features for a batch of observations."""
        return self.net(obs)


def wrappers(cfg):
    """Environment wrappers for this network: select the frames and sensors it sees."""
    return [lambda env: SelectInputs(env, cfg["frames"], cfg["drop_sensors"])]


def policy_kwargs(cfg):
    """Keyword arguments that tell sb3-contrib how to build the recurrent actor and critic."""
    kwargs = dict(
        lstm_hidden_size=cfg["lstm_hidden"],
        n_lstm_layers=cfg["lstm_layers"],
        shared_lstm=False,
        enable_critic_lstm=True,
        net_arch=dict(pi=list(cfg["head"]), vf=list(cfg["head"])),
        log_std_init=cfg["log_std_init"],
        ortho_init=True,
    )
    if cfg["frame_encoder"] > 0:
        kwargs.update(features_extractor_class=FrameEncoder,
                      features_extractor_kwargs=dict(hidden=cfg["frame_encoder"]))
    return kwargs
