"""Network B is an LSTM that builds its own memory of the STANCE episode as it steps through it.
OWNER: Mukul Yadav.

This file defines the network only. It was trained with RecurrentPPO, and demo.py loads the
weights from trained_networks/networkB_lstm/ (model.zip and vecnormalize.pkl).

Data flow:
    Newest frame -> FrameEncoder -> LSTM -> Head layers -> Output
The critic has its own LSTM. The LSTM state carries from step to step and resets every episode.

FrameEncoder is the only layer defined here; RecurrentPPO builds the LSTM, heads and output
layers from policy_kwargs().

Output:
    1. Actor: three values in [-1, 1] for target angle, stiffness K and damping B, plus a
       learned spread per output. ResidualAction turns them into offsets from a fixed anchor
       action: anchor + 0.3 * value.
    2. Critic: one state value.

The Loss is measured with RecurrentPPO, just like for Network A (MLP) (vf_coef 0.5, ent_coef 0.0,
clip_range 0.2, max_grad_norm 0.5).
"""
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from common.inputs import SelectInputs
from common.policies import RecurrentSB3Policy

# The lines below tell demo.py what the network is
NAME = "networkB_lstm"
ALGO = RecurrentPPO
POLICY_NAME = "MlpLstmPolicy"
POLICY = RecurrentSB3Policy

# The default LSTM network architecture. policy_kwargs() and wrappers() read this further down
CONFIG = {
    "frames": 1,  # 1 means only the newest frame is fed to the network every step to help it build memory
    "drop_sensors": [],  # Describes channel indices removed from every frame. Here we are not dropping any channels.
    "frame_encoder": 64,  # Width of the linear layer before the LSTM
    "lstm_hidden": 64,
    "lstm_layers": 1,
    "head": [64],  # gives one 64-unit layer after the LSTM, for both actor and critic
    "log_std_init": 0.0,  # starting exploration noise, 0.0 is the stable baselines default (moderate noise)
}


# This class is the first layer of the network. It takes the 11 sensor readings from the current
# step, passes them through one linear layer with 64 units, and applies tanh. The 64 numbers that come out go into the LSTM
class FrameEncoder(BaseFeaturesExtractor):
    #Linear layer and tanh applied at every step before the LSTM.

    def __init__(self, observation_space, hidden=64):
        super().__init__(observation_space, features_dim=hidden)
        self.net = nn.Sequential(nn.Linear(observation_space.shape[0], hidden), nn.Tanh())

    def forward(self, obs):
        return self.net(obs)


def wrappers(cfg):
    # Returns functions, not a wrapped env, because the env is built elsewhere (once per env
    # and again for evaluation). Each one is applied to a fresh env, with the cuts set by cfg.
    return [lambda env: SelectInputs(env, cfg["frames"], cfg["drop_sensors"])]


def policy_kwargs(cfg):
    # Settings RecurrentPPO uses to build the actor and critic: LSTM size, heads, encoder.
    # kwargs: a dict of named settings, passed to RecurrentPPO as policy_kwargs
    kwargs = dict(
        lstm_hidden_size=cfg["lstm_hidden"],
        n_lstm_layers=cfg["lstm_layers"],
        net_arch=dict(pi=list(cfg["head"]), vf=list(cfg["head"])),
        log_std_init=cfg["log_std_init"],
    )
    if cfg["frame_encoder"] > 0:
        kwargs.update(features_extractor_class=FrameEncoder,
                      features_extractor_kwargs=dict(hidden=cfg["frame_encoder"]))
    return kwargs
