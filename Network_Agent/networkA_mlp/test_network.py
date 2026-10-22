"""Tests for Network A: python -m pytest networkA_mlp -q"""
import numpy as np
import torch

from agent.train import build_model, load_network, make_venv, make_wrappers, merged_config
from common.env_setup import N_SENSORS, make_env
from networkA_mlp.network import HistoryMLP


def test_encoder_shape_and_gradient():
    env = make_env()
    encoder = HistoryMLP(env.observation_space, hidden=(32, 16))
    x = torch.randn(4, env.observation_space.shape[0], requires_grad=True)
    y = encoder(x)
    assert y.shape == (4, 16)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_wrappers_cut_the_observation():
    net = load_network("mlp")
    cfg = merged_config(net, overrides=["frames=3", "drop_sensors=[7]"])
    env = make_env("train", make_wrappers(cfg, net))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3 * (N_SENSORS - 1),)


def test_model_outputs_valid_actions():
    net = load_network("mlp")
    cfg = merged_config(net, overrides=["n_envs=1", "batch_size=256", "hidden=[16]", "head=[8]"])
    wrappers = make_wrappers(cfg, net)
    model = build_model(cfg, net, make_venv(cfg, wrappers, seed=0), seed=0)
    obs, _ = make_env("train", wrappers).reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (3,) and np.all(np.abs(action) <= 1)
