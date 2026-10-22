"""Tests for Network B: python -m pytest networkB_lstm -q"""
import numpy as np

from agent.train import build_model, load_network, make_venv, make_wrappers, merged_config
from common.env_setup import N_SENSORS, make_env


def small_model():
    net = load_network("lstm")
    cfg = merged_config(net, overrides=["n_envs=1", "lstm_hidden=16", "frame_encoder=8", "head=[8]"])
    wrappers = make_wrappers(cfg, net)
    return net, cfg, build_model(cfg, net, make_venv(cfg, wrappers, seed=0), seed=0)


def test_default_input_is_the_newest_frame():
    net = load_network("lstm")
    cfg = merged_config(net)
    obs, _ = make_env("train", make_wrappers(cfg, net)).reset(seed=0)
    assert obs.shape == (N_SENSORS,)


def test_memory_changes_the_action():
    """The same frame after a different history must give a different action."""
    net, _, model = small_model()
    policy = net.POLICY(model)
    x = np.ones(N_SENSORS, dtype=np.float32)
    policy.reset()
    first = policy.act(x)
    for _ in range(5):
        policy.act(np.random.default_rng(0).normal(size=N_SENSORS).astype(np.float32))
    later = policy.act(x)
    assert first.shape == (3,) and np.all(np.abs(first) <= 1)
    assert not np.allclose(first, later)


def test_reset_clears_memory():
    net, _, model = small_model()
    policy = net.POLICY(model)
    x = np.ones(N_SENSORS, dtype=np.float32)
    policy.reset()
    first = policy.act(x)
    policy.act(-x)
    policy.reset()
    np.testing.assert_allclose(first, policy.act(x), atol=1e-6)
