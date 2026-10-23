"""Tests for the agent: python -m pytest agent -q"""
import numpy as np
import pytest

from agent.actions import ResidualAction
from agent.config import AGENT_CONFIG
from agent.train import NETWORKS, REQUIRED, load_network, make_wrappers, merged_config
from common.env_setup import make_env, to_physical


@pytest.mark.parametrize("key", sorted(NETWORKS))
def test_network_follows_the_contract(key):
    net = load_network(key)
    assert all(hasattr(net, name) for name in REQUIRED)
    assert not set(AGENT_CONFIG) & set(net.CONFIG)


def test_overrides_are_applied_and_checked():
    net = load_network("mlp")
    cfg = merged_config(net, smoke=True, overrides=["gamma=0.9", "hidden=[8,8]", "residual=True"])
    assert cfg["gamma"] == 0.9 and cfg["hidden"] == [8, 8] and cfg["residual"] is True
    assert cfg["total_steps"] == 4096
    with pytest.raises(SystemExit):
        merged_config(net, overrides=["not_a_key=1"])
    with pytest.raises(SystemExit):
        merged_config(net, overrides=["gamma"])


def test_zero_offset_is_the_anchor():
    env = ResidualAction(make_env())
    env.reset(seed=0)
    theta, K, B = to_physical(env, env.action(np.zeros(3)))
    assert abs(theta + 20) < 0.01 and abs(K - 500) < 0.01 and abs(B - 10) < 0.001


def test_offsets_are_scaled_and_clipped():
    env = ResidualAction(make_env(), anchor=(0.0, 50.0, 3.0), scale=(0.5, 0.5, 0.5))
    base = env.action(np.zeros(3))
    np.testing.assert_allclose(env.action(np.ones(3)) - base, 0.5 * np.ones(3), atol=1e-6)
    assert np.all(np.abs(env.action(np.full(3, 10.0))) <= 1)


def test_residual_is_applied_only_when_asked():
    net = load_network("mlp")
    plain = make_wrappers(merged_config(net, overrides=["residual=False"]), net)
    residual = make_wrappers(merged_config(net, overrides=["residual=True"]), net)
    assert len(residual) == len(plain) + 1
