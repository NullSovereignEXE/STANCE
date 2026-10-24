"""Tests for the agent: python -m pytest agent -q"""
import numpy as np
import pytest

from agent.actions import ResidualAction
from agent.config import AGENT_CONFIG
from agent.reward import ReweightedImpact
from agent.train import NETWORKS, REQUIRED, load_network, make_wrappers, merged_config
from common.env_setup import make_env, to_norm_action, to_physical
from common.evaluate import EVAL_SEEDS, run_episodes
from common.policies import FixedPolicy


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


def test_reweighting_with_the_original_weights_changes_nothing():
    env = ReweightedImpact(make_env(), 2.94, 1.58)
    rng = np.random.default_rng(0)
    for seed in range(5):
        env.reset(seed=seed)
        done = False
        while not done:
            _, r, term, trunc, info = env.step(rng.uniform(-1, 1, 3).astype(np.float32))
            assert abs(r - info["reward_original"]) < 1e-6
            done = term or trunc


def test_reweighted_fixed_action_matches_the_rescored_table():
    env = ReweightedImpact(make_env("train"), 6.0, 1.58)
    eps = run_episodes(FixedPolicy(to_norm_action(env, -20.0, 250.0, 2.0)), env, EVAL_SEEDS)
    assert abs(np.mean([e["ret"] for e in eps]) - 79.61) < 0.05


def test_reweighting_is_appended_last_only_when_both_keys_are_set():
    net = load_network("mlp")
    base = make_wrappers(merged_config(net), net)
    one = make_wrappers(merged_config(net, overrides=["w_impact=6.0"]), net)
    both = make_wrappers(merged_config(net, overrides=["w_impact=6.0", "lambda_bw=1.58"]), net)
    assert len(one) == len(base) and len(both) == len(base) + 1
    assert isinstance(both[-1](make_env()), ReweightedImpact)
