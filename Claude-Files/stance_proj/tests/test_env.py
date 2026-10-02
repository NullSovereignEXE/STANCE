"""
Gymnasium API conformance. If these fail, Stable-Baselines3 will behave
strangely in ways that look like learning problems but are not.

    pytest tests/test_env.py -v
"""
import numpy as np
import gymnasium as gym
import pytest

import stance_env  # noqa: F401  -- the import registers the environments


def test_registered():
    """Both splits must be reachable by id."""
    env = gym.make("StanceAnkle-v0")
    assert env.unwrapped.ground_split == "train"
    env = gym.make("StanceAnkleTest-v0")
    assert env.unwrapped.ground_split == "test"


def test_spaces_match_observations():
    env = gym.make("StanceAnkle-v0")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), "obs outside declared space"
    assert obs.dtype == np.float32


def test_seeding_is_reproducible():
    """Same seed -> identical episode. Without this, the three-seed
    comparison in the final report means nothing."""
    def rollout(seed):
        env = gym.make("StanceAnkle-v0")
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        out = []
        for _ in range(20):
            o, r, te, tr, _ = env.step(env.action_space.sample())
            out.append((o.copy(), r))
            if te or tr:
                break
        return out

    a, b = rollout(7), rollout(7)
    assert len(a) == len(b)
    for (oa, ra), (ob, rb) in zip(a, b):
        assert np.allclose(oa, ob) and ra == rb


def test_different_seeds_give_different_ground():
    """Randomisation must actually randomise."""
    e1 = gym.make("StanceAnkle-v0"); e1.reset(seed=1)
    e2 = gym.make("StanceAnkle-v0"); e2.reset(seed=2)
    assert not np.allclose(e1.unwrapped.k0, e2.unwrapped.k0)


def test_train_and_test_ranges_are_disjoint():
    """The anti-memorisation claim, checked rather than asserted in prose."""
    train_k, test_k = [], []
    for s in range(60):
        e = gym.make("StanceAnkle-v0"); e.reset(seed=s)
        train_k.extend(e.unwrapped.k0)
        t = gym.make("StanceAnkleTest-v0"); t.reset(seed=s)
        test_k.extend(t.unwrapped.k0)
    train_k, test_k = np.array(train_k), np.array(test_k)
    # every test sample sits outside the training band
    assert np.all((test_k < 5e3 * 1.001) | (test_k > 1e5 * 0.999))
    assert np.all((train_k >= 5e3 * 0.999) & (train_k <= 1e5 * 1.001))


def test_terminated_and_truncated_are_distinct():
    """Falling is `terminated`. Running out of time is `truncated`.
    Never both for the same reason."""
    env = gym.make("StanceAnkle-v0")
    env.reset(seed=3)
    for _ in range(100):
        _, _, term, trunc, _ = env.step(np.zeros(3, np.float32))
        if term or trunc:
            break
    assert not (term and trunc)


def test_sb3_env_checker():
    """The official conformance check."""
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    from stance_env.ankle_env import AnkleEnv
    check_env(AnkleEnv(), warn=True, skip_render_check=True)
