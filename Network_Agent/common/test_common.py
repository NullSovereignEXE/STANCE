"""Tests for the shared code: python -m pytest common -q"""
import gymnasium as gym
import numpy as np
import pytest

from common.env_setup import N_SENSORS, SENSOR_NAMES, make_env, to_norm_action, to_physical
from common.evaluate import run_episodes
from common.inputs import SelectInputs
from common.policies import FixedPolicy


def test_sensor_names_match_sensor_count():
    assert len(SENSOR_NAMES) == N_SENSORS


def test_select_inputs_keeps_newest_frame():
    full, _ = make_env().reset(seed=1)
    obs, _ = SelectInputs(make_env(), frames=1).reset(seed=1)
    assert obs.shape == (N_SENSORS,)
    np.testing.assert_allclose(obs, full[-N_SENSORS:])


def test_select_inputs_drops_a_column_in_every_frame():
    env = make_env()
    full, _ = env.reset(seed=2)
    wrapped = SelectInputs(make_env(), frames=3, drop=[7])
    obs, _ = wrapped.reset(seed=2)
    assert obs.shape == (3 * (N_SENSORS - 1),)
    expected = full.reshape(-1, N_SENSORS)[-3:]
    expected = np.delete(expected, 7, axis=1).reshape(-1)
    np.testing.assert_allclose(obs, expected)
    assert wrapped.observation_space.shape == obs.shape


def test_select_inputs_defaults_keep_everything():
    full, _ = make_env().reset(seed=3)
    obs, _ = SelectInputs(make_env()).reset(seed=3)
    np.testing.assert_allclose(obs, full)


@pytest.mark.parametrize("kwargs", [dict(frames=0), dict(frames=99),
                                    dict(drop=[11]), dict(drop=list(range(N_SENSORS)))])
def test_select_inputs_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        SelectInputs(make_env(), **kwargs)


def test_action_conversion_round_trip():
    env = make_env()
    for phys in [(-20, 500, 10), (0, 50, 3), (10, 5, 0.5)]:
        th, K, B = to_physical(env, to_norm_action(env, *phys))
        np.testing.assert_allclose([th, K, B], phys, rtol=1e-4, atol=0.01)


def test_evaluator_finds_toe_force_and_failure_reason():
    """Keys are found by name: the toe key on any step, the reason key only on the last step."""
    class AddKeys(gym.Wrapper):
        def step(self, a):
            obs, r, term, trunc, info = super().step(a)
            info["toe_force_bw"] = 1.5
            if term:
                info["termination_reason"] = "forward_tilt"
            return obs, r, term, trunc, info

    env = make_env("train", [AddKeys])
    ep = run_episodes(FixedPolicy(np.zeros(3, dtype=np.float32)), env, [1000, 1001, 1002])
    assert all(e["toe_peak"] == 1.5 for e in ep)
    assert all(e["reason"] == "forward_tilt" for e in ep if e["terminated"])
