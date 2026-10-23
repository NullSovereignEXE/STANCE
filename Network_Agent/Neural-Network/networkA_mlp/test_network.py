import torch
import numpy as np
import gymnasium as gym
import pytest

from network import SelectObservations

from network import HistoryMLP

#gradient test
def test_encoder_gradients():

    # 1. Define the observation space
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(110,),
        dtype=np.float32
    )

    # 2. Create our MLP
    encoder = HistoryMLP(
        observation_space,
        hidden=(256, 256),
        activation="tanh"
    )

    # 3. Generate four dummy observations
    observations = torch.randn(
        4, 110,
        requires_grad=True
    )

    # 4. Forward propagation
    output = encoder(observations)

    print("\nOutput shape:", output.shape)

    assert output.shape == (4, 256)

    # 5. Create a simple test loss
    loss = output.sum()

    # 6. Calculate gradients
    loss.backward()

    # 7. Check input gradients
    assert observations.grad is not None
    assert torch.isfinite(observations.grad).all()

    # 8. Check network parameter gradients
    for name, parameter in encoder.named_parameters():

        assert parameter.grad is not None, (
            f"No gradient for {name}"
        )

        assert torch.isfinite(parameter.grad).all(), (
            f"Invalid gradient for {name}"
        )

        print(
            f"{name:30s} | "
            f"Gradient norm: {parameter.grad.norm().item():.6f}"
        )

    print("\nGradient verification passed!")


#Input Selection test
# --------------------------------------------------
# TEST 1: Verify input dimensions
# --------------------------------------------------

@pytest.mark.parametrize(
    "frames,drop_sensors,expected_size",
    [
        (10, [], 110),
        (3, [], 33),
        (1, [], 11),
        (10, [8, 9, 10], 80),
        (10, [7], 100),
        (10, [2], 100),
        (10, [3], 100),
        (3, [2, 3], 27),
    ]
)
def test_input_dimensions(frames, drop_sensors, expected_size):

    # Create a dummy environment with 110 observations
    class DummyEnv(gym.Env):

        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(110,),
                dtype=np.float32
            )

            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32
            )

    # Apply our observation selector
    env = SelectObservations(
        DummyEnv(),
        frames=frames,
        drop_sensors=drop_sensors
    )

    # Check output dimensions
    assert env.observation_space.shape == (expected_size,)

    print(
        f"Frames={frames}, "
        f"Drop={drop_sensors}, "
        f"Expected={expected_size} -> PASSED"
    )

    env.close()


# --------------------------------------------------
# TEST 2: Verify correct sensor values are selected
# --------------------------------------------------

def test_correct_sensor_selection():

    class DummyEnv(gym.Env):

        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(110,),
                dtype=np.float32
            )

            self.action_space = gym.spaces.Box(
                -1.0,
                1.0,
                shape=(3,),
                dtype=np.float32
            )

    env = SelectObservations(
        DummyEnv(),
        frames=3,
        drop_sensors=[7]
    )

    # Create 110 known values
    observation = np.arange(110, dtype=np.float32)

    # Apply selection
    selected = env.observation(observation)

    # Expected: newest 3 frames, excluding sensor 7
    history = observation.reshape(10, 11)

    expected = np.delete(history[-3:], 7, axis=1).flatten()

    assert selected.shape == (30,)

    np.testing.assert_array_equal(selected, expected)

    print("\nCorrect sensor selection verified!")

    env.close()


# --------------------------------------------------
# TEST 3: Verify invalid configurations are rejected
# --------------------------------------------------

@pytest.mark.parametrize(
    "frames,drop_sensors",
    [
        (0, []),
        (11, []),
        (10, [11]),
        (10, [-1]),
        (10, list(range(11))),
    ]
)
def test_invalid_input_configurations(frames, drop_sensors):

    class DummyEnv(gym.Env):

        def __init__(self):
            self.observation_space = gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=(110,),
                dtype=np.float32
            )

            self.action_space = gym.spaces.Box(
                -1.0,
                1.0,
                shape=(3,),
                dtype=np.float32
            )

    with pytest.raises(ValueError):
        SelectObservations(
            DummyEnv(),
            frames=frames,
            drop_sensors=drop_sensors
        )

def test_real_environment_input_selection():

    import sys
    from pathlib import Path

    import gymnasium as gym
    import numpy as np

    from network import SelectObservations

    # Locate our actual Gym environment
    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT / "Gym"))

    import stance_env

    # STEP 1: Create the original MuJoCo environment
    raw_env = gym.make("StanceAnkle-v0")

    # STEP 2: Create our input selector
    wrapped_env = SelectObservations(
        raw_env,
        frames=3,
        drop_sensors=[7]
    )

    try:
        # STEP 3: Reset the environment
        obs, info = wrapped_env.reset(seed=42)

        # STEP 4: Get the original 110 observations
        raw_obs = raw_env.unwrapped._get_obs()

        print("\n===== RAW MUJOCO OBSERVATIONS =====")
        print("Raw observation shape:", raw_obs.shape)

        # STEP 5: Check the selected observations
        print("\n===== SELECTED OBSERVATIONS =====")
        print("Selected observation shape:", obs.shape)

        # STEP 6: Independently calculate expected values
        history = raw_obs.reshape(10, 11)

        expected = np.delete(
            history[-3:],
            7,
            axis=1
        ).flatten()

        # STEP 7: Verify dimensions and values
        assert raw_obs.shape == (110,)
        assert obs.shape == (30,)

        np.testing.assert_array_equal(obs, expected)

        assert np.isfinite(obs).all()

        print("\n===== LATEST SELECTED SENSOR VALUES =====")

        sensor_names = [
            "ankle_angle",
            "ankle_rate",
            "ankle_torque",
            "vertical_grf",
            "pylon_compression",
            "shank_tilt",
            "shank_tilt_rate",
            "vertical_accel",
            "prev_theta_d",
            "prev_K",
            "prev_B",
        ]

        latest = obs.reshape(3, 10)[-1]

        selected_names = [
            name for i, name in enumerate(sensor_names)
            if i != 7
        ]

        for name, value in zip(selected_names, latest):
            print(f"{name:25s}: {value:.5f}")

        print("\nReal MuJoCo input selection passed!")

    finally:
        wrapped_env.close()