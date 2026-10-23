import torch

from stable_baselines3 import PPO

from common.env_setup import make_env
from networkA_mlp.network import CONFIG, policy_kwargs, wrappers


def test_sb3_network_integration():

    # Create Gym environment (same one as the agent uses),
    # with our input selection applied
    env = make_env("train", wrappers(CONFIG))


    print("\n===== ENVIRONMENT =====")
    print("Observation shape:", env.observation_space.shape)
    print("Action shape:", env.action_space.shape)


    # Build PPO model using our MLP
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs(CONFIG),
        n_steps=256,
        batch_size=64,
        device="cpu",
        verbose=0,
    )


    print("\n===== NETWORK ARCHITECTURE =====")
    print(model.policy)


    # Count parameters
    total_params = sum(
        p.numel()
        for p in model.policy.parameters()
        if p.requires_grad
    )

    print("\nTrainable parameters:", total_params)


    # Test with real observations
    obs, info = env.reset(seed=42)

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    print("\n===== ACTOR OUTPUT =====")
    print("Action:", action)

    assert action.shape == (3,)

    # Check Critic output
    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=model.device
    ).unsqueeze(0)

    with torch.no_grad():
        value = model.policy.predict_values(obs_tensor)

    print("\n===== CRITIC OUTPUT =====")
    print("State value:", value.item())

    env.close()

    print("\nSB3 network integration successful!")
