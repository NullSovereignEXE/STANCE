"""
PPO Smoke Test - Network A (MLP)

Purpose:
    1. Create the real MuJoCo environment.
    2. Build our Actor-Critic MLP using Stable-Baselines3.
    3. Train PPO for 1024 timesteps.
    4. Verify Actor and Critic weights are updated.

This is only an integration test, not full training.
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from network import CONFIG, policy_kwargs, wrappers


# ==================================================
# STEP 1: Locate our MuJoCo environment
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

GYM_PATH = ROOT / "Gym"

sys.path.insert(0, str(GYM_PATH))

# Import registers StanceAnkle-v0
import stance_env


# ==================================================
# STEP 2: Create our Gym environment
# ==================================================

def create_env():

    env = gym.make("StanceAnkle-v0")

    # Apply our observation selection wrapper
    for wrap in wrappers(CONFIG):
        env = wrap(env)

    return env


# ==================================================
# STEP 3: Create vectorized environment
# ==================================================

venv = DummyVecEnv([create_env])

# Normalize sensor observations and rewards
venv = VecNormalize(
    venv,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    gamma=0.99
)


print("\n===== ENVIRONMENT =====")
print("Observation shape:", venv.observation_space.shape)
print("Action shape:", venv.action_space.shape)


# ==================================================
# STEP 4: Create PPO Actor-Critic model
# ==================================================

model = PPO(
    policy="MlpPolicy",
    env=venv,

    policy_kwargs=policy_kwargs(CONFIG),

    learning_rate=3e-4,

    n_steps=128,
    batch_size=64,
    n_epochs=4,

    gamma=0.99,
    gae_lambda=0.95,

    clip_range=0.2,

    seed=0,
    device="cpu",
    verbose=1
)


# ==================================================
# STEP 5: Record weights BEFORE training
# ==================================================

actor_before = (
    model.policy.action_net.weight
    .detach()
    .clone()
)

critic_before = (
    model.policy.value_net.weight
    .detach()
    .clone()
)


# ==================================================
# STEP 6: Start PPO training
# ==================================================

print("\n===== STARTING PPO SMOKE TEST =====")

model.learn(
    total_timesteps=1024
)


# ==================================================
# STEP 7: Record weights AFTER training
# ==================================================

actor_after = (
    model.policy.action_net.weight
    .detach()
    .clone()
)

critic_after = (
    model.policy.value_net.weight
    .detach()
    .clone()
)


# ==================================================
# STEP 8: Calculate weight changes
# ==================================================

actor_change = torch.norm(
    actor_after - actor_before
).item()

critic_change = torch.norm(
    critic_after - critic_before
).item()


print("\n===== SMOKE TEST RESULTS =====")

print("Actor weight change:", actor_change)
print("Critic weight change:", critic_change)


# ==================================================
# STEP 9: Verify successful training
# ==================================================

assert np.isfinite(actor_change), "Invalid Actor weight change"
assert np.isfinite(critic_change), "Invalid Critic weight change"

assert actor_change > 0, "Actor weights did not update!"
assert critic_change > 0, "Critic weights did not update!"

print("\nPPO SMOKE TEST PASSED!")


# ==================================================
# STEP 10: Close environment
# ==================================================

venv.close()