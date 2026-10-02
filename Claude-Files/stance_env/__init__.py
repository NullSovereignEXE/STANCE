"""
STANCE environment package.

Importing this module REGISTERS the environment with Gymnasium, which is what
lets you write:

    import gymnasium as gym
    import stance_env                      # noqa: F401  (the import does the work)
    env = gym.make("StanceAnkle-v0")

rather than importing the class by hand. Registration is what makes it a
first-class Gym environment: Stable-Baselines3, the env checker, and the
vectorised wrappers all look environments up by their string id.
"""
from gymnasium.envs.registration import register

from stance_env.ankle_env import AnkleEnv          # noqa: F401
from stance_env.ground import normal_force, friction_force   # noqa: F401

# ---------------------------------------------------------------------------
# Training distribution: ground the agent learns on.
# ---------------------------------------------------------------------------
register(
    id="StanceAnkle-v0",
    entry_point="stance_env.ankle_env:AnkleEnv",
    max_episode_steps=100,            # Gym enforces this -> sets `truncated`
    kwargs={"ground_split": "train"},
)

# ---------------------------------------------------------------------------
# Held-out distribution: NEVER train on this. Evaluation only.
# Separate id so you cannot mix them up by accident in a script.
# ---------------------------------------------------------------------------
register(
    id="StanceAnkleTest-v0",
    entry_point="stance_env.ankle_env:AnkleEnv",
    max_episode_steps=100,
    kwargs={"ground_split": "test"},
)

__all__ = ["AnkleEnv", "normal_force", "friction_force"]
