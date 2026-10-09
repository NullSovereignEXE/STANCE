import os as _os
import sys as _sys

if _sys.platform.startswith("linux") and not _os.environ.get("DISPLAY"):
    _os.environ.setdefault("MUJOCO_GL", "egl")

from gymnasium.envs.registration import register

from stance_env.ankle_gym_env import AnkleEnv
from stance_env.ground import normal_force, friction_force, GroundModel


# Training distribution
'''
This is the training environment for the stance ankle environment. 
It is used to train the agent on a variety of ground conditions. 
The agent should be trained on this environment, and evaluated on the test environment.
'''

register(
    id="StanceAnkle-v0", # This is the training environment for the stance ankle environment. It is used to train the agent on a variety of ground conditions. The agent should be trained on this environment, and evaluated on the test environment.
    entry_point="stance_env.ankle_env:AnkleEnv",
    max_episode_steps=100,  # If the leg doesn't fall in 100 steps, the episode is considered TRUNCATED and ends.
    kwargs={"ground_split": "train"},
)


# Testing distribution
'''
This is the test environment for the stance ankle environment. 
It is used to evaluate the performance of the agent on unseen ground conditions. 
The agent should not be trained on this environment, only evaluated.
'''

register(
    id="StanceAnkleTest-v0", 
    entry_point="stance_env.ankle_env:AnkleEnv",
    max_episode_steps=100,
    kwargs={"ground_split": "test"},
)

__all__ = ["AnkleEnv", "GroundModel", "normal_force", "friction_force"]
