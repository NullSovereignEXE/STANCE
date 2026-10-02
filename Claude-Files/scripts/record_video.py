"""
Record a rollout as an mp4.

Use this in WEEK 2 with random actions, long before any policy exists. You are
checking the physics, not the behaviour: does the foot pass through the floor,
does the leg jitter, does the dent appear under the right contact point? Every
one of those bugs is invisible in a reward curve and obvious in two seconds of
video.

    python scripts/record_video.py                    # random actions
    python scripts/record_video.py runs/smoke_ppo     # a trained policy
"""
import sys
import os

# Headless machines (lab servers, WSL, CI) have no X display, and MuJoCo's
# default GLFW backend will fail with "no OpenGL platform library". EGL works
# without a display. Must be set BEFORE mujoco is imported. On a laptop with a
# desktop session this is harmless.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import imageio.v2 as imageio
import gymnasium as gym

import stance_env  # noqa: F401
from stance_env.ankle_env import AnkleEnv


def record(policy=None, seed=0, env_id="StanceAnkle-v0", out="videos/rollout.mp4"):
    env = AnkleEnv(
        ground_split="test" if "Test" in env_id else "train",
        render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=seed)

    frames, done = [], False
    while not done:
        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=True)   # no exploration noise
        obs, r, term, trunc, info = env.step(action)
        frames.append(env.render())
        done = term or trunc

    os.makedirs(os.path.dirname(out), exist_ok=True)
    # 100 frames at 20 fps = 5 s of slow motion. Real time would be 0.5 s,
    # far too fast to see an impact.
    imageio.mimsave(out, frames, fps=20)
    print(f"{len(frames)} frames -> {out}")
    print(f"  ground k0   : {env.k0.round(0)} N/m")
    print(f"  final dent  : {info['dent_mm'].round(1)} mm")
    print(f"  leg tilt    : {info['leg_tilt_deg']:.1f} deg")
    print(f"  outcome     : {'FELL' if term else 'survived'}")
    env.close()


if __name__ == "__main__":
    pol = None
    if len(sys.argv) > 1:
        from stable_baselines3 import PPO
        pol = PPO.load(sys.argv[1])
    record(pol)
