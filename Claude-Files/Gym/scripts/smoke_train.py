"""
Week 3 smoke test. NOT a real training run.

You are not looking for good behaviour here. You are looking for the reward
curve to trend UPWARD rather than sideways. If it is flat after 50k steps,
something is wrong in the environment, not the agent -- and you want to know
that now rather than in week 6.

    python scripts/smoke_train.py
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym
import stance_env  # noqa: F401


def main(total_steps=50_000, n_envs=8, seed=0):
    # Vectorised envs: 8 copies stepping in parallel. PPO collects from all of
    # them, which is why a cheap simulation matters more than sample efficiency.
    vec = make_vec_env("StanceAnkle-v0", n_envs=n_envs, seed=seed)

    model = PPO("MlpPolicy", vec, verbose=1, seed=seed,
                n_steps=256, batch_size=512, learning_rate=3e-4)
    model.learn(total_timesteps=total_steps)

    # Quick evaluation on BOTH distributions. The gap between them is the
    # headline result of the whole project.
    for env_id in ("StanceAnkle-v0", "StanceAnkleTest-v0"):
        env = Monitor(gym.make(env_id))
        rewards, falls, peak_forces = [], 0, []
        for ep in range(30):
            obs, _ = env.reset(seed=1000 + ep)
            total, done = 0.0, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                total += r
                peak_forces.append(info["normal_force_bw"])
                done = term or trunc
            falls += int(term)
            rewards.append(total)
        print(f"\n{env_id}")
        print(f"  mean return : {np.mean(rewards):6.1f} +/- {np.std(rewards):.1f}")
        print(f"  fall rate   : {falls / 30:6.1%}")
        print(f"  peak GRF    : {np.max(peak_forces):6.2f} body weights")

    model.save("runs/smoke_ppo")


if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    main()
