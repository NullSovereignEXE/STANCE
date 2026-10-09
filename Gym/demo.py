import argparse
import time

import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer

import stance_env  # registers StanceAnkle-v0
from stance_env.ankle_gym_env import FRAME_SKIP

# Target angle about -10 deg, max stiffness, mid-range damping (all in [-1, 1]).
FIXED_ACTION = np.array([-0.5, 1.0, 0.0], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help="use random actions")
    parser.add_argument("--seed", type=int, help="random seed for reproducibility")
    parser.add_argument("--split", choices=["train", "test"], default="train") # designed for future use, but currently only train is supported
    parser.add_argument("--slowmo", type=float, default=10.0, help="playback slowdown factor")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 2**31)

    env_id = "StanceAnkle-v0" if args.split == "train" else "StanceAnkleTest-v0"
    env = gym.make(env_id)
    sim = env.unwrapped
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    g = sim.ground
    print(f"seed {seed} | heel/toe k0 {g.k0.round(0)} N/m | mu {g.mu.round(2)} | f_yield {g.f_yield.round(0)} N")
    step_dt = sim.model.opt.timestep * FRAME_SKIP

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Side view that follows the 70 kg body mass.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.body_bid
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.0, 90.0, -8.0
        sim.ground.update_visual(sim.model)
        viewer.sync()
        time.sleep(1.0)

        total, info = 0.0, {}
        for t in range(100):
            action = env.action_space.sample() if args.random else FIXED_ACTION
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            with viewer.lock():
                sim.ground.update_visual(sim.model)  # sink and dent the ground cells
            viewer.sync()
            time.sleep(step_dt * args.slowmo)
            if terminated or truncated or not viewer.is_running():
                break

        outcome = "fell" if terminated else "completed"
        print(f"{outcome} after {t + 1} steps | return {total:.1f} | "
              f"tilt {info['leg_tilt_deg']:.1f} deg | peak GRF {info['normal_force_bw']:.2f} BW")

        # Keep the final pose on screen until the window is closed.
        while viewer.is_running():
            time.sleep(0.05)
    env.close()


if __name__ == "__main__":
    main()
