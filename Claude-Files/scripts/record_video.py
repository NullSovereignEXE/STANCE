"""
Record a rollout as an mp4.

Use this in WEEK 2 with random actions, long before any policy exists. You are
checking the PHYSICS, not the behaviour: does the foot pass through the floor,
does the leg jitter, does the dent appear under the right contact point, does
the pylon spring explode? Every one of those is invisible in a reward curve and
obvious in two seconds of video.

    python scripts/record_video.py                       # random actions
    python scripts/record_video.py --policy runs/smoke_ppo
    python scripts/record_video.py --split test --track  # held-out ground, camera follows
"""
import argparse

from stance_env.ankle_env import AnkleEnv
from stance_env.viz import Recorder


def record(policy=None, seed=0, split="train", track=False,
           out="videos/rollout.mp4"):
    env = AnkleEnv(ground_split=split)
    obs, _ = env.reset(seed=seed)

    rec = Recorder(env.model, track="body_mass" if track else None)

    done = False
    while not done:
        if policy is None:
            action = env.action_space.sample()
        else:
            # deterministic=True: no exploration noise, the policy's actual
            # best guess rather than a sample around it
            action, _ = policy.predict(obs, deterministic=True)

        obs, r, term, trunc, info = env.step(action)
        env.ground.update_visual(env.model)      # drop the floor to show the dent
        rec.capture(env.data)
        done = term or trunc

    n = rec.save(out)
    rec.close()

    print(f"{n} frames -> {out}")
    print(f"  split       : {split}")
    print(f"  ground k0   : {info['k0']} N/m")
    print(f"  friction mu : {info['mu']}")
    print(f"  final dent  : {info['dent_mm']} mm")
    print(f"  leg tilt    : {info['leg_tilt_deg']:.1f} deg")
    print(f"  outcome     : {'FELL' if term else 'survived'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None, help="path to a saved SB3 model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--track", action="store_true",
                    help="camera follows the body instead of a fixed side view")
    ap.add_argument("--out", default="videos/rollout.mp4")
    a = ap.parse_args()

    pol = None
    if a.policy:
        from stable_baselines3 import PPO
        pol = PPO.load(a.policy)

    record(pol, seed=a.seed, split=a.split, track=a.track, out=a.out)
