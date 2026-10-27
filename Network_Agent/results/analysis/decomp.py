"""Per-episode reward-term decomposition. Read-only: writes only to the scratchpad.

usage (cwd = STANCE/Network_Agent): python decomp.py OUT.csv POLICY [POLICY ...]
POLICY is a run dir (runs/networkB_lstm/final_s0) or fixed:theta,K,B
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agent.train import load_network, make_wrappers
from common.env_setup import make_env, to_norm_action
from common.evaluate import EVAL_SEEDS
from common.policies import FixedPolicy
import stance_env.ankle_gym_env as E


def load(spec):
    if spec.startswith("fixed:"):
        th, K, B = map(float, spec[6:].split(","))
        env = make_env("train")
        return FixedPolicy(to_norm_action(env, th, K, B)), env
    run = Path(spec)
    saved = json.loads((run / "config.json").read_text())
    cfg = saved["config"]
    net = load_network(saved["network"])
    wr = make_wrappers(cfg, net)
    model = net.ALGO.load(run / "model.zip", device="cpu")
    vn = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: make_env("train", wr)]))
    vn.training = False
    return net.POLICY(model, vn), make_env("train", wr)


def episode(policy, env, seed):
    obs, _ = env.reset(seed=seed)
    policy.reset()
    u = env.unwrapped
    t = dict(ret=0.0, survival=0.0, impact=0.0, smooth=0.0, fall=0.0, steps=0, toe_peak=0.0)
    maxerr, done = 0.0, False
    while not done:
        prev = np.array(u.prev_action, dtype=np.float64)
        obs, r, term, trunc, info = env.step(policy.act(obs))
        surv = 1.0
        imp = -E.W_IMPACT * max(0.0, u.peak_toe / u.body_weight - E.LAMBDA_BW) ** 2
        sm = 0.0 if u.step_count == 1 else -E.W_SMOOTH * float(np.sum((u.prev_action - prev) ** 2))
        fall = -E.FALL_PENALTY if term else 0.0
        maxerr = max(maxerr, abs(surv + imp + sm + fall - r))
        t["ret"] += r; t["survival"] += surv; t["impact"] += imp; t["smooth"] += sm; t["fall"] += fall
        t["steps"] += 1
        t["toe_peak"] = max(t["toe_peak"], info["toe_force_bw"])
        done = term or trunc
    t["survived"] = int(not term)
    t["sum_err"] = abs(t["survival"] + t["impact"] + t["smooth"] + t["fall"] - t["ret"])
    t["step_err"] = maxerr
    return t


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    cols = ["policy", "seed", "ret", "survival", "impact", "smooth", "fall",
            "steps", "survived", "toe_peak", "sum_err", "step_err"]
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for spec in specs:
            policy, env = load(spec)
            for s in EVAL_SEEDS:
                t = episode(policy, env, s)
                f.write(",".join(str(v) for v in [spec, s] + [t[c] for c in cols[2:]]) + "\n")
            print("done", spec, flush=True)


if __name__ == "__main__":
    main()
