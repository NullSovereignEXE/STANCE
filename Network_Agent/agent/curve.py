"""The learning-curve logger. OWNER: P3.

`CurveCallback` is handed to the training loop. Every `curve_every` steps it pauses
training, plays a few deterministic episodes on the curve seeds (2000 and up, never the
table seeds) and appends one row to runs/<network>/<run>/curve.csv:

    timesteps, minutes, survival_pct, mean_return, toe_peak_bw, mean_K, mean_theta_deg

Watching mean_theta_deg as well as survival matters: the known failure is the angle
drifting positive while survival stays low.

Uses: stable_baselines3's BaseCallback, evaluate.run_episodes and evaluate.summarise,
and the same environment wrappers as training.
"""
import csv
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from common.env_setup import make_env
from common.evaluate import CURVE_SEEDS, run_episodes, summarise


class CurveCallback(BaseCallback):
    """Evaluate on the curve seeds every `cfg["curve_every"]` steps and log one CSV row.

    cfg: the merged config. wrappers: the same wrapper list used for training.
    policy_cls: the network's evaluation adapter (policies.py). path: the curve.csv file.
    """

    def __init__(self, cfg, wrappers, policy_cls, path):
        super().__init__()
        self.cfg, self.policy_cls, self.path = cfg, policy_cls, Path(path)
        self.env = make_env("train", wrappers)
        self.next = cfg["curve_every"]
        self.t0 = time.time()

    def _on_step(self):
        """Called by SB3 after every environment step; logs a row when a point is due."""
        if self.num_timesteps < self.next:
            return True
        self.next += self.cfg["curve_every"]
        policy = self.policy_cls(self.model, self.model.get_vec_normalize_env())
        s = summarise(run_episodes(policy, self.env, CURVE_SEEDS[: self.cfg["curve_episodes"]]))
        row = {"timesteps": self.num_timesteps,
               "minutes": round((time.time() - self.t0) / 60, 2),
               "survival_pct": round(s["survival_pct"], 1),
               "mean_return": round(s["mean_return"], 2),
               "toe_peak_bw": round(s["toe_peak_bw"], 3),
               "mean_K": round(s["mean_K"], 1),
               "mean_theta_deg": round(s["mean_theta_deg"], 2)}
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print("  curve:", row)
        return True
