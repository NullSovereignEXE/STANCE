"""Fixed-action search on a split under W 6 / L 1.58, seeds 1000-1199.
usage: fixed.py OUT SPLIT [theta,K,B ...]   (no actions: the 60-action grid plus the six w6 mean actions)"""
import sys, csv, itertools
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
from common.env_setup import make_env, to_norm_action
from common.evaluate import run_episodes, summarise, EVAL_SEEDS
from common.policies import FixedPolicy
from agent.reward import ReweightedImpact

OUT, split = Path(sys.argv[1]), sys.argv[2]
if len(sys.argv) > 3:
    acts = [("given", tuple(map(float, a.split(",")))) for a in sys.argv[3:]]
else:
    acts = [("grid", a) for a in itertools.product((-20.0, -18.0, -16.0), (350.0, 400.0, 450.0, 500.0), (0.5, 1.0, 1.5, 2.0, 3.0))]
    with open(OUT / "w6_mean_actions.csv") as f:
        acts += [(r["source"], (float(r["theta"]), float(r["K"]), float(r["B"]))) for r in csv.DictReader(f)]
env = ReweightedImpact(make_env(split), 6.0, 1.58)
rows = []
for src, a in acts:
    eps = run_episodes(FixedPolicy(to_norm_action(env, *a)), env, EVAL_SEEDS)
    m = summarise(eps)
    tag = "%g_%g_%g" % a
    with open(OUT / f"fixed_{split}_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eps[0])); w.writeheader(); w.writerows(eps)
    rows.append({"source": src, "theta": a[0], "K": a[1], "B": a[2], "survival_pct": m["survival_pct"],
                 "toe_peak_bw": m["toe_peak_bw"], "mean_return": m["mean_return"]})
    print(src, a, "surv %.1f toe %.3f ret %.2f" % (m["survival_pct"], m["toe_peak_bw"], m["mean_return"]), flush=True)
with open(OUT / f"fixed_{split}_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
