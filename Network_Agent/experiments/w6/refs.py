"""Fixed-action references under W_IMPACT 6 / LAMBDA_BW 1.58, per seed 1000-1199. Writes ref_<split>_<theta>_<K>_<B>.csv."""
import sys, csv, numpy as np
sys.path.insert(0, ".")
from common.env_setup import make_env, to_norm_action
from common.evaluate import run_episodes, summarise, EVAL_SEEDS
from common.policies import FixedPolicy
from agent.reward import ReweightedImpact
import stance_env, os
out = sys.argv[1]
print("stance_env:", os.path.realpath(os.path.dirname(stance_env.__file__)))
for split, act in [("train", (-20.0, 250.0, 2.0)), ("train", (-20.0, 500.0, 10.0)),
                   ("test", (-20.0, 500.0, 10.0)), ("test", (-20.0, 250.0, 2.0))]:
    env = ReweightedImpact(make_env(split), 6.0, 1.58)
    eps = run_episodes(FixedPolicy(to_norm_action(env, *act)), env, EVAL_SEEDS)
    env.close()
    s = summarise(eps)
    path = f"{out}/ref_{split}_{int(act[0])}_{int(act[1])}_{int(act[2])}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eps[0])); w.writeheader(); w.writerows(eps)
    print(split, act, "surv %.1f toe %.3f return %.2f" % (s["survival_pct"], s["toe_peak_bw"], s["mean_return"]))
