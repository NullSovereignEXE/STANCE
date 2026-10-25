"""Tables for results/w6_findings.md, from the per-seed episode files of the other w6 scripts.

usage (from Network_Agent): python experiments/w6/findings.py REFS_DIR BLIND_DIR
    REFS_DIR   output of refs.py   (ref_<split>_<theta>_<K>_<B>.csv)
    BLIND_DIR  output of blind.py and fixed.py (eps_*_normal/blind.csv, fixed_<split>_summary.csv)
All returns are under W_IMPACT 6 / LAMBDA_BW 1.58; seeds 1000-1199.
"""
import sys, csv
sys.path.insert(0, ".")
import numpy as np
from common.evaluate import _mean_ci

REFS, BLIND = sys.argv[1], sys.argv[2]
RUNS = [(net, label, s) for net, label in (("networkA_mlp", "MLP"), ("networkB_lstm", "LSTM")) for s in range(3)]


def load(p):
    with open(p) as f:
        return [{k: float(v) if k in ("seed", "ret", "terminated", "toe_peak", "theta", "K", "B") else v
                 for k, v in r.items()} for r in csv.DictReader(f)]


def stats(e):
    return (100 * np.mean([1 - x["terminated"] for x in e]), np.nanmean([x["toe_peak"] for x in e]),
            np.mean([x["ret"] for x in e]))


def paired(eps, ref):
    r = {x["seed"]: x["ret"] for x in ref}
    m, lo, hi = _mean_ci([x["ret"] - r[x["seed"]] for x in eps])
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def toe(v, split):
    return f"{v:.3f}" if split == "train" else "capped"


print("### Normal vs blind\n")
print("| run | split | survival % normal / blind | toe peak normal / blind | return normal / blind | "
      "blind − normal [95%] | mean action normal (θ, K, B) | mean action blind |")
print("|---|---|---|---|---|---|---|---|")
for net, label, s in RUNS:
    for sp in ("train", "test"):
        n = load(f"{BLIND}/eps_{net}_w6_s{s}_{sp}_normal.csv")
        b = load(f"{BLIND}/eps_{net}_w6_s{s}_{sp}_blind.csv")
        (sn, tn, rn), (sb, tb, rb) = stats(n), stats(b)
        act = lambda e: "%.1f, %.0f, %.2f" % tuple(np.mean([x[k] for x in e]) for k in ("theta", "K", "B"))
        print(f"| {label} s{s} | {sp} | {sn:.1f} / {sb:.1f} | {toe(tn, sp)} / {toe(tb, sp)} | {rn:.2f} / {rb:.2f} | "
              f"{paired(b, n)} | {act(n)} | {act(b)} |")

print("\n### Top 5 fixed actions on train (60-action grid plus the six w6 mean actions)\n")
print("| rank | action (θ, K, B) | source | survival % | toe peak (BW) | mean return |")
print("|---|---|---|---|---|---|")
with open(f"{BLIND}/fixed_train_summary.csv") as f:
    rows = sorted(csv.DictReader(f), key=lambda r: -float(r["mean_return"]))
for i, r in enumerate(rows[:5], 1):
    print(f"| {i} | ({float(r['theta']):g}, {float(r['K']):g}, {float(r['B']):g}) | {r['source']} | "
          f"{float(r['survival_pct']):.1f} | {float(r['toe_peak_bw']):.3f} | {float(r['mean_return']):.2f} |")

refs = {"train": [("(-20, 250, 2)", load(f"{REFS}/ref_train_-20_250_2.csv")),
                  ("(-20, 500, 10)", load(f"{REFS}/ref_train_-20_500_10.csv"))],
        "test": [("(-20, 250, 2)", load(f"{REFS}/ref_test_-20_250_2.csv")),
                 ("(-20, 500, 10)", load(f"{REFS}/ref_test_-20_500_10.csv"))]}
print("\n### Fixed-action references\n")
print("| action | split | survival % | toe peak (BW) | mean return |")
print("|---|---|---|---|---|")
for sp, l in refs.items():
    for name, e in l:
        sv, t, r = stats(e)
        print(f"| {name} | {sp} | {sv:.1f} | {toe(t, sp)} | {r:.2f} |")

print("\n### Paired return difference of each w6 run vs the fixed actions [95% interval]\n")
print("| run | split | vs (-20, 250, 2) | vs (-20, 500, 10) |")
print("|---|---|---|---|")
for net, label, s in RUNS:
    for sp in ("train", "test"):
        n = load(f"{BLIND}/eps_{net}_w6_s{s}_{sp}_normal.csv")
        print(f"| {label} s{s} | {sp} | " + " | ".join(paired(n, e) for _, e in refs[sp]) + " |")
