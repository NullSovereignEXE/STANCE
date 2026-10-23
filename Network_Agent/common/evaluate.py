"""The shared evaluation protocol. Every result in the reports comes from this file, unchanged.

Protocol
    200 episodes on seeds 1000-1199, deterministic policy, training ground unless the
    split is "test". Seeds 2000-2049 are used only for learning curves, so the table
    seeds are never seen during training. Every policy is compared with the best fixed
    action (REFERENCE) on the same seeds.

What one row contains (see `summarise`)
    survival and its 95% interval, falls by reason, mean steps, toe and total peak
    force in body weights, mean applied angle / K / B, mean return with interval, the
    share of episodes at the return ceiling, and the paired return difference to the
    reference.

Command line
    python -m common.evaluate --show-info    list the environment's info keys
    python -m common.evaluate --reference    score the best fixed action

Uses: the environment through env_setup.make_env, and the policy adapters in policies.py.
The environment's `info` dictionary is searched for the toe force and the failure reason;
see INFO_KEYS.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

from common.env_setup import make_env, to_norm_action, to_physical
from common.policies import FixedPolicy

EVAL_SEEDS = list(range(1000, 1200))
CURVE_SEEDS = list(range(2000, 2050))
CEILING = 99.0
REFERENCE = (-20.0, 500.0, 10.0)
RESULTS = Path(__file__).resolve().parent.parent / "results"
INFO_KEYS = {"toe": None, "total": None, "reason": None}
EP_FIELDS = ["seed", "ret", "steps", "terminated", "reason",
             "toe_peak", "total_peak", "theta", "K", "B"]


def _find_key(info, must, also=()):
    """Return the first scalar key of `info` whose name contains `must` (and every word in `also`)."""
    for k, v in info.items():
        kl = k.lower()
        if must in kl and all(a in kl for a in also) and np.isscalar(v):
            return k
    return None


def _reason(info):
    """Name why an episode ended, from the final step's `info`.

    Uses the environment's own reason key if one exists (it may appear only on the final
    step, so it is searched for here and not on the first step). Otherwise it labels the
    fall "forward" or "back" from the sign of the shank tilt, so a collapse without a
    reason key is counted under one of those two.
    """
    key = INFO_KEYS["reason"] or _find_key(info, "reason")
    if key is not None:
        return str(info[key])
    tilt = info.get("leg_tilt_deg")
    if tilt is None:
        return "fall"
    return "forward" if tilt > 0 else "back"


def run_episodes(policy, env, seeds=EVAL_SEEDS):
    """Play one episode per seed and return a list of per-episode dictionaries (EP_FIELDS).

    policy: any adapter from policies.py. env: a built environment (with the same
    wrappers used in training). The toe and total peaks are the largest value seen in
    the episode. Angle, K and B are averaged over the episode from the action the
    environment actually applied, which includes any residual offset.
    """
    eps, keys = [], None
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        policy.reset()
        ret, steps, done = 0.0, 0, False
        toe, total, phys = [], [], []
        while not done:
            a = policy.act(obs)
            obs, r, term, trunc, info = env.step(a)
            if keys is None:
                keys = {"toe": INFO_KEYS["toe"] or _find_key(info, "toe"),
                        "total": INFO_KEYS["total"] or ("normal_force_bw" if "normal_force_bw" in info
                                                        else _find_key(info, "total", ("bw",)))}
            applied = getattr(env.unwrapped, "prev_action", a)
            phys.append(to_physical(env, applied))
            if keys["toe"]:
                toe.append(float(info[keys["toe"]]))
            if keys["total"]:
                total.append(float(info[keys["total"]]))
            ret += float(r)
            steps += 1
            done = term or trunc
        th, K, B = np.mean(phys, axis=0)
        eps.append({"seed": int(s), "ret": ret, "steps": steps,
                    "terminated": int(term), "reason": _reason(info) if term else "",
                    "toe_peak": max(toe) if toe else math.nan,
                    "total_peak": max(total) if total else math.nan,
                    "theta": th, "K": K, "B": B})
    return eps


def _wilson(k, n, z=1.96):
    """95% Wilson interval, in percent, for k successes out of n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h)))


def _mean_ci(x):
    """Mean and normal-approximation 95% interval of a list of numbers."""
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    h = 1.96 * float(np.std(x, ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else math.nan
    return m, m - h, m + h


def _nanmean(x):
    """Mean ignoring NaN; NaN if every value is NaN (for example when no toe key exists)."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x[~np.isnan(x)])) if np.any(~np.isnan(x)) else math.nan


def summarise(eps, ref_eps=None):
    """Turn per-episode results into one row of the results table.

    ref_eps: the reference controller's episodes on the same seeds. When given, the row
    includes `minus_reference`, the paired return difference with its 95% interval.
    """
    n = len(eps)
    surv = sum(1 - e["terminated"] for e in eps)
    reasons = {}
    for e in eps:
        if e["terminated"]:
            reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    m, lo, hi = _mean_ci([e["ret"] for e in eps])
    s = {"n": n,
         "survival_pct": 100 * surv / n,
         "survival_ci": "[%.1f, %.1f]" % _wilson(surv, n),
         "falls_by_reason": " ".join(f"{k}:{v}" for k, v in sorted(reasons.items())) or "-",
         "mean_steps": float(np.mean([e["steps"] for e in eps])),
         "toe_peak_bw": _nanmean([e["toe_peak"] for e in eps]),
         "total_peak_bw": _nanmean([e["total_peak"] for e in eps]),
         "mean_theta_deg": float(np.mean([e["theta"] for e in eps])),
         "mean_K": float(np.mean([e["K"] for e in eps])),
         "mean_B": float(np.mean([e["B"] for e in eps])),
         "mean_return": m,
         "return_ci": f"[{lo:.2f}, {hi:.2f}]",
         "ceiling_pct": float(100 * np.mean([e["ret"] >= CEILING for e in eps]))}
    if ref_eps is not None:
        ref = {e["seed"]: e["ret"] for e in ref_eps}
        d = [e["ret"] - ref[e["seed"]] for e in eps if e["seed"] in ref]
        if d:
            m, lo, hi = _mean_ci(d)
            s["minus_reference"] = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    return s


def print_summary(name, s):
    """Print one summary row as a readable list."""
    print(f"\n== {name} ({s['n']} episodes) ==")
    for k, v in s.items():
        if k != "n":
            print(f"  {k:18s} {v:.3f}" if isinstance(v, float) else f"  {k:18s} {v}")


def save_episodes(path, eps):
    """Write the per-episode results to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EP_FIELDS)
        w.writeheader()
        w.writerows(eps)


def load_episodes(path):
    """Read a CSV written by `save_episodes` back into the same list of dictionaries."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in EP_FIELDS:
            if k != "reason":
                r[k] = float(r[k])
        r["seed"], r["steps"], r["terminated"] = int(r["seed"]), int(r["steps"]), int(r["terminated"])
    return rows


def append_row(csv_path, row):
    """Append one row to a results CSV, writing the header first if the file is new."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})


def reference_episodes(split="train", seeds=EVAL_SEEDS):
    """Episodes of the best fixed action (REFERENCE) on the given seeds.

    The result is cached in results/reference_<split>_<first>-<last>.csv. Delete that
    file whenever the environment or its reward changes, so the reference is rerun.
    """
    path = RESULTS / f"reference_{split}_{seeds[0]}-{seeds[-1]}.csv"
    if path.exists():
        return load_episodes(path)
    env = make_env(split)
    eps = run_episodes(FixedPolicy(to_norm_action(env, *REFERENCE)), env, seeds)
    env.close()
    save_episodes(path, eps)
    return eps


def evaluate_and_record(name, run_name, policy, env, split="train", seeds=EVAL_SEEDS,
                        run_dir=None, extra=None):
    """Run the protocol, print the summary and record it.

    name: the network's name; the row is appended to results/<name>.csv.
    run_dir: if given, the per-episode CSV is saved there as episodes_<split>.csv.
    extra: extra columns for the row (seed, steps, parameter count).
    Returns the summary dictionary.
    """
    eps = run_episodes(policy, env, seeds)
    s = summarise(eps, reference_episodes(split, seeds))
    print_summary(f"{name}/{run_name} [{split}]", s)
    if run_dir is not None:
        save_episodes(Path(run_dir) / f"episodes_{split}.csv", eps)
    append_row(RESULTS / f"{name}.csv", {"run": run_name, "split": split, **(extra or {}), **s})
    return s


def _main():
    """Command line: --show-info prints the info keys, --reference scores the fixed action."""
    p = argparse.ArgumentParser()
    p.add_argument("--show-info", action="store_true")
    p.add_argument("--reference", action="store_true")
    p.add_argument("--split", default="train")
    p.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    a = p.parse_args()
    if a.show_info:
        env = make_env(a.split)
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample())
        print("obs shape:", env.observation_space.shape, " action shape:", env.action_space.shape)
        for k, v in info.items():
            print(f"  {k:25s} {v}")
        print("detected:", {"toe": INFO_KEYS["toe"] or _find_key(info, "toe"),
                            "reason": INFO_KEYS["reason"] or _find_key(info, "reason")},
              "(None for toe means toe_peak_bw will be nan; set INFO_KEYS)")
    if a.reference:
        seeds = EVAL_SEEDS[: a.episodes]
        s = summarise(reference_episodes(a.split, seeds))
        print_summary(f"reference {REFERENCE} [{a.split}]", s)


if __name__ == "__main__":
    _main()
