"""The evaluation used for every number in the demo and the report.
run_episodes - plays one episode per seed with a policy and records, per episode: 
                return, steps, if it fell and why, the toe and total peak force (body weights), mean applied angle, stiffness K and damping B.
summarise      averages those into one row
"""
import math
import numpy as np
from common.env_setup import to_physical

EVAL_SEEDS = list(range(1000, 1200))
CEILING = 99.0                       # an episode with return >= 99 counts as "at the ceiling"
REFERENCE = (-20.0, 500.0, 10.0)     # best fixed action: degrees, N.m/rad, N.m.s/rad
INFO_KEYS = {"toe": None, "total": None, "reason": None}   # None = find the key by name

def _find_key(info, must, also=()):
    """Return the first scalar key of `info` whose name contains `must` (and every word in `also`)."""
    for k, v in info.items():
        kl = k.lower()
        if must in kl and all(a in kl for a in also) and np.isscalar(v):
            return k
    return None

def _reason(info):
    """
    Name why an episode ended, from the final step's `info`.
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
    """
    Play one episode per seed and return a list of per-episode dictionaries (EP_FIELDS).
    policy: any adapter from policies.py. 
    env: a built environment (with the same wrappers used in training).
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


# Adapted from standard implementations of the Wilson interval equation seen online.
def _wilson(k, n, z=1.96):
    """95% Wilson interval, in percent, for k successes out of n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h)))

# Standard normal distribution (Z-score = 1.96) approximation for a 95% confidence interval.
def _mean_ci(x):
    """Mean and normal-approximation 95% interval of a list of numbers."""
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    h = 1.96 * float(np.std(x, ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else math.nan
    return m, m - h, m + h

def summarise(eps, ref_eps=None):
    """Turn per-episode results into one row of the results table.
    ref_eps: the reference controller's episodes on the same seeds. 
    When given, the row includes `minus_reference`, the paired return difference with its 95% interval.
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
         "toe_peak_bw": float(np.mean([e["toe_peak"] for e in eps])),
         "total_peak_bw": float(np.mean([e["total_peak"] for e in eps])),
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