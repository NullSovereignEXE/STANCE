import sys, numpy as np, pandas as pd
T975 = {2: 4.303, 4: 2.776}  # two-sided 95% t quantiles by degrees of freedom
S = sys.argv[1]
d = pd.concat([pd.read_csv(f"{S}/fresh_lstm.csv"), pd.read_csv(f"{S}/fresh_rest.csv")])
g0 = d.groupby("policy").seed
assert d.sum_err.max() < 1e-4 and (g0.min() == 3000).all() and (g0.max() == 3499).all() and (g0.nunique() == 500).all() and (g0.size() == 500).all()
print("seeds per policy:", d.groupby("policy").seed.agg(["min", "max", "nunique"]).drop_duplicates().to_dict("records"))
REF = "fixed:-20/500/10"
ref = d[d.policy == REF].set_index("seed")
def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)); return m, m - h, m + h
def wilson(k, n, z=1.96):
    p = k / n; dd = 1 + z*z/n; c = (p + z*z/(2*n)) / dd; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / dd
    return 100*(c-h), 100*(c+h)
f = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"
rows = []
for p, g in d.groupby("policy", sort=False):
    g = g.set_index("seed").loc[ref.index]
    n, k = len(g), g.survived.sum()
    dr = g.ret - ref.ret; ds = 100 * (g.survived - ref.survived)
    rows.append(dict(policy=p, n=n, surv=100*k/n, surv_ci="[%.1f, %.1f]" % wilson(k, n), toe=g.toe_peak.mean(),
                     ret=g.ret.mean(), dret=ci(dr), dsurv=ci(ds), _dr=dr.values, _ds=ds.values))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
t = pd.DataFrame(rows)
out = t.drop(columns=["_dr", "_ds"]).copy(); out["dret"] = out.dret.map(f); out["dsurv"] = out.dsurv.map(f)
print(out.round(3).to_string(index=False))
for net in ("networkB_lstm", "networkA_mlp"):
    sub = t[t.policy.str.contains(net)]; k = len(sub)
    tq = T975[k - 1]
    for col, raw in (("surv", None), ("toe", None), ("ret", None), ("dret", "_dr"), ("dsurv", "_ds")):
        vals = np.array([v[0] if isinstance(v, tuple) else v for v in sub[col]])
        m, h = vals.mean(), tq * vals.std(ddof=1) / np.sqrt(k)
        line = f"{net} {k}-seed mean {col}: {m:+.3f}  seed-level t95 [{m-h:+.3f}, {m+h:+.3f}]"
        if raw:
            pooled = np.concatenate(sub[raw].values)
            line += f"  pooled-episode 95% {f(ci(pooled))} (n={len(pooled)})"
            line += f"  seeds with CI above 0: {sum(v[1] > 0 for v in sub[col])}, below 0: {sum(v[2] < 0 for v in sub[col])}"
        print(line)
