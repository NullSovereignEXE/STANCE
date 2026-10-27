import sys, numpy as np, pandas as pd
S = sys.argv[1]; L = 1.58; W = 2.94
ep = pd.concat([pd.read_csv(f"{S}/{n}2.csv") for n in ("lstm", "rest")])
st = pd.concat([pd.read_csv(f"{S}/{n}2_steps.csv") for n in ("lstm", "rest")])
old = pd.concat([pd.read_csv(f"{S}/lstm.csv"), pd.read_csv(f"{S}/rest.csv")])
old["policy"] = old.policy.str.replace(",", "/")
m = ep.merge(old, on=["policy", "seed"], suffixes=("", "_old"))
print("rows", len(ep), "| max |ret - previous run|:", (m.ret - m.ret_old).abs().max())
st["ex"] = (st.toe_force_bw - L).clip(lower=0)
st["above"] = st.toe_force_bw > L
st["ex_early"] = st.ex.where(st.step <= 10, 0.0)
st["ex2"] = st.ex ** 2
a = st.groupby(["policy", "seed"]).agg(n_above=("above", "sum"), sum_ex=("ex", "sum"),
                                       sum_ex_early=("ex_early", "sum"), sum_ex2=("ex2", "sum"),
                                       n=("step", "size"), peak=("toe_force_bw", "max")).reset_index()
a = a.merge(ep, on=["policy", "seed"])
print("steps match:", (a.n == a.steps).all(), "| max |-2.94*sum ex^2 - impact|:", (-W * a.sum_ex2 - a.impact).abs().max(),
      "| max |peak - toe_peak|:", (a.peak - a.toe_peak).abs().max())
a["group"] = a.policy.str.replace(r"runs/networkB_lstm/final_s\d", "LSTM", regex=True) \
                     .str.replace(r"runs/networkA_mlp/final_s\d", "MLP", regex=True).str.replace("fixed:", "")
order = ["LSTM", "MLP", "-20/500/10", "-20/375/10", "-20/500/6"]
cols = ["n_above", "sum_ex", "sum_ex_early", "impact", "toe_peak"]
pd.set_option("display.width", 200); pd.set_option("display.precision", 3)
for name, sub in (("ALL EPISODES", a), ("SURVIVORS ONLY", a[a.survived == 1])):
    per = sub.groupby(["group", "policy"])[cols].mean()
    mean = per.groupby("group").mean().loc[order]; sd = per.groupby("group").std().loc[order]
    cnt = sub.groupby("group").size().loc[order]
    print(f"\n{name} (3-seed mean of per-run means; n episodes = {dict(cnt)})\n", mean)
    print("std across seeds\n", sd.loc[["LSTM", "MLP"]])
    for ref in ("-20/500/10", "-20/375/10"):
        print(f"diff vs {ref}\n", (mean - mean.loc[ref]).loc[["LSTM", "MLP"]])
    print("per seed\n", per[["n_above", "sum_ex"]])
