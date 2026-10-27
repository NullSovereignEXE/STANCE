import pandas as pd, numpy as np, sys
S=sys.argv[1]
d=pd.concat([pd.read_csv(f"{S}/lstm.csv"),pd.read_csv(f"{S}/rest.csv")])
print("max per-episode |sum-ret|:",d.sum_err.max()," max per-step:",d.step_err.max())
# check vs committed episodes_train.csv
for p in d.policy.unique():
    if p.startswith("runs"):
        c=pd.read_csv(f"{p}/episodes_train.csv"); m=d[d.policy==p].merge(c,on="seed",suffixes=("","_c"))
        print(p,"max |ret-committed|=%.2e"%(m.ret-m.ret_c).abs().max(), "steps match:",(m.steps==m.steps_c).all())
d["group"]=d.policy.str.replace(r"runs/networkB_lstm/final_s\d","LSTM",regex=True).str.replace(r"runs/networkA_mlp/final_s\d","MLP",regex=True).str.replace("fixed:","")
d["survived"]=100*d.survived
cols=["ret","survival","impact","smooth","fall","steps","survived","toe_peak"]
per=d.groupby(["group","policy"])[cols].mean()
m=per.groupby("group").mean(); s=per.groupby("group").std()
order=["LSTM","MLP","-20/500/10","-20/375/10","-20/500/6"]
m=m.loc[order]; s=s.loc[order]
pd.set_option("display.width",200); pd.set_option("display.precision",3)
print("\nMEAN per episode (3-seed mean of 200-ep means)\n",m)
print("\nSTD across seeds\n",s.loc[["LSTM","MLP"]])
for ref in ["-20/500/10","-20/375/10"]:
    print(f"\nDIFF vs {ref}\n",(m-m.loc[ref])[["ret","survival","impact","smooth","fall","steps","survived","toe_peak"]])
print("\nper-seed\n",per)
# impact split: survivors vs fallers, and fall count
d["fell"]=100-d.survived
print("\nimpact on first step share:")
