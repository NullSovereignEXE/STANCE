# Analysis scripts for P3_findings.md (sections 6b and 6c)

Run from `Network_Agent/` with `env -u PYTHONPATH` and the frozen `Gym/` (or `STANCE_GYM_PATH`). OUT is any folder outside the repo. The agg scripts need pandas, which is not in requirements.txt.
Fixed actions are written `fixed:theta,K,B`. Final runs: `runs/networkB_lstm/final_s{0,1,2}` and `runs/networkA_mlp/final_s{0,1,2}`.

- 6c terms, seeds 1000-1199: `python results/analysis/decomp.py OUT/lstm.csv <LSTM runs>` and `python results/analysis/decomp.py OUT/rest.csv <MLP runs> fixed:-20,500,10 fixed:-20,375,10 fixed:-20,500,6`.
  Then `sed -i -E 's#^fixed:(-?[0-9.]+),([0-9.]+),([0-9.]+),#fixed:\1/\2/\3,#' OUT/rest.csv` and `python results/analysis/agg.py OUT`.
- 6c per-step toe force, same seeds: `decomp2.py` with the same arguments to `OUT/lstm2.csv` and `OUT/rest2.csv` (it also writes `*_steps.csv`), then `python results/analysis/agg2.py OUT`.
- 6b fresh seeds 3000-3499: `SEED_LO=3000 SEED_HI=3499 python results/analysis/decomp3.py OUT/fresh_lstm.csv <LSTM runs> fixed:-20,500,10`,
  the same to `OUT/fresh_rest.csv` with `<MLP runs> fixed:-20,500,6 fixed:-20,375,10`, then `python results/analysis/agg3.py OUT`.
- 6a is written by `python -m agent.train` into `results/<network>.csv`; it has no script here.
