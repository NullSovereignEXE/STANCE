# P3 agent experiments: findings

All numbers are read by code from `results/P3_agent_experiments.csv` and `results/reference_train_1000-1199.csv`. Final evaluation: 200 episodes, seeds 1000-1199, deterministic policy, frozen environment (STANCE/Gym). Network: networkA_mlp, 300k steps unless stated. Runs named nb_* and smoke_* are excluded.

| fixed action (-20 deg, K 500, B 10) | survival_pct | toe_peak_bw | mean_return |
|---|---|---|---|
| reference | 92.5 | 1.739 | 87.90 |

## base

Setup: `--set (none)`

| metric | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| survival_pct | 88.5 | 57.5 | 17.0 | 54.3 |
| toe_peak_bw | 1.938 | 1.604 | 1.604 | 1.715 |
| minus_reference | -2.65 [-3.91, -1.39] | -14.34 [-17.82, -10.86] | -29.19 [-32.43, -25.95] | -15.39 |

base vs fixed action: survival -38.2 points, toe peak -0.023 BW (means over seeds).

## nonr

Setup: `--set norm_reward=False`

| metric | seed 0 | mean |
|---|---|---|
| survival_pct | 84.0 | 84.0 |
| toe_peak_bw | 1.766 | 1.766 |
| minus_reference | -5.81 [-7.60, -4.02] | -5.81 |

nonr vs base: survival +29.7 points, toe peak +0.050 BW, minus_reference +9.58 (means over seeds; nonr seeds [0], base seeds [0, 1, 2]).

## g995

Setup: `--set gamma=0.995`

| metric | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| survival_pct | 85.5 | 49.0 | 32.5 | 55.7 |
| toe_peak_bw | 1.745 | 1.602 | 1.629 | 1.659 |
| minus_reference | -6.58 [-8.34, -4.82] | -18.87 [-22.94, -14.79] | -25.45 [-28.97, -21.93] | -16.97 |

g995 vs base: survival +1.3 points, toe peak -0.057 BW, minus_reference -1.57 (means over seeds; g995 seeds [0, 1, 2], base seeds [0, 1, 2]).

## lowlr

Setup: `--set learning_rate=1e-4 n_epochs=5`

| metric | seed 0 | mean |
|---|---|---|
| survival_pct | 60.0 | 60.0 |
| toe_peak_bw | 1.659 | 1.659 |
| minus_reference | -18.83 [-21.92, -15.74] | -18.83 |

lowlr vs base: survival +5.7 points, toe peak -0.056 BW, minus_reference -3.44 (means over seeds; lowlr seeds [0], base seeds [0, 1, 2]).

## ent

Setup: `--set ent_coef=0.01`

| metric | seed 0 | mean |
|---|---|---|
| survival_pct | 60.0 | 60.0 |
| toe_peak_bw | 1.667 | 1.667 |
| minus_reference | -15.40 [-18.91, -11.88] | -15.40 |

ent vs base: survival +5.7 points, toe peak -0.048 BW, minus_reference -0.01 (means over seeds; ent seeds [0], base seeds [0, 1, 2]).

## std

Setup: `--set log_std_init=-1.0`

| metric | seed 0 | mean |
|---|---|---|
| survival_pct | 38.5 | 38.5 |
| toe_peak_bw | 1.544 | 1.544 |
| minus_reference | -17.63 [-21.59, -13.67] | -17.63 |

std vs base: survival -15.8 points, toe peak -0.172 BW, minus_reference -2.24 (means over seeds; std seeds [0], base seeds [0, 1, 2]).

## resid

Setup: `--set residual=True`

| metric | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| survival_pct | 90.0 | 91.0 | 90.0 | 90.3 |
| toe_peak_bw | 1.714 | 1.800 | 1.728 | 1.747 |
| minus_reference | +0.19 [-1.06, +1.44] | -1.52 [-2.53, -0.52] | -2.34 [-3.23, -1.45] | -1.22 |

resid vs base: survival +36.0 points, toe peak +0.032 BW, minus_reference +14.17 (means over seeds; resid seeds [0, 1, 2], base seeds [0, 1, 2]).

## resid_in

Setup: `--set residual=True anchor=[-14.0,250.6,6.38]`

| metric | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| survival_pct | 91.5 | 87.0 | 86.5 | 88.3 |
| toe_peak_bw | 1.751 | 1.713 | 1.695 | 1.720 |
| minus_reference | +0.93 [-1.04, +2.89] | +1.67 [-0.59, +3.93] | +0.72 [-1.54, +2.97] | +1.11 |

resid_in vs resid: survival -2.0 points, toe peak -0.028 BW, minus_reference +2.33 (means over seeds; resid_in seeds [0, 1, 2], resid seeds [0, 1, 2]).

## resid_sc01

Setup: `--set residual=True scale=[0.1,0.1,0.1]`

| metric | seed 0 | mean |
|---|---|---|
| survival_pct | 91.5 | 91.5 |
| toe_peak_bw | 1.734 | 1.734 |
| minus_reference | -0.67 [-1.14, -0.20] | -0.67 |

resid_sc01 vs resid: survival +1.2 points, toe peak -0.013 BW, minus_reference +0.55 (means over seeds; resid_sc01 seeds [0], resid seeds [0, 1, 2]).

## resid_sc06

Setup: `--set residual=True scale=[0.6,0.6,0.6]`

| metric | seed 0 | mean |
|---|---|---|
| survival_pct | 92.0 | 92.0 |
| toe_peak_bw | 1.858 | 1.858 |
| minus_reference | -0.33 [-1.25, +0.59] | -0.33 |

resid_sc06 vs resid: survival +1.7 points, toe peak +0.110 BW, minus_reference +0.89 (means over seeds; resid_sc06 seeds [0], resid seeds [0, 1, 2]).

## resid_in_std

Setup: `--set residual=True anchor=[-14.0,250.6,6.38] log_std_init=-1.0`

| metric | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| survival_pct | 84.5 | 73.5 | 75.0 | 77.7 |
| toe_peak_bw | 1.665 | 1.622 | 1.618 | 1.635 |
| minus_reference | +0.74 [-1.63, +3.11] | -3.59 [-6.40, -0.77] | -2.46 [-5.30, +0.38] | -1.77 |

resid_in_std vs resid_in: survival -10.7 points, toe peak -0.084 BW, minus_reference -2.88 (means over seeds; resid_in_std seeds [0, 1, 2], resid_in seeds [0, 1, 2]).

## resid_in_1m

Setup: `--set residual=True anchor=[-14.0,250.6,6.38] total_steps=1000000`

| metric | seed 0 | seed 1 | mean |
|---|---|---|---|
| survival_pct | 90.0 | 85.5 | 87.8 |
| toe_peak_bw | 1.759 | 1.703 | 1.731 |
| minus_reference | +0.14 [-1.58, +1.86] | +0.22 [-2.04, +2.48] | +0.18 |

resid_in_1m vs resid_in: survival -0.6 points, toe peak +0.012 BW, minus_reference -0.93 (means over seeds; resid_in_1m seeds [0, 1], resid_in seeds [0, 1, 2]).

---

# Traceable findings for the report

The sections above come from `Claude-Files/Network_Agent/results/P3_agent_experiments.csv`, which stays in its old folder.
The sections below use the frozen environment (`Gym/`) with its original reward, the final runs `runs/<network>/final_s0..s2`
(300k steps, config v1), deterministic actions, and fixed actions written (θ deg, K N·m/rad, B N·m·s/rad). Scripts are in
`results/analysis/` (see its README). Per-episode CSVs are not committed.

## 1. Fixed-action grid (45 actions), best single actions, per-seed hindsight ceiling

**Output missing.** The grid table and the per-seed ceiling (93.5% survival on train) were not saved and no script remains.
Not recomputed.

## 2. Two-phase schedules (36)

**Output missing.** The schedule table (0 of 36 beat the best constant action; 19 clearly worse; best +0.33) was not saved
and no script remains. Not recomputed.

## 3. Reward weightings (20), offline re-scoring

**Output missing.** The re-scoring table and the list of 20 (W_IMPACT, LAMBDA_BW) pairs were not saved. The per-step toe
forces it would need do exist for the final runs and the three fixed actions (section 6c, `decomp2.py`), so the table can
be rebuilt offline without new simulation. Not recomputed.

## 4. Survival vs toe-peak trade-off (non-dominated actions)

**Output missing.** Built from the section 1 grid, which was not saved. Not recomputed.

## 5. Re-evaluation differences between machines

**Output missing.** The comparison (return -0.22 to +0.27, survival up to ±0.5) was not saved. Not recomputed.

## 6a. Final runs

Seeds 1000-1199, 200 episodes per row, train and test ground. Source: `results/networkA_mlp.csv` and
`results/networkB_lstm.csv` (rows `final_s*`), written by `python -m agent.train` through `common/evaluate.py`.
Both networks reach 86-92% survival on train against 92.5% for the fixed action, with a higher mean return.

| run | split | survival % [95%] | toe peak (BW) | mean return | return − (-20, 500, 10) [95%] |
|---|---|---|---|---|---|
| MLP s0 | train | 91.5 [86.8, 94.6] | 1.751 | 88.83 | +0.93 [-1.04, +2.89] |
| MLP s1 | train | 87.0 [81.6, 91.0] | 1.713 | 89.57 | +1.67 [-0.59, +3.93] |
| MLP s2 | train | 86.5 [81.1, 90.6] | 1.695 | 88.62 | +0.72 [-1.54, +2.97] |
| MLP mean | train | 88.3 | 1.720 | 89.01 | |
| LSTM s0 | train | 92.0 [87.4, 95.0] | 1.747 | 89.76 | +1.86 [+0.56, +3.17] |
| LSTM s1 | train | 88.5 [83.3, 92.2] | 1.668 | 91.06 | +3.16 [+0.97, +5.34] |
| LSTM s2 | train | 90.0 [85.1, 93.4] | 1.682 | 91.80 | +3.90 [+1.50, +6.30] |
| LSTM mean | train | 90.2 | 1.699 | 90.87 | |
| (-20, 500, 10) | train | 92.5 | 1.739 | 87.90 | |
| MLP s0 | test | 51.0 [44.1, 57.8] | 0.704 | 79.31 | +0.74 [-0.95, +2.44] |
| MLP s1 | test | 48.0 [41.2, 54.9] | 0.704 | 78.37 | -0.20 [-1.96, +1.56] |
| MLP s2 | test | 45.0 [38.3, 51.9] | 0.704 | 77.09 | -1.48 [-3.39, +0.42] |
| MLP mean | test | 48.0 | 0.704 | 78.26 | |
| LSTM s0 | test | 52.5 [45.6, 59.3] | 0.704 | 79.20 | +0.63 [-0.67, +1.93] |
| LSTM s1 | test | 40.5 [33.9, 47.4] | 0.704 | 76.20 | -2.37 [-4.40, -0.34] |
| LSTM s2 | test | 40.0 [33.5, 46.9] | 0.704 | 75.93 | -2.64 [-4.59, -0.69] |
| LSTM mean | test | 44.3 | 0.704 | 77.11 | |
| (-20, 500, 10) | test | 50.5 | 0.704 | 78.57 | |

On the test ground the toe peak is the same for every policy (set at touchdown by the ground's yield force).

## 6b. Fresh seeds

Seeds 3000-3499, 500 episodes per row, train ground. Script: `decomp3.py`, then `agg3.py`. Final seeds 0-2 only
(`final_s3` and `final_s4` do not exist). Paired differences are per episode against (-20, 500, 10) on the same seed,
mean with a normal 95% interval; survival interval is Wilson. On fresh seeds the LSTM keeps a higher return with slightly
lower survival; the MLP survives less and its return gain is not significant.

| policy | survival % [95%] | toe peak (BW) | mean return | Δ return [95%] | Δ survival, points [95%] |
|---|---|---|---|---|---|
| LSTM s0 | 92.6 [90.0, 94.6] | 1.746 | 90.83 | +1.40 [+0.72, +2.07] | -0.2 [-1.1, +0.7] |
| LSTM s1 | 90.4 [87.5, 92.7] | 1.663 | 91.86 | +2.42 [+1.19, +3.65] | -2.4 [-3.7, -1.1] |
| LSTM s2 | 90.8 [87.9, 93.0] | 1.666 | 92.04 | +2.61 [+1.30, +3.91] | -2.0 [-3.2, -0.8] |
| LSTM mean (t over 3 seeds) | 91.3 | 1.691 | 91.58 | +2.14 [+0.52, +3.76] | -1.5 [-4.4, +1.4] |
| MLP s0 | 91.2 [88.4, 93.4] | 1.753 | 89.43 | -0.01 [-1.00, +0.98] | -1.6 [-3.1, -0.1] |
| MLP s1 | 90.4 [87.5, 92.7] | 1.720 | 90.50 | +1.06 [-0.15, +2.26] | -2.4 [-3.9, -1.0] |
| MLP s2 | 88.6 [85.5, 91.1] | 1.683 | 89.67 | +0.24 [-0.96, +1.43] | -4.2 [-6.1, -2.4] |
| MLP mean (t over 3 seeds) | 90.1 | 1.719 | 89.87 | +0.43 [-0.96, +1.82] | -2.7 [-6.0, +0.6] |
| (-20, 500, 10) | 92.8 [90.2, 94.8] | 1.738 | 89.44 | 0 | 0 |
| (-20, 500, 6) | 93.0 [90.4, 94.9] | 1.845 | 89.35 | -0.09 [-0.56, +0.39] | +0.2 [-0.8, +1.2] |
| (-20, 375, 10) | 91.2 [88.4, 93.4] | 1.672 | 88.49 | -0.95 [-1.52, -0.38] | -1.6 [-2.7, -0.5] |

## 6c. Reward-term decomposition

Seeds 1000-1199, train ground, 200 episodes per run (600 per network over seeds 0-2, 200 per fixed action).
Scripts: `decomp.py`, then `agg.py` (terms); `decomp2.py`, then `agg2.py` (per-step toe force). Per-step reward =
survival (+1) + impact (-2.94 · max(0, toe/BW - 1.58)², every step) + smoothness (-0.5 · ‖Δa‖², from step 2) + fall
(-25, once); the terms sum to the episode return within 2e-13, and the returns match the committed `episodes_train.csv`
exactly. The networks' return gain comes only from a smaller impact penalty; they spend as many steps above 1.58 BW as the
fixed action but exceed it by less.

Mean per episode, 3-seed mean for the networks:

| policy | return | survival | impact | smoothness | fall | survived % | toe peak (BW) |
|---|---|---|---|---|---|---|---|
| LSTM | 90.87 | 99.36 | -5.69 | -0.34 | -2.46 | 90.2 | 1.699 |
| MLP | 89.01 | 99.04 | -5.43 | -1.68 | -2.92 | 88.3 | 1.720 |
| (-20, 500, 10) | 87.90 | 99.48 | -9.70 | 0 | -1.88 | 92.5 | 1.739 |
| (-20, 375, 10) | 87.74 | 99.26 | -9.27 | 0 | -2.25 | 91.0 | 1.676 |
| (-20, 500, 6) | 88.16 | 99.51 | -9.59 | 0 | -1.75 | 93.0 | 1.835 |

Difference to (-20, 500, 10):

| policy | return | survival | impact | smoothness | fall |
|---|---|---|---|---|---|
| LSTM | +2.97 | -0.12 | +4.01 | -0.34 | -0.58 |
| MLP | +1.11 | -0.44 | +4.27 | -1.68 | -1.04 |

Per-step toe force above 1.58 BW, mean per episode (all episodes / surviving episodes only):

| policy | steps above 1.58 BW | sum of excess (BW) | excess in steps 1-10 (BW) |
|---|---|---|---|
| LSTM | 15.87 / 16.83 | 3.89 / 4.07 | 0.91 / 0.94 |
| MLP | 15.82 / 16.26 | 3.96 / 4.00 | 0.79 / 0.78 |
| (-20, 500, 10) | 15.63 / 16.79 | 4.84 / 5.22 | 1.20 / 1.28 |
| (-20, 375, 10) | 16.65 / 17.79 | 5.13 / 5.54 | 1.18 / 1.27 |
| (-20, 500, 6) | 15.42 / 16.18 | 4.82 / 5.03 | 1.26 / 1.31 |

Surviving episodes: LSTM 541, MLP 530, (-20, 500, 10) 185, (-20, 375, 10) 182, (-20, 500, 6) 186.

## 6d. Blind-input test of the final runs

**Output missing.** No blind-input evaluation of the final runs was saved. Not recomputed.
