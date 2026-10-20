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
