# W6 experiment: reweighted impact penalty

Reward: the frozen environment's reward with the impact term reweighted to W_IMPACT 6, LAMBDA_BW 1.58
(`agent/reward.py`, `ReweightedImpact`). All returns below are under this reward. Seeds 1000-1199,
deterministic, 200 episodes per row.

Runs: `python -m agent.train --network {mlp,lstm} --run-name w6 --seed {0,1,2} --test-split
--set w_impact=6 lambda_bw=1.58 "anchor=[-14.0,250.6,2.0]"` (config v1, 300k steps). The evaluator's
`minus_reference` column in `results/*.csv` uses the reference under the original reward; ignore it for
these runs and use the paired differences below.

Blind: every observation replaced by the run's VecNormalize mean (so the network sees zeros after
normalisation). Toe peak on test is set by the ground's yield force, so it is the same for every policy.

Scripts: `experiments/w6/` (`refs.py` fixed-action references, `blind.py` normal and blind evaluation,
`fixed.py` fixed-action search, `findings.py` these tables). Intervals are 95% intervals of the mean
(`common.evaluate._mean_ci`); paired differences are per seed.

## Summary

1. On train, all six runs beat the best fixed action under this reward, (-20, 250, 2), by +3.6 to +9.3 (every interval above zero), and beat the finer-grid best (-16, 350, 0.5) by +4.8 to +10.4.
2. On test, five of six runs beat (-20, 250, 2), but against (-20, 500, 10) only MLP s2 and LSTM s2 are clearly better; MLP s1 is clearly worse (-2.10).
3. The networks use their input: blinded, every run collapses to the stiff corner (about -20 deg, K 500), survives more (91% on train) but loses 6.6 to 11.9 return on train because the toe peak rises to about 1.9 BW.
4. Normal policies sit at K 430-490 with low damping (B 1.3-2.5), not near the anchor K 250.6; LSTM s1 survives 76.5% on train, below the fixed action's 79%.
5. No fixed action in the finer grid (K 350-500, B 0.5-3) beats (-20, 250, 2); its best sits at the grid edge (K 350, B 0.5), so the fixed-action optimum lies at lower K.

### Normal vs blind

| run | split | survival % normal / blind | toe peak normal / blind | return normal / blind | blind − normal [95%] | mean action normal (θ, K, B) | mean action blind |
|---|---|---|---|---|---|---|---|
| MLP s0 | train | 85.5 / 91.0 | 1.732 / 1.942 | 85.60 / 76.44 | -9.16 [-14.08, -4.24] | -19.0, 454, 1.61 | -20.0, 500, 1.28 |
| MLP s0 | test | 52.0 / 53.5 | capped / capped | 79.52 / 77.30 | -2.22 [-5.53, +1.09] | -19.0, 480, 1.45 | -20.0, 500, 1.28 |
| MLP s1 | train | 81.5 / 91.0 | 1.667 / 1.933 | 84.96 / 76.83 | -8.13 [-13.52, -2.74] | -17.7, 441, 2.06 | -20.0, 500, 1.88 |
| MLP s1 | test | 47.0 / 52.0 | capped / capped | 76.47 / 76.77 | +0.30 [-3.08, +3.68] | -18.6, 445, 2.07 | -20.0, 500, 1.88 |
| MLP s2 | train | 87.5 / 91.0 | 1.783 / 1.942 | 84.27 / 76.44 | -7.83 [-12.11, -3.56] | -17.5, 466, 1.89 | -20.0, 500, 1.28 |
| MLP s2 | test | 56.0 / 53.5 | capped / capped | 81.30 / 77.30 | -4.00 [-7.20, -0.80] | -19.5, 488, 1.89 | -20.0, 500, 1.28 |
| LSTM s0 | train | 87.0 / 91.5 | 1.689 / 1.875 | 86.00 / 76.94 | -9.05 [-14.24, -3.87] | -18.9, 452, 2.48 | -19.9, 499, 3.06 |
| LSTM s0 | test | 51.5 / 54.0 | capped / capped | 79.36 / 79.11 | -0.25 [-2.88, +2.37] | -18.9, 473, 2.16 | -19.9, 498, 3.05 |
| LSTM s1 | train | 76.5 / 91.0 | 1.674 / 1.882 | 83.23 / 76.61 | -6.62 [-12.35, -0.89] | -17.4, 429, 2.15 | -19.9, 499, 2.01 |
| LSTM s1 | test | 49.5 / 53.0 | capped / capped | 78.76 / 78.68 | -0.08 [-2.90, +2.74] | -18.4, 457, 1.76 | -19.9, 499, 2.01 |
| LSTM s2 | train | 89.0 / 91.5 | 1.675 / 1.913 | 88.91 / 77.05 | -11.86 [-17.29, -6.43] | -18.6, 461, 1.63 | -20.0, 499, 2.46 |
| LSTM s2 | test | 53.5 / 52.5 | capped / capped | 80.72 / 77.54 | -3.18 [-6.35, -0.01] | -19.3, 474, 1.30 | -20.0, 499, 2.43 |

### Top 5 fixed actions on train (60-action grid plus the six w6 mean actions)

| rank | action (θ, K, B) | source | survival % | toe peak (BW) | mean return |
|---|---|---|---|---|---|
| 1 | (-16, 350, 0.5) | grid | 83.0 | 1.844 | 78.48 |
| 2 | (-16, 350, 1) | grid | 83.5 | 1.832 | 78.35 |
| 3 | (-16, 350, 1.5) | grid | 83.5 | 1.818 | 78.02 |
| 4 | (-16, 350, 3) | grid | 83.5 | 1.771 | 77.79 |
| 5 | (-16, 350, 2) | grid | 83.5 | 1.801 | 77.75 |

### Fixed-action references

| action | split | survival % | toe peak (BW) | mean return |
|---|---|---|---|---|
| (-20, 250, 2) | train | 79.0 | 1.747 | 79.61 |
| (-20, 500, 10) | train | 92.5 | 1.739 | 77.81 |
| (-20, 250, 2) | test | 44.5 | capped | 77.31 |
| (-20, 500, 10) | test | 50.5 | capped | 78.57 |

### Paired return difference of each w6 run vs the fixed actions [95% interval]

| run | split | vs (-20, 250, 2) | vs (-20, 500, 10) |
|---|---|---|---|
| MLP s0 | train | +5.99 [+3.52, +8.46] | +7.80 [+2.83, +12.76] |
| MLP s0 | test | +2.21 [+1.20, +3.21] | +0.95 [-0.79, +2.69] |
| MLP s1 | train | +5.35 [+2.24, +8.45] | +7.15 [+1.52, +12.78] |
| MLP s1 | test | -0.84 [-1.84, +0.16] | -2.10 [-3.99, -0.21] |
| MLP s2 | train | +4.66 [+2.56, +6.77] | +6.47 [+2.09, +10.84] |
| MLP s2 | test | +3.99 [+2.84, +5.14] | +2.73 [+1.03, +4.44] |
| LSTM s0 | train | +6.39 [+3.16, +9.61] | +8.19 [+2.83, +13.55] |
| LSTM s0 | test | +2.05 [+0.91, +3.19] | +0.79 [-0.69, +2.27] |
| LSTM s1 | train | +3.62 [+0.70, +6.54] | +5.43 [-0.10, +10.95] |
| LSTM s1 | test | +1.45 [+0.47, +2.42] | +0.19 [-1.61, +1.98] |
| LSTM s2 | train | +9.30 [+6.17, +12.43] | +11.11 [+5.48, +16.73] |
| LSTM s2 | test | +3.41 [+2.29, +4.53] | +2.15 [+0.39, +3.91] |
