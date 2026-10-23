# Config v1 in plain words

`agent/config.py` holds every setting for **how training runs**. The network files (`networkA_mlp/`,
`networkB_lstm/`) say what the layers are. The environment says what the world is. This file is the third part.

You do not edit it. Use it as it is, and change a value for one run with `--set`:

```bash
python -m agent.train --network mlp --run-name myrun --set gamma=0.995
python -m agent.train --network mlp --run-name plain --set residual=False
```

## What config v1 changes: the residual action

Before, the network chose the whole action: target angle, stiffness K and damping B. Now it chooses a small
**nudge** from a starting action (the anchor):

`action = anchor + scale x nudge`

| Setting | Value | Meaning |
|---|---|---|
| `residual` | `True` | use the nudge instead of the whole action |
| `anchor` | `[-14.0, 250.6, 6.38]` | the starting stance: angle in degrees, K, B. A network that outputs zero behaves like this |
| `scale` | `[0.3, 0.3, 0.3]` | the largest nudge is 0.3 in normalised units on [-1, 1], which is 15% of the full range (in log units for K and B), in either direction |

The anchor sits inside the allowed range so nudges work in both directions. The best fixed action
(-20 deg, K 500, B 10) sits on the edge of every range, where half of each nudge was cut off.

## Why we use it

The same plain PPO setup, three training seeds, 200 evaluation episodes each:

| | Survival | Toe peak |
|---|---|---|
| Best fixed action (no learning) | 92.5% | 1.74 BW |
| Plain PPO, seeds 0 / 1 / 2 | 88.5% / 57.5% / 17.0% | 1.6 to 1.9 BW |
| Residual (config v1), seeds 0 / 1 / 2 | 91.5% / 87.0% / 86.5% | 1.69 to 1.75 BW |

Plain PPO depends heavily on the training seed. Starting from a good action makes training reliable.
Longer training (1M steps) and less exploration noise did not help.

**What it does not mean:** the learned policy is tied with the best fixed action, not better. Judge your
network by whether it matches survival while lowering toe force, and always run at least 2 seeds.

## The rest: standard PPO settings

| Setting | Value | Plain meaning |
|---|---|---|
| `total_steps` | 300,000 | training budget (about 6 minutes for Network A) |
| `n_envs`, `n_steps` | 8, 256 | 2,048 samples are collected before each update |
| `batch_size`, `n_epochs` | 512, 10 | each update runs 4 mini-batches for 10 passes |
| `learning_rate` | 3e-4 | size of each gradient step |
| `gamma` | 0.99 | how much later reward counts |
| `gae_lambda` | 0.95 | smoothing of the advantage estimate |
| `clip_range` | 0.2 | stops one update from changing the policy too much |
| `ent_coef`, `vf_coef` | 0, 0.5 | weight of the exploration bonus and of the value error in the loss |
| `max_grad_norm` | 0.5 | gradients are clipped to this length before each step |
| `norm_obs`, `norm_reward` | True | running scaling of inputs and rewards |

The loss PPO minimises: `-(clipped policy objective) + vf_coef x (value error)^2 - ent_coef x entropy`.

## Rules for the networks

- Do not repeat any of these keys in your network's `CONFIG`. `train.py` stops with an error if you do.
- `--run-name` defaults to `base`, which now means config v1. Give your runs explicit names.
- Your network's own settings (layers, frames, sensors) stay in your network file.
