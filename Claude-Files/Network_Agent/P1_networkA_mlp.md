# P1: Network A, feed-forward network and input design

**Your question:** which sensors and how much history does the network really need? You built the leg,
so you know what each sensor measures. Your job is to find which of them carry information the
network uses, and to build the best feed-forward network on that input. This network is also the
baseline the recurrent network is compared against.

## Your files

| File | What it does | Edit it? |
|---|---|---|
| `networkA_mlp/network.py` | Defines `HistoryMLP` (the layers), `CONFIG` (your settings), `wrappers()` (which inputs the network sees) and `policy_kwargs()` (how the actor and critic are assembled). The docstring at the top explains the inputs, data flow, outputs and loss. | **Yes** |
| `networkA_mlp/test_network.py` | Tests the encoder's shape and gradients, the input cut, and that the model gives valid actions. | Yes, add a test for anything you add |
| `agent/*`, `common/*` | Training, evaluation and shared code. | **No.** Ask P3 if you need something. |

The 11 sensor channels, newest frame last, repeated for 10 frames (110 numbers in total):

| Index | Sensor | Index | Sensor |
|---|---|---|---|
| 0 | ankle angle | 6 | shank tilt rate |
| 1 | ankle rate | 7 | vertical acceleration |
| 2 | ankle torque | 8 | previous target angle |
| 3 | vertical ground force | 9 | previous stiffness K |
| 4 | pylon compression | 10 | previous damping B |
| 5 | shank tilt | | |

## Your settings (`CONFIG` in `networkA_mlp/network.py`)

| Key | Default | Meaning |
|---|---|---|
| `frames` | 10 | newest frames kept, 1 to 10 |
| `drop_sensors` | `[]` | sensor indices removed from every frame |
| `hidden` | `[256, 256]` | widths of the encoder layers |
| `activation` | `"tanh"` | `"tanh"`, `"relu"` or `"elu"` |
| `layer_norm` | `False` | LayerNorm after each encoder layer |
| `head` | `[64]` | widths of the layers after the encoder |
| `log_std_init` | `0.0` | starting action noise; the standard deviation is exp(this) |

Training settings (learning rate, discount, loss weights) belong to the agent, which P3 owns. Do not
tune them yourself before 22 Oct; use the defaults so your results stay comparable.

## Run it

Run from the `Network/` folder.

```bash
python -m pytest networkA_mlp -q                              # seconds
python -m agent.train --network mlp --summary                 # layers and parameter counts
python -m agent.train --network mlp --smoke                   # 1-2 minutes, whole pipeline
python -m agent.train --network mlp --run-name base           # 300k steps, a few minutes
python -m agent.train --network mlp --run-name f1 --set frames=1
python -m agent.train --network mlp --run-name noprev --set "drop_sensors=[8,9,10]"
python -m agent.train --network mlp --run-name small --set "hidden=[64,64]"
python -m agent.train --eval-only runs/networkA_mlp/base_s0 --test-split
```

Each run saves to `runs/networkA_mlp/<name>_s<seed>/` (model, `config.json`, `curve.csv`, per-episode
CSV) and appends a row to `results/networkA_mlp.csv`.

## What you have to do

- [ ] **16-17 Oct:** tests pass, `--summary` and the smoke run work.
- [ ] **17 Oct:** baseline (`--run-name base`). Expect survival near the earlier 26%.
- [ ] **17-21 Oct, input design (your main result).** One change per run, 300k steps, seed 0:
  - [ ] **History:** `frames` = 1, 3 and 10. Does the network need the history at all?
  - [ ] **Previous actions:** `drop_sensors=[8,9,10]`. These feed the last action back in as memory.
  - [ ] **Vertical acceleration:** `drop_sensors=[7]`. An acceleration reading, which tends to be noisy.
  - [ ] **Torque:** `drop_sensors=[2]`. The torque is computed from the angle, the angle rate and the last action by the impedance law, so it may add nothing new.
  - [ ] **Ground force:** `drop_sensors=[3]`. Is the force reading something the network relies on?
  - Record the table: which dropped sensor made survival or toe force worse?
- [ ] **21-24 Oct, network design.** With the best inputs: `hidden=[64,64]` against `[256,256]`,
      `layer_norm=True`, `activation=relu`, `log_std_init=-1.0`.
- [ ] **22 Oct:** post your first results row in the group chat.
- [ ] **24-26 Oct, final runs.** Use your best settings and P3's frozen agent config. 1M steps, three seeds,
      with the held-out ground:
      `python -m agent.train --network mlp --run-name final --seed 0 --test-split --set total_steps=1000000 ...`
      (add your best settings after `--set`), then `--seed 1` and `--seed 2`.
- [ ] **27-28 Oct:** write your Report 2 sections (below).

You do not need every experiment. Do the history and the previous-actions runs first; they matter most.

## What you can change

Anything in `networkA_mlp/`: add layers, residual connections, a different head, or your own input
features (copy `SelectInputs` from `common/inputs.py` into your folder and extend it). If you go beyond what
`policy_kwargs` supports, subclass SB3's `ActorCriticPolicy` in `network.py`. Keep `ALGO`, `POLICY_NAME`
and the other contract names (see `00_WHO_OWN_WHAT.md`) so `agent/train.py` can still use it.

**Do not change:** the evaluation seeds or metrics, `common/`, `agent/`, or the environment.

## What you write in Report 2

1. **Shared setup** (short, for the whole report): what the network sees (11 sensors over 10 frames), what
   it outputs (target angle, K and B in [-1, 1], mapped by the environment), the reward in one sentence,
   and the evaluation protocol (copy the table from `00_WHO_OWN_WHAT.md`).
2. **Network A section**, following the lecture's checklist:
   - Inputs and preprocessing: which frames and sensors, and the normalisation.
   - Data flow: layers, widths, activations (a table or figure from `--summary`).
   - Outputs: three means with a learned spread, and one value.
   - Losses and gradients: written up with P3 (PPO loss, clipping, Adam).
   - Parameter count, learning curve (`curve.csv`), result row on train and test, mean over 3 seeds.
   - The input-design table, which is your ablation.

## Hints

- Watch `mean_theta_deg` in `curve.csv` as well as survival. The known failure is the angle drifting
  positive, so a run that keeps it near -20 degrees is working even if survival is slow to follow.
- One seed proves little. If two settings differ by less than about 5 points of survival, call it a tie
  until you have run a second seed.
- If nothing learns well, that is a finding. Say so and show the curve.

**Done when:** `results/networkA_mlp.csv` has the baseline, the input-design runs and three final-seed rows on
train and test, and your Report 2 sections are written.
