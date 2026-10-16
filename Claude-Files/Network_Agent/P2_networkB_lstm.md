# P2: Network B, recurrent (LSTM) network

**Your question:** does the network need its own memory of the stance? The ground's stiffness, yield
point and dent history never show up in a single sensor frame. You built the ground, so you know best
what is hidden. A recurrent network keeps a state that can pile up evidence across the whole 100-step
stance, so it could in principle work out how the ground is behaving. Your job is to find out whether
that helps in practice, compared with Network A's fixed window of 10 frames.

## Your files

| File | What it does | Edit it? |
|---|---|---|
| `networkB_lstm/network.py` | Defines `FrameEncoder` (the layer before the LSTM), `CONFIG` (your settings), `wrappers()` (which inputs the network sees) and `policy_kwargs()` (how the recurrent actor and critic are assembled). The docstring at the top explains the inputs, data flow, outputs and loss. | **Yes** |
| `networkB_lstm/test_network.py` | Tests that the default input is one frame, that memory changes the action, and that `reset()` clears memory. | Yes, add a test for anything you add |
| `agent/*`, `common/*` | Training, evaluation and shared code. | **No.** Ask P3 if you need something. |

The 11 sensor channels are listed in `P1_networkA_mlp.md` and in `common/env_setup.py` (`SENSOR_NAMES`).

## Your settings (`CONFIG` in `networkB_lstm/network.py`)

| Key | Default | Meaning |
|---|---|---|
| `frames` | 1 | newest frames fed at every step, 1 to 10 (1 means all memory is the LSTM's) |
| `drop_sensors` | `[]` | sensor indices removed from every frame |
| `frame_encoder` | 64 | width of the layer before the LSTM; 0 feeds the raw observation straight in |
| `lstm_hidden` | 64 | LSTM state size |
| `lstm_layers` | 1 | stacked LSTM layers |
| `head` | `[64]` | widths of the layers after the LSTM |
| `log_std_init` | `0.0` | starting action noise; the standard deviation is exp(this) |

Training settings belong to the agent, which P3 owns. Do not tune them before 22 Oct; use the defaults so
your results stay comparable.

## Run it

Run from the `Network/` folder.

```bash
python -m pytest networkB_lstm -q                             # seconds
python -m agent.train --network lstm --summary                # layers and parameter counts
python -m agent.train --network lstm --smoke                  # 1-2 minutes, whole pipeline
python -m agent.train --network lstm --run-name time --set total_steps=20000   # time your budget
python -m agent.train --network lstm --run-name base          # 300k steps, slow (see below)
python -m agent.train --network lstm --run-name f3 --set frames=3
python -m agent.train --network lstm --run-name h128 --set lstm_hidden=128
python -m agent.train --network lstm --run-name noenc --set frame_encoder=0
python -m agent.train --eval-only runs/networkB_lstm/base_s0 --test-split
```

**Recurrent training is about 4-5 times slower per step than Network A.** Time the 20k-step run first and
plan around it. Start long runs overnight, and start your final runs on 24 Oct as soon as P3 freezes the config.

Each run saves to `runs/networkB_lstm/<name>_s<seed>/` and appends a row to `results/networkB_lstm.csv`.

## What you have to do

- [ ] **16-17 Oct:** tests pass, `--summary` and the smoke run work, and you have timed 20k steps.
- [ ] **17 Oct:** baseline (`--run-name base`).
- [ ] **17-22 Oct, memory experiments.** One change per run, 300k steps, seed 0:
  - [ ] **Memory ablation (your main result):** `frames` = 1 (LSTM only), 3, and 10 (LSTM plus the full window).
        Does the LSTM add anything on top of a window?
  - [ ] **LSTM size:** `lstm_hidden` = 32, 64 and 128.
  - [ ] **Encoder:** `frame_encoder=0` against 64.
  - [ ] **Depth:** `lstm_layers=2`.
  - [ ] **Starting noise:** `log_std_init=-1.0`.
- [ ] **22 Oct:** post your first results row in the group chat.
- [ ] **24-26 Oct, final runs.** Best settings plus P3's frozen agent config. 1M steps, three seeds, with the
      held-out ground (add your best settings after `--set`):
      `python -m agent.train --network lstm --run-name final --seed 0 --test-split --set total_steps=1000000 ...`
      then `--seed 1` and `--seed 2`.
- [ ] **26 Oct:** run `python -m common.compare` and `python -m common.compare --split test`; draft the comparison.
- [ ] **27-28 Oct:** write your Report 2 sections (below).

Do the memory ablation first; your section is built on it.

## What you can change

Anything in `networkB_lstm/`: layers, encoder shape, head sizes, number of frames. sb3-contrib provides only an
LSTM. A GRU would need your own recurrent policy; only try it if the LSTM results are in and you have spare time.
Keep `ALGO`, `POLICY_NAME` and the other contract names (see `00_WHO_OWN_WHAT.md`) so `agent/train.py` can still use it.

**Do not change:** the evaluation seeds or metrics, `common/`, `agent/`, or the environment.

## What you write in Report 2

1. **Network B section**, following the lecture's checklist:
   - Inputs and preprocessing: one frame per step, and the normalisation.
   - Data flow: encoder, LSTM, heads (a table or figure from `--summary`), and why memory matters here.
   - Outputs: three means with a learned spread, and one value.
   - Losses and gradients: the same PPO loss as Network A, with gradients through time inside each stored
     sequence; written up with P3.
   - Parameter count, learning curve, result row on train and test, mean over 3 seeds.
   - Memory ablation table (`frames` 1, 3 and 10, plus LSTM size) and the wall-clock cost.
2. **Comparison and discussion** for the whole report:
   - The final-rows table from `python -m common.compare`.
   - Which network wins on survival and toe force against the best fixed action.
   - Whether memory helped on the held-out ground (`--split test`), where inferring the ground should matter most.

## Hints

- The held-out ground is your strongest argument. If the LSTM matches Network A on the training ground but
  holds up better on the test ground, that is a result even if it does not win overall.
- If the LSTM loses, that is also a result: it means a 50 ms window is enough for this 0.5 s stance. Say so.
- A bigger LSTM with this little data trains slowly and is not automatically better.

**Done when:** `results/networkB_lstm.csv` has the baseline, the memory runs and three final-seed rows on train
and test, and your Report 2 sections and the comparison are written.
