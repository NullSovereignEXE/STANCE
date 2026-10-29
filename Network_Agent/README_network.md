# STANCE networks: demo

## How to run

We trained two networks with PPO on our ankle environment: a feed-forward MLP (Network A) and an LSTM (Network B).
The demo loads their trained weights and compares each one with the best fixed action on the same episodes.

Folder layout: `Gym/stance_env/` and `Gym/model/` must sit next to `Network_Agent/` inside `STANCE/`. Run every command from `Network_Agent/`.

```bash
pip install -r requirements.txt                                      # Python 3.10, CPU-only torch
conda env create -f ../Gym/environment.yml && conda activate stance  # alternative
python demo_network.py    # in the submitted zip this file is named demo.py
```

What you will see, for each network:
- Inputs and preprocessing: the observation shape, the frames and sensors the network sees, and the saved normalisation statistics.
- Data flow: the layers as PyTorch prints them, and the parameter count of each part.
- Outputs: what the three action values mean, their physical ranges and the residual mapping.
- One real observation: one forward pass on the first step of an episode; for the LSTM, a check that its memory changes the action.
- Weights: shape, mean, std, min and max of each parameter tensor, with histograms saved to `demo_output/`.
- Learning curve: survival and toe peak against training steps, saved to `demo_output/`.
- Evaluation: survival, toe peak, mean angle, K, B and return, next to the fixed action on the same seeds.
- Forward and backward pass: a short rollout, the PPO loss, the gradient norm of every tensor, and Adam steps on a copy, with a check that the saved weights did not change.

With both networks, a side-by-side table ends the output.

If it fails:
- If ROS or another tool sets `PYTHONPATH`, run `env -u PYTHONPATH python demo_network.py`.
- If `Gym/` is somewhere else, run `STANCE_GYM_PATH=/path/to/Gym python demo_network.py`.

The trained weights are in `submission_network/networkA_mlp/` and `submission_network/networkB_lstm/`.

`demo_network.py` shows the two networks without training anything. It loads each network's
saved weights from `submission_network/`, prints how the network is built and what its weights look
like, runs it once on a real observation, and evaluates it next to the best fixed action
(-20 deg, K 500, B 10) on the same seeds.

| Folder | Network | Owner |
|---|---|---|
| `submission_network/networkA_mlp/` | Network A: feed-forward MLP on the last 10 sensor frames | P1 |
| `submission_network/networkB_lstm/` | Network B: LSTM that builds its own memory of the stance | P2 |

The files in `submission_network/` are the final seed-0 runs, copied unchanged from
`runs/networkA_mlp/final_s0` and `runs/networkB_lstm/final_s0`.

The demo uses the project code unchanged: `common/` (environment, evaluation, policy
adapters), `agent/actions.py` (the residual action) and the two network files. It does not
import `agent/train.py` and writes only to `demo_output/`.

## Commands

```bash
# Both networks, 20 episodes, plots into demo_output/
python demo_network.py

# One network, with enough episodes to compare with the report
python demo_network.py --networks mlp --episodes 200
python demo_network.py --networks lstm --episodes 200

# Held-out test ground, no plots
python demo_network.py --split test --no-plots
```

| Argument | Meaning |
|---|---|
| `--networks mlp lstm` | which networks to show (default: every usable folder under `--submission`) |
| `--episodes N` | evaluation episodes on seeds 1000 to 1000+N-1 (default 20) |
| `--split {train,test}` | ground to evaluate on (default `train`) |
| `--no-plots` | do not write `demo_output/*.png` |
| `--submission DIR` | folder holding the run folders (default `submission_network`) |

If no usable folder is found, the demo says where to put the files and exits.

## What each section shows

For each network, in this order:

1. **Inputs and preprocessing.** The observation shape after the network's own input
   selection, how many of the 10 frames and which of the 11 sensors it sees, and the
   normalisation statistics saved from training (VecNormalize count, and the mean and
   variance for each sensor in the newest frame).
2. **Data flow.** The PyTorch print of the policy (every layer, its width and activation)
   and the number of parameters in each part (actor and critic encoders, LSTMs, heads).
3. **Outputs.** What the three action values mean (target ankle angle, stiffness K,
   damping B), their physical ranges, and, with the residual action, the anchor, the
   scale and the range the network can actually reach.
4. **One real observation.** One forward pass on the first step of seed 1000: the
   network's output, the action after the residual mapping, the physical action and the
   critic's value estimate. For the LSTM, the same observation is fed five times in a row;
   a change in the action shows that the network uses its memory.
5. **Weights.** For each parameter tensor: shape, mean, standard deviation, minimum and
   maximum. Histograms of the weight matrices go to `demo_output/<mlp|lstm>_weights.png`.
6. **Learning curve.** Survival and toe peak force against training steps, from
   `curve.csv` (seeds 2000-2049, never used in the evaluation), saved to
   `demo_output/<mlp|lstm>_curve.png`.
7. **Evaluation.** The network, through its own evaluation adapter, and the best fixed
   action on the same seeds:

| Row | Meaning |
|---|---|
| survival (%) | share of episodes that reach the 100-step limit without tipping over or collapsing |
| toe peak (BW) | largest toe normal force in the episode, in body weights, averaged over episodes (lower is gentler) |
| mean angle (deg) | target ankle angle the policy applied, averaged over steps and episodes |
| mean K (N.m/rad) | ankle stiffness applied, averaged the same way |
| mean B (N.m.s/rad) | ankle damping applied, averaged the same way |
| mean return | total reward per episode: +1 per step, minus the toe-force, smoothness and fall penalties |

8. **Forward and backward pass.** One training step's computation, without changing the saved
   weights. It samples 128 stochastic steps from the saved policy (seed 0, train ground) and
   computes GAE advantages and returns. It then runs `evaluate_actions` in training mode and
   prints the PPO loss split into its policy, value and entropy terms, with every coefficient
   read from `config.json`. After `loss.backward()` it prints the gradient norm of every
   parameter tensor, the total norm against `max_grad_norm`, and a flag for any missing, zero
   or non-finite gradient. Finally it takes 10 Adam steps on a copy of the policy, prints the
   loss before and after, and confirms with a checksum that the original weights are unchanged.
   This section takes a second or two per network.

With both networks present, the demo ends with a side-by-side table of the two networks and
the fixed action. On the test ground the toe peak is the same for every policy (it is set at
touchdown, before the policy can act), so compare survival there, not toe peak. Use
`--episodes 200` for numbers that can be compared with the report; 5 or 20 episodes only
show that it works.

## Replacing a submitted network

Each owner copies four files from their final run folder (written by
`python -m agent.train ... --run-name <name>`) into their folder under `submission_network/`:

```bash
# P1, Network A
cp runs/networkA_mlp/<final run>/{model.zip,vecnormalize.pkl,config.json,curve.csv} submission_network/networkA_mlp/

# P2, Network B
cp runs/networkB_lstm/<final run>/{model.zip,vecnormalize.pkl,config.json,curve.csv} submission_network/networkB_lstm/
```

Then run `python demo_network.py --networks mlp` (or `lstm`) to check it loads. The demo
rebuilds the network from `config.json`, so the network file must still describe the same
layers as when the run was trained.

## Final results

TODO: replace after the final runs (200 episodes, seeds 1000-1199, train ground).

| | Network A (MLP) | Network B (LSTM) | fixed action |
|---|---|---|---|
| final run folder | TODO | TODO | |
| survival (%) | TODO | TODO | 92.5 |
| toe peak (BW) | TODO | TODO | 1.74 |
| mean return | TODO | TODO | 87.90 |

TODO: test ground (`--split test --episodes 200`), survival only.
