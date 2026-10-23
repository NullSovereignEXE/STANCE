# STANCE networks: demo

`demo_network.py` shows the two networks without training anything. It loads each network's
saved weights from `submission/`, prints how the network is built and what its weights look
like, runs it once on a real observation, and evaluates it next to the best fixed action
(-20 deg, K 500, B 10) on the same seeds.

| Folder | Network | Owner |
|---|---|---|
| `submission/networkA_mlp/` | Network A: feed-forward MLP on the last 10 sensor frames | P1 |
| `submission/networkB_lstm/` | Network B: LSTM that builds its own memory of the stance | P2 |

> **PLACEHOLDERS.** The files in `submission/` are not the final networks. They were copied
> from existing config v1 runs so the demo can be built and tested now:
> `submission/networkA_mlp/` from `runs/networkA_mlp/integ_s0` (300k steps, seed 0) and
> `submission/networkB_lstm/` from `runs/networkB_lstm/base_s0` (300k steps, seed 0).
> Each owner replaces them with their final run folder (see below).

The demo uses the project code unchanged: `common/` (environment, evaluation, policy
adapters), `agent/actions.py` (the residual action) and the two network files. It does not
import `agent/train.py` and writes only to `demo_output/`.

## Setup

1. Install the packages (Python 3.10):

   ```bash
   pip install -r requirements.txt
   pip install matplotlib        # for the plots; not in requirements.txt
   ```

   The environment also needs MuJoCo; install it from the environment's own files
   (`STANCE/Gym/environment.yml`, which pins `mujoco` and `gymnasium`).

2. Point the code at the frozen environment, the folder that contains `stance_env/`:

   ```bash
   export STANCE_GYM_PATH=/path/to/STANCE/Gym
   ```

   Without it, `common/env_setup.py` takes the first `Gym/` folder above this one.

3. Run every command from the `Network_Agent` folder. If `PYTHONPATH` is set by another
   tool (for example ROS), run with `env -u PYTHONPATH python demo_network.py ...`.

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
| `--submission DIR` | folder holding the run folders (default `submission`) |

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

With both networks present, the demo ends with a side-by-side table of the two networks and
the fixed action. On the test ground the toe peak is the same for every policy (it is set at
touchdown, before the policy can act), so compare survival there, not toe peak. Use
`--episodes 200` for numbers that can be compared with the report; 5 or 20 episodes only
show that it works.

## Replacing a placeholder with the final network

Each owner copies four files from their final run folder (written by
`python -m agent.train ... --run-name <name>`) into their submission folder:

```bash
# P1, Network A
cp runs/networkA_mlp/<final run>/{model.zip,vecnormalize.pkl,config.json,curve.csv} submission/networkA_mlp/

# P2, Network B
cp runs/networkB_lstm/<final run>/{model.zip,vecnormalize.pkl,config.json,curve.csv} submission/networkB_lstm/
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
