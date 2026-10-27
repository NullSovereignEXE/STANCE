# STANCE learning agent: demo

`demo_agent.py` shows the learning agent end to end without the long training commands. It
either loads a saved run or trains a network from scratch, prints the network's layers and
parameter count, then plays evaluation episodes with the trained policy and with the best
fixed action (-20 deg, K 500, B 10) on the same seeds and prints the two side by side.

It reuses the project code unchanged: `agent/train.py` (network loading, config, wrappers,
model building), `agent/curve.py`, `common/env_setup.py`, `common/evaluate.py` and
`common/policies.py`. It writes nothing to `results/` or `runs/`; a model trained by the demo
is saved to a temporary folder whose path is printed.

## Setup

1. Install the packages (Python 3.10):

   ```bash
   pip install -r requirements.txt
   ```

   The environment also needs MuJoCo; install it from the environment's own files
   (`STANCE/Gym/environment.yml`, which pins `mujoco` and `gymnasium`).

2. Point the code at the frozen environment, the folder that contains `stance_env/`:

   ```bash
   export STANCE_GYM_PATH=/path/to/STANCE/Gym
   ```

   Without it, `common/env_setup.py` takes the first `Gym/` folder above this one, which may
   be a different copy of the environment.

3. Run every command from the `Network_Agent` folder. If `PYTHONPATH` is set by another
   tool (for example ROS), run with `env -u PYTHONPATH python demo_agent.py ...`.

## Commands

```bash
# Quick check that the whole pipeline works (tiny training, about 30 s)
python demo_agent.py --train --smoke --episodes 3

# Train the MLP from scratch with the agent's settings (300k steps, about 6.4 minutes)
python demo_agent.py --train

# Same for the LSTM network
python demo_agent.py --network lstm --train

# Evaluate a saved run (a folder with model.zip, vecnormalize.pkl and config.json)
python demo_agent.py --model <run folder> --episodes 200
```

| Argument | Meaning |
|---|---|
| `--network {mlp,lstm}` | network to train with `--train` (default `mlp`); `--model` reads it from `config.json` |
| `--model DIR` | load and evaluate a saved run folder |
| `--train` | train from scratch, then evaluate |
| `--smoke` | with `--train`: tiny training run, only to check the pipeline |
| `--episodes N` | evaluation episodes on seeds 1000 to 1000+N-1 (default 20) |

With neither `--model` nor `--train`, the demo prints the options and exits.

**Final model:** TODO: `runs/networkA_mlp/<final run>` (folder to submit with the demo)

## Output

1. During `--train`: a `curve:` line every `curve_every` steps, with survival on the
   learning-curve seeds (2000 and up), so you can watch training progress.
2. The network's layers, the parameters in each part, and the total parameter count.
3. A table with one column for the trained network and one for the fixed action, both on the
   same seeds and with deterministic actions:

| Row | Meaning |
|---|---|
| survival (%) | share of episodes that reach the 100-step limit without tipping over or collapsing |
| toe peak (BW) | largest toe normal force in the episode, in body weights, averaged over episodes (lower is gentler) |
| mean angle (deg) | target ankle angle the policy applied, averaged over steps and episodes |
| mean K (N.m/rad) | ankle stiffness applied, averaged the same way |
| mean B (N.m.s/rad) | ankle damping applied, averaged the same way |
| mean return | total reward per episode: +1 per step, minus the toe-force, smoothness and fall penalties |

A network beats the fixed action if it matches its survival with a lower toe peak, or
survives more often at a similar toe peak. Use `--episodes 200` for numbers that can be
compared with the report; 3 or 20 episodes only show that it works.

## Results

TODO: final table (200 episodes, seeds 1000-1199) for the submitted model against the fixed action.

| | final model | fixed action |
|---|---|---|
| survival (%) | TODO | 92.5 |
| toe peak (BW) | TODO | 1.74 |
| mean return | TODO | 87.90 |
