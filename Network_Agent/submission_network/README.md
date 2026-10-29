# STANCE Neural Network Submission (Group 42)

These are the two networks we trained with PPO to control a powered prosthetic ankle during weight acceptance on randomized compliant ground.
Network A is an MLP that reads the last 10 sensor frames, and Network B is an LSTM that builds its own memory of the stance.

This folder holds:
- `Gym/`: the STANCE environment (`stance_env/`) and the MuJoCo leg model (`model/leg.xml`) from our previous submission
- `Network_Agent/networkA_mlp/`, `Network_Agent/networkB_lstm/`: the two network definitions
- `Network_Agent/agent/`, `Network_Agent/common/`: the residual action, environment setup, policy adapters and evaluation
- `Network_Agent/trained_networks/`: our trained networks (weights, normalisation statistics, config and learning curve)
- `Network_Agent/demo.py`
- `environment.yml`

## Setup and running

```bash
conda env create -f environment.yml
conda activate stance
cd Network_Agent
python demo.py
#or 
python3 demo.py
```

## If you want to train a fresh network:
```bash
cd Network_Agent
conda env create -f environment.yml # If you havent already
conda activate stance
python -m agent.train --network mlp
python -m agent.train --network lstm
```

## What the demo shows:

- Inputs and preprocessing: observation shape, frames, sensors and normalisation statistics
- Data flow: the layers and the parameter count of each part
- Outputs: what the three action values mean and their physical ranges
- One real observation: one forward pass with the action and value estimate
- Weights: shape and statistics per layer, with histograms saved to `demo_output/`
- Learning curve: survival and toe peak against training steps, saved to `demo_output/`
- Evaluation: the network next to the best fixed action on the same seeds
- Forward and backward pass: a short rollout, the PPO loss, its gradients, and 10 Adam steps on a copy

It ends with a side-by-side table of both networks and the fixed action.

## Notes:

- If ROS is sourced in your shell: run `env -u PYTHONPATH python demo.py`.
- If the environment is not found: set `STANCE_GYM_PATH` to the `Gym/` folder of this package.

## AI Declreation:

We used Claude (Anthropic) to draft a template code for us such that if we worked on it induvidually, there were keys that made sure our tests remained common across files. Additionally we used it to help debug our code and to flag discrepencies during merges.