# STANCE Neural Network Submission (Group 42)

These are the two networks we trained with PPO to control a powered prosthetic ankle during weight acceptance on randomized compliant ground.
Network A is an MLP that reads the last 10 sensor frames, and Network B is an LSTM that builds its own memory of the stance.

This folder holds:
- `Gym/`: the STANCE environment (`stance_env/`) and the MuJoCo leg model (`model/leg.xml`)
- `Network_Agent/networkA_mlp/`, `Network_Agent/networkB_lstm/`: the two network definitions
- `Network_Agent/agent/`, `Network_Agent/common/`: the residual action, environment setup, policy adapters and evaluation used by the demo
- `Network_Agent/submission_network/`: our trained weights for both networks
- `Network_Agent/demo.py`: the demo
- `environment.yml`: the conda environment (CPU-only PyTorch)

## Setup and running

```bash
conda env create -f environment.yml
conda activate stance
cd Network_Agent
python demo.py
```

## What the demo prints

For each network, nothing is trained:
- Inputs and preprocessing: observation shape, frames, sensors and normalisation statistics
- Data flow: the layers and the parameter count of each part
- Outputs: what the three action values mean and their physical ranges
- One real observation: one forward pass with the action and value estimate
- Weights: shape and statistics per layer, with histograms saved to `demo_output/`
- Learning curve: survival and toe peak against training steps, saved to `demo_output/`
- Evaluation: the network next to the best fixed action on the same seeds
- Forward and backward pass: a short rollout, the PPO loss, its gradients, and 10 Adam steps on a copy

It ends with a side-by-side table of both networks and the fixed action.

## Troubleshooting

- If ROS is sourced in your shell, its `PYTHONPATH` can shadow the conda packages: run `env -u PYTHONPATH python demo.py`.
- If the environment is not found, set `STANCE_GYM_PATH` to the `Gym/` folder of this package.
