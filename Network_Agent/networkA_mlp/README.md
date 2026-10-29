# Neural Network Architecture — Network A (MLP)

## Network Structure

Network A uses a **feed-forward multilayer perceptron (MLP)** to control the prosthetic ankle from a stacked history of sensor observations. The default input contains **10 frames × 11 sensors = 110 features**, ordered oldest frame first. Sensors describe ankle motion and torque, vertical ground reaction force, pylon compression, shank motion, vertical acceleration, and previous control actions.

Observations are selected by the input wrapper and normalized using **VecNormalize** statistics collected during training.

### MLP Feature Extractor

Each of the Actor and Critic has its **own** feature extractor with two fully connected 256-unit layers and Tanh activations:

```text
110 inputs → Linear(110, 256) → Tanh → Linear(256, 256) → Tanh
```

### Actor and Critic Heads

Each feature extractor feeds a separate 64-unit Tanh head. Stable-Baselines3 adds the output layers:

```text
                          110 sensor inputs
                         /                 \
              Actor feature MLP       Critic feature MLP
                110 → 256 → 256         110 → 256 → 256
                       |                       |
                   256 → 64                256 → 64
                       |                       |
                 3 action means           1 state value
                   (θd, K, B)                 V(s)
```

- **Actor:** Produces three means for a Gaussian action distribution corresponding to target ankle angle (θd), stiffness (K), and damping (B). The policy also learns three log standard deviations for exploration. Deterministic evaluation uses the means. The standard SB3 Gaussian mean head is not itself Tanh-squashed; action bounds are handled when interacting with the environment.
- **Critic:** Produces one scalar state-value estimate, V(s), used to train PPO. The Critic is not needed to select actions during deployment.

The complete policy contains **221,575 trainable parameters**, including separate Actor and Critic feature extractors and the action-distribution parameters.

### Training and Residual Action Control

Network A uses **PPO from Stable-Baselines3**, with 300,000 environment timesteps per run and three training seeds (0, 1, 2). The shared config v1 uses eight parallel environments and normalizes observations and rewards. The PPO agent and shared evaluator are maintained separately from this network definition.

Rather than commanding the full action directly, the policy supplies a bounded adjustment around a fixed anchor:

```text
a_env = clip(a_anchor + 0.3 × a_network, -1, 1)
```

The anchor corresponds to **θd = −14°, K = 250.6 N·m/rad, B = 6.38 N·m·s/rad**. The adjustment magnitude of 0.3 normalized units is **15% of the full [−1, 1] range**. The environment maps the resulting normalized action into physical units.

### Input Selection and Ablation Support

The input wrapper can keep the newest 1, 3, or 10 frames and remove selected sensor channels. The submitted default model uses **10 frames and all 11 sensors**.

| Input configuration | Features |
| --- | ---: |
| 10 frames, all sensors (submitted default) | 110 |
| 3 frames, all sensors | 33 |
| 1 frame, all sensors | 11 |
| 10 frames, excluding previous actions (indices 8, 9, 10) | 80 |
| 10 frames, excluding vertical acceleration (index 7) | 100 |
| 10 frames, excluding ankle torque (index 2) | 100 |
| 10 frames, excluding vertical ground reaction force (index 3) | 100 |

These are **supported input configurations**, not a claim that all of them were evaluated in the final report. Network A's submitted architecture is an MLP; it has **no convolutional branch**, and this README does not claim a CNN-versus-MLP ablation.

### Evaluation Summary

The shared evaluator uses **200 deterministic episodes per split (seeds 1000–1199)**. For the final config-v1 runs, the consolidated three-seed means reported by P3 are:

| Metric | Training ground | Held-out test ground |
| --- | ---: | ---: |
| Survival | 88.3% | 48.0% |
| Mean return | 89.01 | 78.26 |
| Return difference vs best fixed controller | +1.11 | −0.31 |
| Peak toe force | 1.720 BW | Not compared (ground-yield cap) |

The best fixed controller is **(θd = −20°, K = 500 N·m/rad, B = 10 N·m·s/rad)**. Network A's overall return is approximately tied with this strong fixed baseline. Its lower accumulated impact penalty is partly offset by smoothness, fall, and episode-length costs. It does not show a clear advantage on held-out ground.

For authoritative results and qualifications, see `REPORT2_DATA.md` and the shared `results/` tables.
