# Neural Network Architecture

## Network Structure
The architecture utilizes two branches. Recent history tells the agent what the ground is, while the current reading tells it what the leg is doing right now.

### Convolutional Branch
Treats the 10x11 history as 11 sensor signals sampled at 10 points in time. Two 1D convolution layers scan along the time axis to pick out how each signal changes over the window. This captures cues such as how fast the ground force rises, which we expect to differ between firm and soft ground.

### MLP Branch
A small fully connected network over the current step alone, so the present state is available directly rather than having to be recovered from history. 

### Shared Layers & Heads
The two outputs are joined and passed through two shared 256-unit layers (tanh activation) into two heads:
* **Actor:** Produces the three action values as the means of three independent Gaussians. Each mean is squashed and rescaled to its own physical range, rather than being used as a raw, unbounded output.
* **Critic:** Estimates how good the current state is; it is used only to guide learning and is discarded at deployment.

## Ablation Architecture
A plain MLP over the flattened history is implemented first as a reference architecture. Both the CNN and MLP versions are evaluated under the full history-length ablation (10, 3, and 1 stacked steps), so the architecture comparison and the history-length comparison are reported together rather than as two separate sets of experiments.
