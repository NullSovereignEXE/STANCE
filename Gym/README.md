# Gym & Environment Submission

The simulation environment the STANCE agent trains in, built on the Gym/Gymnasium interface and uses MuJoCo for visualization.

The environment models a powered ankle walking on compliant ground (for example concrete, wet grass,
gravel, and sand). Each surface compresses, damps, and grips differently, and some stay deformed after
contact. The focus is the weight-acceptance window, the 100–200 ms after heel strike.

This folder holds:
- The environment definition (observation space, action space, step/reset)
- The ground contact and terrain models
- The reward function, which penalises both over-stiff impacts and leg collapse

## Requirements

Tested on Ubuntu (dual boot) with Python 3.10 and a display available. Windows and macOS are not supported.

## Running

Run from this `Gym/` folder:

```bash
conda env create -f environment.yml # Note that we are installing the CPU-only version of Torch to speed up the download. In actual training we will use the GPU.
conda activate stance
python demo.py              # Fixed impedance policy, new random episode each run
python demo.py --random     # Random actions instead
python demo.py --seed 7     # Replay a specific episode (seed is printed each run)
python demo.py --split test # Run on the held-out test ground instead of the training ground
python demo.py --slowmo 1   # Real-time playback (default is 10x slow motion)
python -m pytest            # Run the test suite
```

`demo.py` opens the MuJoCo viewer and plays the episode at 10x slow motion. Close the window to exit. If ROS is sourced in your shell, run pytest as `env -u PYTHONPATH python -m pytest`.

Expected console output: a line with the episode seed, the heel/toe base stiffness `k0`, friction `mu` and yield force `f_yield`, followed after the episode by a summary of the form
`completed after N steps | return R | tilt T deg | peak GRF G BW` (or `fell after ...` if the leg tipped or collapsed).

# Extra info (Based on our Proposal and Changes)

## Action and observation layout

**Action (3 values in [-1, 1]):** target ankle angle $\theta_d$ (linear map to -20° to +20°), ankle stiffness $K$ (log map to 5–500 N·m/rad) and ankle damping $B$ (log map to 0.5–10 N·m·s/rad). The ankle torque is $\tau = K(\theta_d - \theta) - B\dot{\theta}$, clipped to ±150 N·m.

**Observation (110 values):** the last 10 sensor vectors, flattened. Each vector has 11 channels:

| Index | Channel |
|---|---|
| 0 | Ankle angle |
| 1 | Ankle rate |
| 2 | Ankle torque (motor command) |
| 3 | Vertical ground reaction force |
| 4 | Pylon compression |
| 5 | Shank tilt |
| 6 | Shank tilt rate |
| 7 | Vertical acceleration |
| 8–10 | Previous action |

Ground parameters are never exposed to the agent.

## Problem Model (From Project Proposal)

**The mechanism (5 degrees of freedom, 1 motor):**
A 0.26 m foot is free to translate and rotate in the plane, held up only by ground contact, with contact points at heel and toe. A powered ankle hinge (-20° to +20°, 150 N·m maximum) is the only actuated joint. Above it, a sliding joint with a passive spring and damper represents shaft and socket flexibility (non-actuated). A 70 kg point mass represents the user. The proposal specified a 0.85 m leg; the built model has a total mass of 72.41 kg and a standing height of 0.90 m, since the foot, shank and pylon add mass and height.

**The ground:**
MuJoCo's built-in contact solver is disabled; we supply our own contact force $F$ at each point instead, based on sinking depth $d$ and sinking speed $\dot{d}$:

$$F = k(d)\,d + c\,\dot{d}, \qquad F \ge 0$$

Stiffness grows with depth to mimic how sand and mud firm up when you sink into it:

$$k(d) = k_0\left(1 + \alpha \frac{d}{d_{ref}}\right)$$

where $k_0$ is the ground's baseline (undeformed) stiffness, $\alpha$ controls how fast it stiffens with depth, and $d_{ref}$ is a reference depth that sets the scale of that stiffening. Past a preset yield force, the surface stays permanently dented.

The heel and toe share the same ground properties within an episode (stiffness, damping, stiffening, yield force and friction are drawn once per episode and applied to both contact points), while each point keeps its own dent depth. Letting heel and toe draw independent properties, so the foot can straddle two surfaces, is left as future work.

**Grip Dynamics:**
Grip is modeled with velocity-regularized Coulomb friction, using a tanh instead of a hard sign function to avoid solver-breaking discontinuity at zero velocity:

$$F_t = -\mu F_n \tanh\left(\frac{v_t}{v_{\epsilon}}\right)$$

**Physics Verification:**
Because the surface is a flat plane contacted at two points, penetration depth is a simple subtraction and each point's dent depth is stored as a state variable, so no mesh or finite-element modelling is required. Parameter ranges come from published terrain and prosthetic foot compliance data, and the model is verified against closed-form results: with the motor disabled the foot must settle at $mg/k$ (exact for $\alpha = 0$), oscillate at $\sqrt{k/m}$, and stay dented after yielding.

**The episode:**
Starts at heel strike with randomized forward speed (0.8 - 1.2 m/s), downward speed (0.10 - 0.50 m/s), ankle angle and leg tilt. Each episode lasts 100 steps covering heel strike, weight acceptance and mid-stance. Episode ends if the leg tips more than 15° backward or 35° forward from vertical, or the body drops below 70% of standing height.

**Randomization:**
Ground properties are randomized per episode. A separate range is reserved for testing only, so the agent cannot succeed by memorizing surfaces. The test ranges lie outside the training ranges for $k_0$, $\alpha$, yield force and friction; the damping ratio is shared between the two splits.

## Assumptions:
(i) Joint damping: leg.xml gives foot_x and foot_pitch a damping of 0.1, foot_z 1.0, and the ankle 0.1. That’s why the item says “0.1–1.0”. I also mentioned the ankle. Take it out if you only want the foot joints.
(ii) Approximate yield depth: ground.py computes f_yield / k using the stiffness at the current depth. It doesn’t solve k(d)·d = f_yield, so the dent is exact only when α = 0.
(iii) Test grounds yield easily: body weight is 72.4 kg × 9.81 ≈ 710 N, which matches your figure.
(iv) Test shifts several parameters at once: the test split changes k₀, α, f_yield and μ together. Only ζ keeps the same range as training, so the item says that.

## AI Declaration
We used Claude Opus 5 (Anthropic) to evaluate ideas, explain concepts/bugs and help resolve inter-file conflicts and help with our report's LaTeX syntax. 
We are completely responsible for the content and quality of the submitted work.
