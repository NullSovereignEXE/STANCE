# Gym

The simulation environment the STANCE agent trains in, built on the Gym/Gymnasium interface.

The environment models a powered ankle walking on compliant ground (for example concrete, wet grass,
gravel, and sand). Each surface compresses, damps, and grips differently, and some stay deformed after
contact. The focus is the weight-acceptance window, the 100–200 ms after heel strike.

This folder holds:
- The environment definition (observation space, action space, step/reset)
- The ground contact and terrain models
- The reward function, which penalises both over-stiff impacts and leg collapse

Run from this `Gym/` folder:

```bash
conda env create -f environment.yml # Note that we are installing the CPU-only version of Torch to speed up the download. In actual training we will use the GPU.
conda activate stance
python demo.py              # Fixed impedance policy, new random episode each run
python demo.py --random     # Random actions instead
python demo.py --seed 7     # Replay a specific episode (seed is printed each run)
python -m pytest            # Run the test suite
```

`demo.py` opens the MuJoCo viewer and plays the episode at 10x slow motion (`--slowmo 1` for real time). Close the window to exit. If ROS is sourced in your shell, run pytest as `env -u PYTHONPATH python -m pytest`.

## Problem Model (From Project Proposal)

**The mechanism (5 degrees of freedom, 1 motor):** 
A 0.26 m foot is free to translate and rotate in the plane, held up only by ground contact, with contact points at heel and toe. A powered ankle hinge (-20° to +20°, 150 N·m maximum) is the only actuated joint. Above it, a sliding joint with a passive spring and damper represents shaft and socket flexibility (non-actuated). A 70 kg point mass on a 0.85 m leg represents the user.

**The ground:** 
MuJoCo's built-in contact solver is disabled; we supply our own contact force (F) at each point instead, based on sinking depth d and sinking speed d_dot:
F = k(d)d + c(d_dot); F >= 0

Stiffness grows with depth (k(d)) to mimic how sand and mud firm up when you sink into it:
k(d) = k_0(1 + \alpha \cdot d/d_{ref})
where k_0 is the ground's baseline (undeformed) stiffness, \alpha controls how fast it stiffens with depth, and d_{ref} is a reference depth that sets the scale of that stiffening. Past a preset yield force, the surface stays permanently dented.

The heel and toe portions of the foot model draw their properties independently, so the foot can straddle two surfaces.

**Grip Dynamics:** 
Grip is modeled with velocity-regularized Coulomb friction, using a tanh instead of a hard sign function to avoid solver-breaking discontinuity at zero velocity:
F_t = \mu F_n \tanh(v_t / v_{\epsilon})

**Physics Verification:** 
Because the surface is a flat plane contacted at two points, penetration depth is a simple subtraction and each point's dent depth is stored as a state variable, so no mesh or finite-element modelling is required. Parameter ranges come from published terrain and prosthetic foot compliance data, and the model is verified against closed-form results: with the motor disabled the foot must settle at mg/k, oscillate at \sqrt{k/m}, and stay dented after yielding.

**The episode:** 
Starts at heel strike with randomized forward speed (0.8 - 1.4 m/s), downward speed (0.10 - 0.50 m/s), ankle angle and leg tilt. Each episode lasts 100 steps covering heel strike, weight acceptance and mid-stance. Episode ends if the leg tips more than 15° backward or 35° forward from vertical, or the body drops below 70% of standing height.

**Randomization:** 
Ground properties are randomized per episode. A separate range is reserved for testing only, so the agent cannot succeed by memorizing surfaces.

## AI Declaration
We used Claude Opus 5 (Anthropic) to evaluate ideas, explain concepts/bugs and help resolve inter-file conflicts. We are completely responsible for the content and quality of the submitted work.
