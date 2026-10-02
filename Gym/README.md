# Gym

The simulation environment the STANCE agent trains in, built on the Gym/Gymnasium interface.

The environment models a powered ankle walking on compliant ground (for example concrete, wet grass,
gravel, and sand). Each surface compresses, damps, and grips differently, and some stay deformed after
contact. The focus is the weight-acceptance window, the 100–200 ms after heel strike.

This folder holds:
- The environment definition (observation space, action space, step/reset)
- The ground contact and terrain models
- The reward function, which penalises both over-stiff impacts and leg collapse
