# Agent
This folder holds the agent and policy code, the training loop, and the evaluation scripts.

## Agent Behavior
In training, it tries combinations of target angle, stiffness, and damping across thousands of randomized surfaces, scored on falls and impact force. At deployment, the weights are frozen: nothing reveals the ground before contact, so it lands with the best setting on average, then reads the force response and converges on the stiffness that suited similar surfaces. It adapts within the step but does not learn during it.

## State Space (11 values per step)
* Ankle angle and angular velocity (encoder)
* Ankle torque (motor current)
* Vertical ground reaction force (pylon load cell)
* Pylon compression
* Shank tilt angle and tilt rate (IMU)
* Shank vertical acceleration
* Previous action (3)

The true ground parameters (stiffness, damping and friction) are never given to the agent, and it must infer them from the signals above.

## Action Space
The RL agent outputs a three-dimensional continuous action at each control step (200 Hz):
a = [\theta_d, K, B]

**Impedance control law:**
\tau = K(\theta_d - \theta) - B\dot{\theta}

* **Desired ankle angle:** \theta_d \in [-20°, 20°]
* **Ankle stiffness:** K \in [5, 500] N·m/rad
* **Ankle damping:** B \in [0.5, 20] N·m·s/rad
* **Motor torque limit:** \tau \in [-150, 150] N·m
*(Where \theta is actual ankle angle and \dot{\theta} is ankle angular velocity)*

## Reward Function
The total reward consists of three terms:
r = r_{survival} + r_{impact} + r_{smooth}

1. **Survival Reward:** r_{survival} = +1
   +1 reward for every step without falling. If the agent falls, the episode terminates.
2. **Impact Penalty:** r_{impact} = -w_1 [\max(0, \frac{F}{mg} - \lambda)]^2
   This penalizes excessive ground reaction forces above the selected threshold.
3. **Smoothness Penalty:** r_{smooth} = -w_2 ||a - a_{prev}||^2
   This penalizes large changes in desired ankle angle, stiffness, and damping between consecutive actions, encouraging smooth and realistic control.
