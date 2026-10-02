# Agent

The reinforcement learning agent that controls the powered prosthetic ankle.

At each control step the agent sets three outputs at once:
- **Target angle**: the ankle joint setpoint
- **Stiffness**: the impedance controller's spring term
- **Damping**: the impedance controller's damper term

It gets only signals a real prosthesis can measure: encoder, motor current, load cells, and IMU.

This folder holds the agent and policy code, the training loop, and the evaluation scripts.
