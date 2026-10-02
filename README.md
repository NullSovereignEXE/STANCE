# STANCE: Stiffness and Torque Adaptation for Non-linear Compliant Environments

**Group 42:** Vishal Mangla, Mukul Yadav, Muthaiah Mahadevan

## Why (Problem and Proposed Solution)
Most passive lower-limb prostheses use a fixed-stiffness ankle where a prosthetist sets its compliance once, and it stays the same regardless of terrain. This works fine on flat, rigid, predictable ground, where that single setting can be tuned in advance.

Powered prostheses improve on this by adjusting for walking speed and different rigid terrains. But no existing control strategy adjusts for compliant ground, where the surface itself gives way unpredictably underfoot, because the ankle can't be pre-set for mechanics it won't know until the moment it lands. As a consequence, more than half of lower-limb prosthesis users fall at least once a year, many during "weight acceptance": the 100-200 ms window after heel strike when full body weight shifts onto the prosthetic leg. In that window, the ankle must get its stiffness right immediately. Too stiff, and it slams the impact into the residual limb, causing pain. Too compliant, and the leg simply folds.

Which failure occurs isn't up to the user. It's decided by the ground. Concrete, wet grass, gravel, and sand compress, damp, and grip differently, and some stay dented afterward, so a stiffness that's safe on one surface can fail on the next. No commercial sensor can measure ground compliance before contact, and vision doesn't help either. We aim to train a single agent that controls a powered ankle by setting three things at once: target angle, stiffness, and damping. Moreover, we'll use only real prosthesis signals: encoder, motor current, load cells, and IMU.

## Why Reinforcement Learning

| Method | Why it fails here | What RL does instead |
| :--- | :--- | :--- |
| **FSM impedance control (clinical standard)** | Stiffness is fixed in advance for a listed set of situations. The surface about to be landed on is not on that list. | Learning continuous mapping from sensor history to stiffness, with no list of surfaces to enumerate. |
| **Online estimation + adaptive control** | Needs force data, that only appears after impact begins, to estimate stiffness. | Sets stiffness at touchdown from what it learned in training, then refines it as force builds - rather than waiting for a measurement that arrives too late to use. |
| **Model Predictive Control** | Requires a contact model. No fixed parameter set describes stochastic ground deformation. Solving a contact problem at 200 Hz on a prosthesis is unrealistic. | Moves the optimization offline into training. Inference is a fixed, small cost per step. |

* **Structural Underactuation:** The system is heavily underactuated. Five degrees of freedom, one motor. No mathematical inverse exists from a desired body motion to an ankle torque. This is structural, not a matter of difficulty.
* **Non-Stationary Ground:** The ground cannot be memorized. Properties are drawn from continuous ranges, re-drawn every episode, and the surface dents permanently. The ground's effective stiffness depends on how the agent itself has been loading it. There is no fixed environment parameter to memorize.

## Algorithm & Evaluation
**Algorithm:** PPO (Proximal Policy Optimization). It is an actor-critic method, matching the network above, which suits our continuous three-value action space. PPO is on-policy. Since our simulation is cheap (5 DOF, 100-step episodes), generating data is not the bottleneck. Furthermore, by limiting how far the policy can change in a single step, it makes training tolerant of an imperfect reward function. This algorithm will be compared against a fixed-gain FSM impedance controller.

**Evaluation Metrics:**
Each of the following metrics will be averaged over at least three training runs started from different random seeds and reported as mean ± standard deviation, since any single run can succeed or fail by chance.
1. Peak ground reaction force (in body weights) against ground stiffness.
2. Fall rate, split by failure type: tipping over versus collapsing.
3. Generalization: metrics 1 and 2 measured on training ground versus testing ground.
4. History length: the same experiment repeated with 10, 3 and 1 stacked steps. If shortening the history makes performance worse, the agent is genuinely reading the ground from the force pattern rather than ignoring it.

## AI Declaration
We used Claude Opus 5 (Anthropic) to evaluate ideas, summarize research papers that were used to justify the proposal and to simplify technical concepts. We are completely responsible for the content and quality of the submitted work.
