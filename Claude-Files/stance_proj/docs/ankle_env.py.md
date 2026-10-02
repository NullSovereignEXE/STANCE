# `stance_env/ankle_env.py` — the environment

**Owner: B** · The Gymnasium contract. This is where the proposal's logic lives.

## What a Gym environment is

A **contract**. The learning algorithm knows nothing about prosthetics or
MuJoCo. It knows it can call two functions and gets arrays back:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

Plus two declarations: `action_space` and `observation_space`. That is the
entire interface. Everything else is yours.

## The loop PPO runs, millions of times

```python
obs, info = env.reset(seed=0)
done = False
while not done:
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## `terminated` vs `truncated` — get this right

The single most common beginner bug, and it corrupts training silently.

| Flag | Means | Here |
|---|---|---|
| `terminated` | the episode genuinely ended; no future exists | the leg fell |
| `truncated` | **you** cut it off; the leg was fine | 100 steps elapsed |

The critic estimates "how much reward is still to come". After a fall the
answer is zero. After a time limit it is **not** zero — the leg would have kept
earning +1 per step. Report a time limit as `terminated` and the critic learns
that surviving is worthless.

## `reset()` — three jobs

**1. Seed properly.** `super().reset(seed=seed)` sets up `self.np_random`. Use
it for *every* random draw. Use bare `np.random` and your runs are not
reproducible, which makes the three-seed comparison in your report meaningless.

**2. Draw the ground.** Per contact point, from `GROUND_RANGES[split]`. Heel
and toe are drawn **independently** — that asymmetry is what tips the foot.

**3. Place the leg so the heel is exactly touching.** The one piece of real
maths. Rotating a foot-frame offset `(x, z)` about +y by `φ` gives world offset
`(x cos φ + z sin φ, −x sin φ + z cos φ)`. Setting heel world height to zero:

```
foot_z = x_heel · sin(φ) − z_heel · cos(φ)
```

Sanity check: at `φ = 0` this gives 0.05 m, which is exactly the site's z
offset. Get this wrong and every episode starts either already penetrating the
ground or in free fall.

## `step()` — the order matters

```
for 5 physics steps:          ← frame_skip: 1 ms physics, 200 Hz control
    apply ground forces
    compute impedance torque   ← INSIDE the loop, see below
    mj_step
    clear qfrc_applied         ← forces are per-step, not sticky
```

**The torque is recomputed every physics step, not once per control step.**
It depends on ankle angle and velocity, which change every millisecond.
Computing it once would make the ankle far less responsive than intended.

## The action: why logarithmic

The policy outputs three numbers in `[-1, 1]`; `_rescale` maps them to physical
units. Stiffness spans 5–500 N·m/rad — a factor of 100. A **linear** map would
put 250–500 in half the action range and squeeze 5–15 into the last 2%, making
soft behaviour nearly unreachable. The log map gives equal resolution per
*factor* rather than per absolute increment.

Note also: the ±150 N·m clamp sits **outside** the network. Whatever the policy
outputs, the device's maximum force is fixed.

## The state: 11 values

Ankle angle and rate (encoder) · ankle torque (motor current) · vertical GRF
(pylon load cell) · pylon compression · shank tilt and rate (IMU) · shank
vertical acceleration · previous action ×3.

Every one exists on real hardware. **Ground properties are never included** —
inferring them is the agent's entire job.

The network sees the last **10 steps stacked** (50 ms). This matters: a single
snapshot says "400 N right now", which reads identically on firm and soft
ground. What separates them is how fast the force *rose*, and that only exists
across time.

> The reward is computed inside the simulator, so it *may* use privileged state
> (true body height) that the policy cannot see. Restricting the observation
> does not restrict reward design.

## The two TODOs — do not outsource these

**`_reward`** — three terms: survival `+1`, impact penalty, smoothness penalty.
The weights given are a starting point, not an answer. The one ratio that
matters is `W_IMPACT` against the survival bonus: *how many steps of successful
support is one body weight of excess force worth?*

Expect your first policy to sit at maximum stiffness. That is not a mysterious
failure — it is the fixed-gain baseline's behaviour, and the fix is to raise
`W_IMPACT`.

**`_failed`** — toppled (tilt > 15°) or collapsed (body below 70% height).

## Constants at the top

| Name | Value | Change it if… |
|---|---|---|
| `HISTORY_LEN` | 10 | running the history ablation (10 / 3 / 1) |
| `MAX_TILT` | 15° | the leg topples too easily to ever learn |
| `LAMBDA_BW` | 2.0 | normal walking peaks near 1.1–1.2 BW |
| `W_IMPACT` | 10.0 | **the main tuning knob** |
| `W_SMOOTH` | 0.5 | keep small; too large and the policy goes sluggish |
