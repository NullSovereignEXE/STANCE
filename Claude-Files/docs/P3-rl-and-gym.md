# P3 — The RL cast and Gymnasium

**You own:** `stance_env/ankle_env.py`, `stance_env/__init__.py`,
`tests/test_env.py`

---

## What a Gym environment is

A **contract**. The learning algorithm knows nothing about prosthetics or
MuJoCo. It knows it can call two functions and gets arrays back:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

Plus two declarations: `action_space` and `observation_space`. That is the
entire interface.

---

## `terminated` vs `truncated` — get this right

The most common beginner bug, and it corrupts training **silently**.

| Flag | Means | Here |
|---|---|---|
| `terminated` | genuinely ended; no future exists | the leg fell |
| `truncated` | **you** cut it off; the leg was fine | 100 steps elapsed |

The critic estimates "how much reward is still to come". After a fall: zero.
After a time limit: **not** zero — the leg would have kept earning +1 per step.
Report a time limit as `terminated` and the critic learns surviving is
worthless.

---

## Registration (`__init__.py`)

Importing the package registers two ids:

| id | ground | Use |
|---|---|---|
| `StanceAnkle-v0` | train | training |
| `StanceAnkleTest-v0` | test | **evaluation only** |

Separate ids rather than a flag, so you cannot accidentally train on held-out
ground. The gap between them is your headline result; if they mix, it is
worthless.

`gym.make` wraps you in `TimeLimit` (enforces 100 steps, sets `truncated`),
`OrderEnforcing` (errors if you `step` before `reset`) and `PassiveEnvChecker`
(validates shapes and dtypes).

### Adding ablation variants

`history_len` is already an `__init__` argument, so the history ablation is
three `register` calls:

```python
for n in (1, 3, 10):
    register(id=f"StanceAnkleH{n}-v0",
             entry_point="stance_env.ankle_env:AnkleEnv",
             max_episode_steps=100,
             kwargs={"ground_split": "train", "history_len": n})
```

---

## `reset()` — three jobs

**Seed properly.** `super().reset(seed=seed)` sets up `self.np_random`. Use it
for *every* random draw. Use bare `np.random` and runs are not reproducible —
which makes the three-seed comparison in your report meaningless.

**Draw the ground.** One call: `self.ground.sample(self.np_random)`.

**Place the leg so the heel is exactly touching.** The one piece of real maths.
Rotating a foot-frame offset `(x, z)` about +y by `φ` gives world offset
`(x cos φ + z sin φ, −x sin φ + z cos φ)`. Setting heel world height to zero:

```
foot_z = x_heel · sin(φ) − z_heel · cos(φ)
```

At `φ = 0` this gives 0.05 m — exactly the site's z offset. Useful check that
the constants still match P1's XML.

---

## `step()` — order matters

```
for 5 physics steps:            ← frame_skip: 1 ms physics, 200 Hz control
    apply ground forces
    compute impedance torque     ← INSIDE the loop
    mj_step
    clear qfrc_applied           ← forces are per-step, not sticky
```

**The torque is recomputed every physics step**, not once per control step. It
depends on ankle angle and velocity, which change every millisecond. Computing
it once makes the ankle far less responsive than intended.

---

## The action: why logarithmic

The policy outputs three numbers in `[-1, 1]`; `_rescale` maps them to physical
units. Stiffness spans 5–500 N·m/rad — a factor of 100. A **linear** map puts
250–500 in half the range and squeezes 5–15 into the last 2%, making compliant
behaviour nearly unreachable. The log map gives equal resolution per *factor*.

The ±150 N·m clamp sits **outside** the network: whatever the policy outputs,
the device's maximum force is fixed.

---

## The state: 11 values

Ankle angle and rate (encoder) · ankle torque (motor current) · vertical GRF
(pylon load cell) · pylon compression · shank tilt and rate (IMU) · shank
vertical acceleration · previous action ×3.

Every one exists on real hardware. **Ground properties are never included** —
inferring them is the agent's entire job.

The network sees the last **10 steps stacked** (50 ms). This is the point: a
single snapshot says "400 N right now", identical on firm and soft ground. What
separates them is how fast the force *rose*, which only exists across time.

> The reward runs inside the simulator, so it **may** use privileged state
> (true body height) the policy cannot see. Restricting the observation does
> not restrict reward design.

---

## Your two real tasks

**`_reward`** — survival `+1`, impact penalty, smoothness penalty. The weights
are a starting point, not an answer. The one ratio that matters is `W_IMPACT`
against the survival bonus: *how many steps of successful support is one body
weight of excess force worth?*

Expect the first policy to sit at **maximum stiffness**. That is not a
mysterious failure — it is the fixed-gain baseline's behaviour, reached by a
different route, and the fix is to raise `W_IMPACT`.

**`_failed`** — toppled (tilt > 15°) or collapsed (body below 70% height).

Both have working defaults so everything runs. They are where the proposal's
argument lives — do not outsource them.

---

## Constants to know

| Name | Value | Change when |
|---|---|---|
| `HISTORY_LEN` | 10 | the history ablation (10 / 3 / 1) |
| `MAX_TILT` | 15° | the leg topples before it can ever learn |
| `LAMBDA_BW` | 2.0 | normal walking peaks near 1.1–1.2 BW |
| `W_IMPACT` | 10.0 | **the main tuning knob** |
| `W_SMOOTH` | 0.5 | keep small, or the policy goes sluggish |

---

## Your tests

```bash
pytest tests/test_env.py -v
```

Two earn their place in the report:

**`test_seeding_is_reproducible`** — your report promises mean ± std over three
seeds. If seeding is broken those numbers are noise.

**`test_train_and_test_ranges_are_disjoint`** — 60 seeds per split, no overlap.
Your professor rejected the last proposal partly over memorising randomness.
This converts the defence into something checkable.

`test_sb3_env_checker` runs Stable-Baselines3's own conformance check.

---

## Week 1 for you

1. Read `ankle_env.py` top to bottom. It already runs end to end.
2. `pytest tests/test_env.py -v`.
3. Write `_reward` and `_failed` properly.
4. Train PPO on stock `InvertedPendulum-v5` — ten lines — and watch a learning
   curve go up. You will have done the whole weeks 4–6 pipeline on a solved
   problem, which de-risks week 6 almost entirely.
