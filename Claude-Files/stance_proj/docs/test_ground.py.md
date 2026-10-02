# `tests/test_ground.py` — physics checks

**Owner: A** · This file *is* the "verified against closed-form results" claim
in your proposal. Run it after every change to the force law.

```bash
pytest tests/test_ground.py -v
```

## Two kinds of test

**Pure function checks** — call `normal_force` / `friction_force` directly with
known inputs and compare against hand arithmetic. Fast, exact, and they pin
down the sign errors that are otherwise invisible.

| Test | Checks |
|---|---|
| `test_linear_spring_exact` | `α=0` → `F = k·d` exactly |
| `test_no_contact_no_force` | above the surface, nothing |
| `test_ground_never_pulls` | rising fast cannot produce downward force |
| `test_stiffening_with_depth` | `α=1` at `d=D_REF` doubles `k` |
| `test_force_saturates_at_yield` | force capped, dent reported |
| `test_friction_opposes_motion` | **sign**, and `|F_t| ≤ μF_n` |

**Dynamics checks** — run a real simulation and compare against the textbook
mass-spring answers: settles at `mg/k`, rings at `√(k/m)/2π` Hz. Both within 2%.

## The trap we already hit — worth knowing

The obvious test is "drop the whole leg and check it settles at `mg/k`". It
fails, and the reason is instructive: **the leg is an inverted pendulum, so it
topples during the drop.** You cannot compare a toppling system against a 1-DOF
closed form.

The second attempt — clamping the other joints each step — was worse. Repeatedly
zeroing a *compressed* spring's position pumps energy in, and the leg shot
800 mm into the air.

The fix was structural: test the force law on a **minimal one-mass model**
(`MINIMAL` in the file) where the closed form genuinely holds. That is why
`ground.py` is a separate module.

Second trap: for the frequency test, the mass must **start in contact**, 3 mm
below equilibrium. Drop it from a height with light damping and it bounces
clear of the ground — that is a bouncing ball, a much slower and completely
different oscillation.

## Adding a test

If you change the force law, add a check with an answer you worked out by hand.
"It looks right in the video" is not a test.
