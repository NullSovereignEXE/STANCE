# `stance_env/ground.py` — the force law

**Owner: A (physics)** · Two pure functions. No MuJoCo, no Gym, no state.

## Why it is a separate file

It is the part most likely to hide a sign error, and a pure function can be
tested against closed-form answers. The environment imports from here, so A can
change the physics and B's environment picks it up with no coordination.

## `normal_force(depth, sink_rate, k0, c, alpha, f_yield)`

How hard the ground pushes back at one contact point.

```
F = k(d)·d + c·ḋ          where   k(d) = k₀(1 + α·d/D_REF)
```

Read it as a spring plus a damper:

- `k(d)·d` — **Hooke's law.** Push in twice as far, get twice the force.
- `c·ḋ` — **damping.** Resists how fast you sink. This is what makes soft
  ground feel dead rather than bouncy.
- `k` grows with depth — **sand and mud firm up as you sink in.** A plain
  spring does not do this.

Three behaviours to know:

| Condition | What happens | Why |
|---|---|---|
| `depth <= 0` | returns 0 | foot is above the surface |
| `F < 0` | clamped to 0 | **ground pushes, never pulls** |
| `F > f_yield` | capped at `f_yield`, returns extra dent | plastic yield |

The yield case is the important one. Returning a second value (`extra_dent`)
is how the function tells the caller *"the surface moved down by this much,
permanently."* That is what makes the ground non-memorisable: its stiffness
now depends on how the agent has been loading it.

## `friction_force(normal, tangential_velocity, mu)`

```
F_t = −μ · F_n · tanh(v / V_EPS)
```

**The minus sign matters.** Friction opposes motion. Without it the foot
accelerates in whatever direction it is already sliding.

Textbook Coulomb friction switches abruptly between stuck and sliding. That
discontinuity makes an explicit integrator chatter — visible jitter you cannot
tune away. `tanh` smooths the transition over a narrow band (`V_EPS` = 1 mm/s).

The cost: the foot *creeps* instead of truly sticking. Over a 0.5 s episode
that is under half a millimetre, and your failure mode is the foot sliding
*out* — a large fast motion where the two models are identical.

## Tunable constants

| Name | Value | Meaning |
|---|---|---|
| `D_REF` | 0.02 m | depth at which `α` has its full effect |
| `V_EPS` | 1e-3 m/s | friction smoothing width — smaller is stiffer and jitterier |

## Testing

`tests/test_ground.py` checks this file against hand calculations:
exact `k·d` with `α = 0`, doubling at `d = D_REF` with `α = 1`, saturation at
yield, friction sign, and — on a minimal one-mass model — settling at `mg/k`
and ringing at `√(k/m)`.

```bash
pytest tests/test_ground.py -v
```

Run them after **every** change here.
