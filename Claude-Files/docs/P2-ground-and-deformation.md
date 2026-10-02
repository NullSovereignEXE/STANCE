# P2 — The ground and its deformation

**You own:** `stance_env/ground.py`, `tests/test_ground.py`

Everything about the surface: the force law, the material sampling, the dent
that persists, and how the dent is drawn. One file. You can rewrite all of it
without touching anyone else's.

---

## The contract (don't break this)

```python
ground.sample(rng)                              # new episode
forces = ground.forces(positions, velocities)   # (n,3) in -> (n,3) out
ground.last_normal                              # float, read by obs and reward
ground.update_visual(model)                     # show the dent
```

No MuJoCo types cross that line except `update_visual`. Change anything inside
and P3 never needs to know.

---

## The force law

```
F = k(d)·d + c·ḋ          where   k(d) = k₀(1 + α·d/D_REF)
```

A spring plus a damper:

- **`k(d)·d`** — Hooke's law. Twice as deep, twice the force.
- **`c·ḋ`** — damping. Resists *how fast* you sink. This is what makes soft
  ground feel dead rather than bouncy.
- **`k` grows with depth** — sand and mud firm up as you sink in. A plain
  spring does not do this.

Three behaviours:

| Condition | Result | Why |
|---|---|---|
| `depth <= 0` | 0 | foot is above the surface |
| `F < 0` | clamped to 0 | **ground pushes, never pulls** |
| `F > f_yield` | capped, extra dent returned | plastic yield |

### Why yield is the important one

Returning a second value (`extra_dent`) is how the function says *"the surface
moved down permanently."* That is what makes the ground non-memorisable: its
stiffness now depends on how the agent has been loading it. No fixed parameter
set describes it — which is one of the two structural arguments in your
proposal.

### The bug already found here

Damping can push `F` past `f_yield` while the *spring* term is still below it
(fast sinking, shallow depth). Then `depth < elastic_depth`, the naive
difference is negative, and **the surface rises**. A real rollout showed a dent
of −12.3 mm.

Fixed with `max(depth - elastic_depth, 0.0)`, and two regression tests now
guard it. Worth understanding: it is the kind of sign/ordering error that
produces plausible-looking nonsense rather than a crash.

---

## Friction

```
F_t = −μ · F_n · tanh(v / V_EPS)
```

**The minus sign matters.** Friction opposes motion; without it the foot
accelerates in whatever direction it is already sliding.

Textbook Coulomb switches abruptly between stuck and sliding. That
discontinuity makes an explicit integrator chatter — visible jitter you cannot
tune away. `tanh` smooths it over a narrow band (`V_EPS` = 1 mm/s).

The cost: the foot *creeps* instead of truly sticking, under 0.5 mm over an
episode. Your failure mode is the foot sliding **out** — large and fast, where
both models agree exactly. Name the approximation in Report 1 rather than let
someone find it.

---

## Material sampling

`GROUND_RANGES` holds training and held-out ranges. Heel and toe draw
**independently** — that asymmetry is the mechanical link between unknown
ground and falling: softer under the heel, foot tilts, leg tilts, body is
pushed toward the edge of the foot.

`k0` supports **multiple disjoint bands** (the test split has two: very soft
and very firm). The sampler picks a band, then samples log-uniformly inside it.

> **The test ranges must never overlap the training ranges.**
> `test_train_and_test_ranges_are_disjoint` checks this across 60 seeds. It is
> your professor's memorisation critique, answered with a check rather than a
> paragraph.

---

## `update_visual` — your other visual job

MuJoCo's floor collision is off, so without this the leg appears to stand on
nothing and sink into empty space. The current version drops the whole visual
floor to the deepest dent.

**The better version, and a good task for you:** a strip of thin boxes along
the walking direction, each with its own height, so heel and toe visibly dent
*differently*. That makes the asymmetry — the project's core mechanism —
actually visible on screen instead of implied. Roughly 20 lines in `leg.xml`
plus a loop here.

Coordinate with P1 only on where the geoms live in the XML; the logic is yours.

---

## Your tests

```bash
pytest tests/test_ground.py -v
```

**Pure checks** — call the functions with known inputs, compare against hand
arithmetic. Exact `k·d` at `α=0`, doubling at `d=D_REF` with `α=1`, saturation
at yield, friction sign, dent never negative.

**Dynamics checks** — settles at `mg/k`, rings at `√(k/m)/2π` Hz, both within
2%.

### Two traps already hit, worth knowing

**The full leg cannot be used for a drop test.** It is an inverted pendulum and
topples during the drop, which invalidates any 1-DOF closed form. Clamping the
other joints was worse — repeatedly zeroing a *compressed* spring pumps energy
in, and the leg shot 800 mm into the air. The dynamics tests use a minimal
one-mass model instead, which is why the force law is a pure function.

**The frequency test must start in contact**, 3 mm below equilibrium. Drop from
a height with light damping and it bounces clear of the ground — a bouncing
ball, a completely different and much slower oscillation.

---

## Week 1 for you

1. Read `ground.py` top to bottom; run `pytest tests/test_ground.py -v`.
2. Change `D_REF` or a range, watch which tests move.
3. `python scripts/record_video.py` and look at the dent.
4. Source real numbers: terrain stiffness, damping and friction for concrete,
   rubber matting, turf, gravel, sand. Your proposal promises these come from
   published data — someone has to find it, and it is your file. Half a day of
   reading that also feeds Report 1's related work.
5. Then: the per-cell visual strip.
