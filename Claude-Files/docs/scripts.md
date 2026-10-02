# `scripts/` — the two you run

**Owner: C**

---

## `smoke_train.py` — the week-3 gate

```bash
python scripts/smoke_train.py
```

**You are not looking for good behaviour.** You are looking for the reward curve
to trend *upward* rather than sideways. 50k steps, about ten minutes on a laptop.

If it is flat, the problem is almost certainly your environment, not the agent —
and discovering that in week 3 with a known-good algorithm is the entire point.
Discovering it in week 6 means you cannot tell which of the three is broken.

It also prints a first look at your headline numbers: mean return, fall rate and
peak GRF on **both** the training and held-out distributions.

What the settings mean:

| Argument | Why |
|---|---|
| `n_envs=8` | eight simulations stepping in parallel — PPO collects from all of them |
| `n_steps=256` | steps per env before each update → 2048 samples per batch |
| `deterministic=True` at eval | no exploration noise; you want the policy's actual best guess |

---

## `record_video.py` — the debugging tool

```bash
python scripts/record_video.py                 # random actions
python scripts/record_video.py runs/smoke_ppo  # a trained policy
```

**Use this in week 2, with random actions, long before any policy exists.** You
are checking the physics, not the behaviour:

- Does the foot pass through the floor?
- Does the leg jitter? (friction smoothing too sharp)
- Does the dent appear under the right contact point?
- Does the pylon spring explode?

Every one of those is invisible in a reward curve and obvious in two seconds of
video. Teams that skip this spend week 6 wondering why nothing learns, when the
answer was a sign error they could have seen.

### Headless machines

The script sets `MUJOCO_GL=egl` before importing MuJoCo. Without it, a machine
with no display (lab server, WSL, CI) fails with *"an OpenGL platform library
has not been loaded"*. Harmless on a laptop with a desktop session.

### Frame rate

100 frames at 20 fps = 5 s of slow motion. Real time would be 0.5 s — far too
fast to see an impact.

### The invisible-ground problem

Because MuJoCo's floor collision is off, the leg would appear to stand on
nothing and sink into empty space. `AnkleEnv.render()` moves the visual floor
geom down to match the dent, so you can actually *see* the ground deform and
stay deformed. That is the most distinctive thing about your simulation and
would otherwise be completely invisible.

### For the final report

Run the **same seed** through both controllers and stack the frames side by
side with `np.hstack`. On average ground they look near-identical. On a held-out
soft surface, one stays up and the other folds. That single clip is worth more
in a presentation than any plot.

Save the seed, the ground parameters and the model checkpoint alongside every
video — you will produce dozens and will not remember which was which.
