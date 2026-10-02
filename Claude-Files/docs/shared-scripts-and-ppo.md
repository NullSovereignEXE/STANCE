# Shared — scripts, PPO, validation

**Owner: whoever finishes first.** Nobody owns `scripts/` so anyone can pick it
up without blocking a teammate.

---

## `scripts/record_video.py` — use it in week 2

```bash
python scripts/record_video.py                        # random actions
python scripts/record_video.py --split test --track   # held-out ground, tracking camera
python scripts/record_video.py --policy runs/smoke_ppo
```

Run it with **random actions**, long before any policy exists. You are checking
physics, not behaviour: foot through the floor, leg jitter, dent under the
wrong point, pylon spring exploding. All invisible in a reward curve, all
obvious in two seconds of video.

This already paid off — a real rollout printed a dent of **−12.3 mm**, which is
a surface that *rose*. That was a sign bug in the yield branch, now fixed and
guarded by tests.

---

## `scripts/smoke_train.py` — the week-3 gate

```bash
python scripts/smoke_train.py
```

50k steps, roughly ten minutes on a laptop. **You are not looking for good
behaviour.** You want the reward curve trending *upward* rather than sideways.

If it is flat, the problem is almost certainly the environment, not the agent —
and finding that in week 3, with a known-good algorithm, is the whole point.
Finding it in week 6 means you cannot tell which of the three layers is broken.

It also prints first-look numbers on both distributions: mean return, fall
rate, peak GRF.

| Setting | Why |
|---|---|
| `n_envs=8` | eight simulations in parallel; PPO collects from all |
| `n_steps=256` | → 2048 samples per update |
| `deterministic=True` at eval | no exploration noise; the policy's actual best guess |

---

## Still to be written

**The baseline controller.** A fixed-gain FSM impedance controller — the
clinical standard. Two states (contact / no contact) triggered by a force
threshold, with one fixed `(θ_d, K, B)` each, chosen by grid search on the
*average* training ground.

This is the thing your learned policy must beat, and it must be tuned properly.
A badly tuned baseline is a straw man and a reviewer will say so.

**The evaluation script.** The four figures from the proposal:

1. Peak GRF (body weights) vs ground stiffness, learned vs baseline
2. Fall rate, split by failure type
3. Generalisation gap — 1 and 2 on training vs held-out ground
4. History ablation — 10, 3, 1 stacked steps

All as mean ± std over **at least three seeds**.

Figure 4 is the one that turns this from a project into a result: if the
1-step policy does as well as the 10-step one, the agent never learned to read
the ground, and your central claim is false. That is why it is worth running.

---

## Housekeeping that will save you

**Save the seed, the ground parameters and the model checkpoint alongside every
video.** You will produce dozens of clips over two weeks and will not remember
which was which.

**Freeze the reward at the end of week 6.** Weeks 7–8 are tuning and
benchmarking. Any reward change after that invalidates every result collected
before it.

**Run the full suite before every push**, not just your own tests. The
interfaces between the three files are exactly where a silent break hides.
