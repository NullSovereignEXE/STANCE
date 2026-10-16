# Who owns what

Three people, one Python file each, no shared files. Nobody edits anyone
else's file; if you need something from it, ask for a function.

| Person | Owns | Also owns |
|---|---|---|
| **Muthu — Leg & motion** | `model/leg.xml`, `stance_env/viz.py` | `tests/test_leg.py` |
| **Mukul — Ground & deformation** | `stance_env/ground.py` | `tests/test_ground.py` |
| **Vishal — RL cast & Gym** | `stance_env/ankle_env.py`, `stance_env/__init__.py` | `tests/test_env.py` |
| **Whoever finishes first** | `scripts/` — PPO, evaluation, the baseline controller | |

## The three interfaces — agree these, then stop coordinating

Everything crosses between files through exactly three contracts. Change one
and you must tell the other two; change anything else and you need not.

**1. Ground → Environment**

```python
forces = ground.forces(positions, velocities)   # (n,3) in, (n,3) out
ground.sample(rng)                              # new episode
ground.last_normal                              # float, for obs and reward
```

No MuJoCo types cross this line. P2 can rewrite the entire force law without
P3 knowing.

**2. Leg (XML) → Environment**

Joint order, site names, and the site offsets:

```
joints: foot_x, foot_z, foot_pitch, ankle_hinge, pylon_slide
sites:  heel_site, toe_site
HEEL_OFFSET = (-0.07, -0.05)      TOE_OFFSET = (0.19, -0.05)
```

`tests/test_leg.py` fails loudly if P1 changes any of these without telling P3.
That is its main job.

**3. Environment → everyone**

The 11-value observation order, and the 3-value action order. Fixed. Never
reorder without a group decision — the network's input is positional.

## Day-one order

P1 and P2 can start immediately and independently. P3 depends on both, but the
file already runs end to end, so P3 can work on reward and termination from the
first hour.

## Running it

```bash
pip install -r requirements.txt && pip install -e .
pytest -q                                   # expect 27 passed
python scripts/record_video.py              # watch a rollout
```

Each person runs `pytest tests/test_<theirs>.py` before every commit, and the
full suite before every push.
