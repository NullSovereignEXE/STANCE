# STANCE

Powered prosthetic ankle learning to set its own impedance on unknown
compliant ground. Group 42 — Vishal, Mukul, Muthu.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
pytest -q                       # expect: 27 passed
```

## Use

```python
import gymnasium as gym
import stance_env                        # the import registers the envs

env = gym.make("StanceAnkle-v0")         # training ground
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`StanceAnkleTest-v0` is the held-out distribution. **Evaluation only.**

```bash
python scripts/record_video.py --track   # watch a rollout   (week 2)
python scripts/smoke_train.py            # 50k sanity check  (week 3)
```

## Who owns what

| Person | Files |
|---|---|
| **P1 — leg & motion** | `model/leg.xml`, `stance_env/viz.py`, `tests/test_leg.py` |
| **P2 — ground & deformation** | `stance_env/ground.py`, `tests/test_ground.py` |
| **P3 — RL cast & Gym** | `stance_env/ankle_env.py`, `stance_env/__init__.py`, `tests/test_env.py` |
| **first one free** | `scripts/` — PPO, baseline controller, evaluation |

One Python file each, no shared files. Start with `docs/00-WHO-OWNS-WHAT.md`,
then your own `docs/P<n>-*.md`.

## Git, day one

```bash
git init && git add -A
git commit -m "STANCE: environment skeleton"
git remote add origin <your repo>
git push -u origin main
```

Branch per person (`feat/ground-model`, `feat/reward`), PR into main, and
**nobody merges code a teammate cannot explain** — the explanation you give on
Tuesday is the report paragraph you write on Friday.

## Outstanding

`_reward` and `_failed` in `ankle_env.py`. Working defaults are in place so the
environment runs end to end, but the weights are a starting point, not an
answer. These are where the project's argument lives — do not outsource them.
