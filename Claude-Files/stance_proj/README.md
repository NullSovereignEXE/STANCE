# STANCE

Powered prosthetic ankle learning to set its own impedance on unknown
compliant ground. Group 42 — Vishal, Mukul, Muthu.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
pytest -q                       # expect: 16 passed
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
python scripts/record_video.py     # watch a rollout  (week 2)
python scripts/smoke_train.py      # 50k-step sanity check (week 3)
```

## Layout and ownership

| Path | Owner | What |
|---|---|---|
| `model/leg.xml` | A | the robot: 5 DOF, 1 motor |
| `stance_env/ground.py` | A | the ground force law (pure functions) |
| `stance_env/ankle_env.py` | B | the Gym environment — **the two TODOs are here** |
| `stance_env/__init__.py` | B | registration |
| `tests/test_ground.py` | A | closed-form physics checks |
| `tests/test_env.py` | B | Gym API conformance |
| `scripts/` | C | video + smoke training |

One file per person, so no merge conflicts in week 1.

Each file has a plain-language explainer in `docs/`.

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

Two TODOs in `ankle_env.py`: `_reward` and `_failed`. Working defaults are in
place so the environment runs end to end, but the weights are a starting point,
not an answer. These are where the project's argument lives — do not outsource
them.
