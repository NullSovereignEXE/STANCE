# `stance_env/__init__.py` — registration

**Owner: B** · Short file, but it is what makes this a *real* Gym environment.

## What registration does

Without it you must import the class by hand:

```python
from stance_env.ankle_env import AnkleEnv
env = AnkleEnv()
```

With it, the environment has a **string id**:

```python
import gymnasium as gym
import stance_env          # the import itself does the registering
env = gym.make("StanceAnkle-v0")
```

This matters because Stable-Baselines3, the env checker, and the vectorised
wrappers all look environments up **by id**. `make_vec_env("StanceAnkle-v0",
n_envs=8)` only works if the id exists.

## What `gym.make` wraps around you, for free

```
TimeLimit< OrderEnforcing< PassiveEnvChecker< AnkleEnv >>>
```

- **TimeLimit** — enforces `max_episode_steps=100` and sets `truncated`
- **OrderEnforcing** — errors if you call `step()` before `reset()`
- **PassiveEnvChecker** — validates shapes and dtypes on the first calls

Note the consequence: because `TimeLimit` handles truncation, the environment's
own `step_count` check is belt-and-braces. Both agree, which is fine.

## Two ids, deliberately

| id | `ground_split` | Use |
|---|---|---|
| `StanceAnkle-v0` | `"train"` | training |
| `StanceAnkleTest-v0` | `"test"` | **evaluation only** |

Separate ids rather than a flag so you cannot accidentally train on held-out
ground. The gap between the two is the headline result of your project — if
they ever mix, that result is worthless.

`tests/test_env.py::test_train_and_test_ranges_are_disjoint` checks they never
overlap, across 60 seeds. That turns the anti-memorisation argument in your
proposal from a prose claim into something verified.

## Adding a variant

For the history ablation you will want more ids. Add here:

```python
register(
    id="StanceAnkleShortHistory-v0",
    entry_point="stance_env.ankle_env:AnkleEnv",
    max_episode_steps=100,
    kwargs={"ground_split": "train", "history_len": 3},
)
```

(That needs `history_len` added as an `__init__` argument first.)

## Why `# noqa: F401`

Linters flag `import stance_env` as unused, because nothing references the
name. It is not unused — the *side effect* of importing is the registration.
The comment tells the linter to leave it alone.
