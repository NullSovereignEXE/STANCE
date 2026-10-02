# `tests/test_env.py` — API conformance

**Owner: B** · If these fail, Stable-Baselines3 misbehaves in ways that look
like learning problems but are not. Run before every training run.

```bash
pytest tests/test_env.py -v
```

| Test | Why it matters |
|---|---|
| `test_registered` | both ids reachable via `gym.make` |
| `test_spaces_match_observations` | shape and `float32` dtype match the declared space |
| `test_seeding_is_reproducible` | same seed → identical episode |
| `test_different_seeds_give_different_ground` | randomisation actually randomises |
| `test_train_and_test_ranges_are_disjoint` | **the anti-memorisation claim, verified** |
| `test_terminated_and_truncated_are_distinct` | never both for the same reason |
| `test_sb3_env_checker` | SB3's own official conformance check |

## The two that earn their place in your report

**`test_seeding_is_reproducible`** — your final report promises results as
mean ± std over at least three seeds. If seeding is broken, those numbers are
noise and the comparison means nothing.

**`test_train_and_test_ranges_are_disjoint`** — samples 60 seeds from each split
and asserts no overlap in `k0`. Your professor rejected the last proposal partly
over memorising randomness. This converts your defence from a claim into
something checkable.

## `importorskip`

`test_sb3_env_checker` calls `pytest.importorskip("stable_baselines3")`, so the
suite still runs if SB3 is not installed — that test just skips. You will see
`15 passed, 1 skipped` rather than an error.
