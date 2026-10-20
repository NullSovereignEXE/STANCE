# P3: the learning agent

**Your question:** what training makes the network actually learn? Plain PPO reached only 26% survival
after 300k steps, against 92.5% for the best fixed action, and it drifted the target angle the wrong way.
You own the environment, its reward and its termination, so you know where the problem sits. The agent is
the main program in the lecture's structure: it connects the network to the environment and decides how
learning happens.

## Your files

| File | What it does | Edit it? |
|---|---|---|
| `agent/config.py` | Every training setting (PPO hyperparameters, loss weights, normalisation, residual action, logging). The docstring explains each key. | **Yes** |
| `agent/train.py` | The main program. Loads a network, merges the configs, builds the environments and the PPO model, trains with a learning-curve logger, saves and evaluates. The docstring explains each step. | **Yes** |
| `agent/actions.py` | `ResidualAction`: the policy outputs an offset from a fixed anchor action. The docstring describes the edge-of-range problem. | **Yes** |
| `agent/curve.py` | Logs a learning-curve point every `curve_every` steps. | Yes |
| `agent/test_agent.py` | Tests the network contract, the config overrides and the residual action. | Yes, add tests for anything you add |
| `common/*` | Shared code. | Only the day-1 fixes below; ask before anything else |
| `networkA_mlp/`, `networkB_lstm/` | P1's and P2's networks. | **No** |

## Run it

Run from the `Network/` folder.

```bash
env -u PYTHONPATH python -m pytest agent common -q
env -u PYTHONPATH python -m common.evaluate --show-info                         # the environment's info keys
env -u PYTHONPATH python -m common.evaluate --reference                         # the best fixed action
env -u PYTHONPATH python -m agent.train --network mlp --smoke
env -u PYTHONPATH python -m agent.train --network mlp --run-name resid --set residual=True
env -u PYTHONPATH python -m agent.train --network mlp --run-name lowlr --set learning_rate=1e-4 n_epochs=5
env -u PYTHONPATH python -m agent.train --network mlp --run-name g995 --set gamma=0.995
env -u PYTHONPATH python -m agent.train --network mlp --run-name nonr --set norm_reward=False
env -u PYTHONPATH python -m agent.train --network mlp --run-name long --set total_steps=1000000
env -u PYTHONPATH python -m agent.train --eval-only runs/networkA_mlp/resid_s0 --test-split
```

`--set` changes any key from `agent/config.py` or from the network's `CONFIG` for one run. Use Network A (`mlp`)
as your test bed because it trains fastest.

## What you have to do

- [ ] **16-17 Oct, day-1 checks** (details in `00_WHO_OWN_WHAT.md`): the environment ids and constants work;
      pin the toe and reason keys in `INFO_KEYS`; the reference gives about 92.5% survival and 1.74 BW toe peak;
      commit the pinned keys and `results/reference_train_1000-1199.csv`; tell the group.
- [ ] **16-17 Oct:** send the professor the three questions (Stable-Baselines3 allowed, hand-written loss needed,
      separate deadlines). If he wants hand-written code, add a PPO implementation in `agent/` behind the same contract.
- [ ] **17 Oct:** baseline (`--network mlp --run-name base`). It should land near the earlier 26%.
- [ ] **17-21 Oct, the learning problem.** One change per run, 300k steps, seed 0:
  - [ ] `norm_reward=False`: reward normalisation can hide the fall penalty's size.
  - [ ] `gamma=0.995`: makes the late fall penalty count for more.
  - [ ] `learning_rate=1e-4` with `n_epochs=5`: slower, steadier updates.
  - [ ] `ent_coef=0.01`: more exploration.
  - [ ] `log_std_init=-1.0`: less starting noise (a network setting, but `--set` reaches it).
  - [ ] `total_steps=1000000`: is it slow learning or no learning?
- [ ] **18-21 Oct, the residual action.** Run `--set residual=True`. The first `curve.csv` row should already be
      close to the best fixed action's survival (what `--reference` prints) and far above plain PPO's 11%; if it
      starts near 11%, the offset is not active. Then solve the edge problem
      described in `agent/actions.py` and test it against plain `clip`:
  - [ ] Map the policy output only inward from the anchor (edit `actions.py`).
  - [ ] Or move the anchor inside the range, for example `--set residual=True "anchor=[-15.0,300.0,6.0]"`.
  - [ ] Or keep the clip and show with data that it is harmless.
  - Vary `scale` too: `"scale=[0.1,0.1,0.1]"`, `"scale=[0.6,0.6,0.6]"`.
- [ ] **22 Oct:** post "config v1": the best settings so far, written into `agent/config.py` and committed.
- [ ] **24 Oct:** freeze the final config in `agent/config.py`, commit it and tell the group. P1 and P2 run their
      final runs with it.
- [ ] **26 Oct:** comparison day; help decide which network goes into the agent.
- [ ] **27-29 Oct:** write the training-setup and loss parts of Report 2, then assemble the whole report.
- [ ] **31 Oct - 6 Nov, the agent for submission:**
  - [ ] Fix `render()` in the environment so `gym.make("StanceAnkle-v0", render_mode="rgb_array")` works.
        The committed version imports a deleted module and fails.
  - [ ] Record a rollout video like Lecture 6's `record_video`, using `VecVideoRecorder`. It needs `moviepy`,
        which is in `requirements.txt`.
  - [ ] Final benchmark of the chosen network, with P1 and P2 helping.
  - [ ] Optional if time allows: a tuned two-state impedance controller as a stronger classical baseline
        than the best fixed action.

## What you can change

Anything in `agent/`. Add keys to `AGENT_CONFIG` for anything new you make tunable and document them in the
docstring. A key must never appear in both `agent/config.py` and a network's `CONFIG` (`train.py` stops with an
error if it does). If you change what `train.py` expects from a network, tell P1 and P2, because the contract in
`00_WHO_OWN_WHAT.md` is the only link between you.

**Day-1 exception for `common/`:** `TRAIN_ID`, `TEST_ID` and `_consts()` in `env_setup.py`, and `INFO_KEYS` in
`evaluate.py`. Make these edits once, on day 1, and tell the group.

**Do not change:** the evaluation seeds or metrics, or the environment, without telling everyone. If you change
the environment, delete `results/reference_*.csv` and tell P1 and P2 to re-run.

## What you write in Report 2

1. **Training setup and loss**, following the lecture's checklist:
   - The loop: reset, choose an action from the network, step, record, update every `n_steps` steps.
   - The settings and why (from `agent/config.py`).
   - The loss PPO minimises: the clipped policy objective, the value-error term and the entropy bonus,
     with the weights in `agent/config.py`; gradient clipping (`max_grad_norm`) and the Adam optimiser.
   - What you tried for the learning problem and the numbers, including the ones that did not help.
2. **Integration:** merge the three parts and the comparison, make the figures consistent, keep within the page
   limit, fill in the AI-use declaration, and check every number in the text against `results/*.csv`.

## Hints

- Judge runs on survival, toe force and `mean_theta_deg`, not on return. Return is mostly a survival counter.
- The residual action starts near the best fixed action, so a policy that beats it must lower toe force without
  more falls. Look at `minus_reference` and `toe_peak_bw` together.
- If the policy collapses to a zero offset everywhere, it is stuck at the anchor. Try a larger `scale` or
  `log_std_init=-0.5`.
- You are the one who knows the environment. If anything looks wrong (odd info keys, a reference that does not
  match), fix it with the group, not silently.

**Done when:** `agent/config.py` holds the frozen final config, the residual question has an answer backed by
runs, `results/` contains your experiments, and the assembled Report 2 and the agent submission are finished.
