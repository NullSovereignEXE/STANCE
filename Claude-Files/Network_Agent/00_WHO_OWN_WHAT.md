# Who owns what: neural network and agent

**Deadlines:** the neural network and Report 2 are due **30 Oct**, the learning agent is due **6 Nov**.
Check both dates against the course brief.

## The idea in one picture

The lecture splits a learning project into three modules. We already built the third.

```
   Neural network  <---->  Learning agent  <---->  Gym environment
   (function approximator)  (main program,          (our StanceAnkle env,
                             the interface)          already done)
```

- **Network:** the layers that turn an observation into an action and a value.
- **Agent:** the program that resets the environment, asks the network for actions, steps
  the environment, records what happened and updates the network. It also owns the
  training settings (learning rate, discount, loss weights) and the evaluation.

## Who does what

| Person | Role | Folder | The question it answers |
|---|---|---|---|
| **P1 (Muthu)** | **Network A**: feed-forward network on the stacked history, plus the input design | `networkA_mlp/` | Which sensors and how much history does the network really need? |
| **P2 (Mukul)** | **Network B**: recurrent (LSTM) network on single frames | `networkB_lstm/` | Does the network need its own memory of the stance, to cope with hidden ground? |
| **P3 (Vishal)** | **The learning agent**: training loop and settings, residual action, evaluation, rollout video | `agent/` | What training makes the network actually learn (plain PPO got 26% survival)? |

Why this split: P1 built the leg and knows what the 11 sensors mean. P2 built the ground, whose
stiffness and dent history are the hidden state a memory could pick up. P3 owns the environment,
its reward and termination, so P3 knows where the learning problem sits. Swap on day 1 if you like.

## Folder layout

```
Network/
├── 00_WHO_OWN_WHAT.md          this file
├── P1_networkA_mlp.md          P1's brief
├── P2_networkB_lstm.md         P2's brief
├── P3_agent.md                 P3's brief
├── common/                     SHARED and FROZEN: read it, do not edit it
│   ├── env_setup.py            finds the environment, builds it, converts actions
│   ├── inputs.py               SelectInputs: which frames and sensors the network sees
│   ├── policies.py             adapters so every policy looks the same to the evaluator
│   ├── evaluate.py             THE evaluation protocol: seeds, metrics, results rows
│   ├── compare.py              one table across both networks
│   └── test_common.py
├── agent/                      P3 only
│   ├── config.py               every agent setting (PPO, loss weights, residual, logging)
│   ├── train.py                the main program: python -m agent.train
│   ├── actions.py              the residual action wrapper
│   ├── curve.py                the learning-curve logger
│   └── test_agent.py
├── networkA_mlp/               P1 only:  network.py   test_network.py
├── networkB_lstm/              P2 only:  network.py   test_network.py
├── results/                    one CSV per network, plus the cached reference episodes
└── runs/                       saved models and curves (git-ignored)
```

## How the pieces connect

`agent/train.py` is the only program you run. It loads a network module and needs each one to provide
the same seven names (the contract; `agent/test_agent.py` checks it):

| Name | Meaning |
|---|---|
| `NAME` | short name; used for the runs folder and `results/<NAME>.csv` |
| `CONFIG` | the network's own settings; must not repeat any key in `agent/config.py` |
| `ALGO`, `POLICY_NAME` | the Stable-Baselines3 class (PPO or RecurrentPPO) and its policy name |
| `POLICY` | the evaluation adapter class from `common/policies.py` |
| `wrappers(cfg)` | environment wrappers that choose the network's inputs |
| `policy_kwargs(cfg)` | the description of the network's layers |

Settings are split by owner. The network's file holds the layers and the inputs. `agent/config.py`
holds everything about training. Both are merged into one config per run, saved in the run's folder,
and any value can be changed for one run on the command line: `--set "key=value"`.

## The rules

1. **Edit only your own folder.** Your files cannot conflict with anyone else's. Use your own branch
   (`nn-p1`, `nn-p2`, `nn-p3`) and merge whenever you like.
2. **`common/` is frozen.** If you need something different, copy the function into your folder.
   The one exception is P3's day-1 fixes below.
3. **Report numbers only from `common/evaluate.py`.** The same seeds and metrics for everyone.
4. **The environment is frozen** at the tagged commit. If you find a bug, tell the group; everyone
   re-runs, and the cached reference file in `results/` is deleted.
5. **Rows in `results/<NAME>.csv` are only appended.** Re-evaluating a run (`--eval-only`) adds a new row.
6. **Quote list values** in `--set`: `--set "hidden=[64,64]"`, because some shells treat brackets specially.

## Setup (everyone, 10 minutes)

Run every command from inside the `Network_Agent/` folder. It finds `../Gym/stance_env` by itself. If your
folders are laid out differently, set `STANCE_GYM_PATH` to the folder that contains `stance_env/`.

```bash
pip install -r requirements.txt
python -m pytest -q                              # all tests; up to a minute the first time
python -m agent.train --network mlp --smoke      # tiny run of the whole pipeline
python -m agent.train --network lstm --smoke
```

If ROS is sourced in your shell, put `env -u PYTHONPATH` in front of each command.

## Day-1 checks (P3, about 30 minutes, nobody waits on these)

1. **Environment ids.** `python -m common.evaluate --show-info` must run. If it cannot find
   `StanceAnkle-v0`, or the constants `ANKLE_RANGE`, `K_RANGE`, `B_RANGE`, edit `TRAIN_ID`, `TEST_ID`
   or `_consts()` in `common/env_setup.py`.
2. **Toe key.** The same command prints `detected: {'toe': ...}`. If toe is `None`, pin the real key names
   in `INFO_KEYS` at the top of `common/evaluate.py`. Until then `toe_peak_bw` shows `nan`.
3. **Reference.** `python -m common.evaluate --reference` should give about **92.5% survival** and a
   toe peak of about **1.74 BW**. If it does not, the environment is not the frozen one; stop and check.
4. Commit the pinned keys and `results/reference_train_1000-1199.csv`, and tell the group.
5. **Ask the professor** (one message):
   - Is Stable-Baselines3's PPO acceptable for the network and the agent? (Lecture 6's notebook uses it.)
   - Does Report 2 need a hand-written loss and gradient step? The lecture lists them under the network.
   - Are the 30 Oct and 6 Nov deadlines separate?

   If he wants hand-written code, P3 adds a PPO implementation in `agent/` behind the same contract, and the
   networks do not change. Keep the Stable-Baselines3 results as the reference it must match.

## Evaluation protocol (what every row means)

200 episodes on seeds 1000-1199, deterministic policy, training ground unless the row says `test`.
Learning curves use seeds 2000-2049, so the table seeds are never seen in training.

| Column | Meaning | How to read it |
|---|---|---|
| `survival_pct`, `survival_ci` | episodes that reach step 100 without falling, with a 95% interval | **headline metric** |
| `falls_by_reason` | forward / back counts, or the environment's own reason | nearly all falls so far are forward |
| `toe_peak_bw` | mean over episodes of the episode's peak toe force, in body weights | **headline metric**: lower is softer |
| `mean_K`, `mean_theta_deg`, `mean_B` | average applied stiffness, angle and damping | shows what the policy learned |
| `mean_return`, `return_ci` | episode return | secondary; mostly a survival counter |
| `ceiling_pct` | share of episodes with return of 99 or more | how saturated the return is |
| `minus_reference` | paired return difference to the best fixed action on the same seeds | above 0 with the interval clear of 0 means better |

**The bar to beat** is the best fixed action (-20 degrees, K 500, B 10): about 92.5% survival and 1.74 BW toe
peak. About 7% of seeds had no surviving action on the 45-action grid in Phase 2b, so roughly 93% survival
is a practical ceiling. A network wins if it **matches survival and lowers toe force**, or improves
survival at the same toe force.

**Facts everyone should know:**
- Plain PPO (300k steps, one seed) reached 26% survival on the current reward.
- It drifted the target angle to +4 to +7 degrees, the wrong way, because a softer foot lowers toe force
  immediately while the fall penalty arrives later.
- The best fixed action sits at the edge of every range: angle -20 is the minimum, K 500 and B 10 are
  the maximums.

## How training settings reach the networks

How well a network learns depends on the training settings, which P3 is tuning. So:

- Until **22 Oct**, P1 and P2 train with the defaults in `agent/config.py`.
- **22 Oct:** P3 posts "config v1", the best settings so far.
- **24 Oct:** P3 freezes the final config in `agent/config.py` and tells the group.
- **24-26 Oct:** P1 and P2 run their final runs with it, so both networks are compared under identical training.

## Fair comparison, 26 Oct

Each network brings its best configuration, trained with the frozen agent config for **1M steps on 3 seeds**
(`--seed 0`, `--seed 1`, `--seed 2`) and evaluated with `--test-split`. Then:

```bash
python -m common.compare
python -m common.compare --split test
```

Report the mean and range over the three seeds, not the luckiest seed.

## Timeline

| Date | What happens |
|---|---|
| **16-17 Oct** | Setup, smoke runs, tests pass. P3 does the day-1 checks and asks the professor. |
| **17-21 Oct** | Baselines (300k steps), then each person's experiments (see your brief). |
| **22 Oct** | First full row in each `results/<NAME>.csv`. P3 posts config v1. |
| **23-25 Oct** | Remaining experiments. P3 freezes the final config on 24 Oct. Final runs start. |
| **26 Oct** | Comparison. Decide which network goes into the agent. |
| **27-28 Oct** | Write the Report 2 sections and assemble. |
| **29 Oct** | Buffer: check every number against `results/`. |
| **30 Oct** | Submit the network and Report 2. |
| **31 Oct - 6 Nov** | The agent: P3 leads; P1 and P2 help with tuning, benchmarking and the video. Submit 6 Nov. |

Load: about 15 hours each until 30 Oct. After that P3 carries the agent and P1 and P2 each give about 5 hours.

## Report 2: who writes what

Check this against the official brief; if it asks for something else, reshuffle.

| Section | Writer |
|---|---|
| Shared setup: inputs, outputs, the evaluation protocol | P1 |
| Network A section | P1 |
| Network B section | P2 |
| Comparison table and discussion (from `common.compare`) | P2 |
| Training setup and the loss (what the agent does), integration, figures, page limit, AI-use declaration | P3 |

Each network section follows the lecture's checklist: **inputs and preprocessing**, **data flow** (layer types,
widths, activations), **outputs** (size and activation), **losses and gradients**, **parameter count**
(`python -m agent.train --network mlp --summary` prints the layers and counts), the **learning curve**
(from `runs/<NAME>/<run>/curve.csv`), the **result row** and **one ablation**.
