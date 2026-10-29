"""
Demo of the two STANCE networks (no training). For each network found under trained_networks/ it
runs a forward/backward pass: one batch of 128 steps from the saved policy, the PPO loss and the
gradients. Then it evaluates both networks and the existing best fixed action on the same episodes, on the
train ground and on the held-out test ground, and prints a comparison table per ground.
"""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agent.actions import ResidualAction
from common.env_setup import make_env, to_norm_action
from common.evaluate import EVAL_SEEDS, REFERENCE, run_episodes, summarise
from common.policies import FixedPolicy

NETWORKS = {"mlp": "networkA_mlp.network", "lstm": "networkB_lstm.network"}
REQUIRED_FILES = ("model.zip", "vecnormalize.pkl", "config.json")
ROWS = [("survival_pct", "survival (%)"), ("toe_peak_bw", "toe peak (BW)"), ("mean_return", "mean return")]

def load_run(run_dir):
    """Load one saved run: its config, the network's own wrappers, the model and the VecNormalize."""
    saved = json.loads((run_dir / "config.json").read_text())
    config_load = saved["config"]
    net = importlib.import_module(NETWORKS[saved["network"]])
    wrappers = list(net.wrappers(config_load))
    if config_load["residual"]:
        wrappers.append(lambda e: ResidualAction(e, tuple(config_load["anchor"]), tuple(config_load["scale"])))
    model = net.ALGO.load(run_dir / "model.zip", device="cpu")
    vn = VecNormalize.load(str(run_dir / "vecnormalize.pkl"),
                           DummyVecEnv([lambda: make_env("train", wrappers)]))
    vn.training = False
    return net, config_load, wrappers, model, vn


# Forward/Backward Batch Collect
def forward_backward_batch(model, wrappers, vn, gamma, n_steps=128, seed=0):
    """Run the saved policy (random actions from its Gaussian) for n_steps on the train ground."""
    policy = model.policy
    policy.set_training_mode(False)
    recurrent = hasattr(policy, "lstm_actor")
    env = make_env("train", wrappers)
    low, high = env.action_space.low, env.action_space.high
    if recurrent:  # the LSTM carries a hidden state between steps; it starts at zero
        def zeros(lstm):
            z = torch.zeros(lstm.num_layers, 1, lstm.hidden_size)
            return z, z.clone()
        states = RNNStates(zeros(policy.lstm_actor), zeros(policy.lstm_critic or policy.lstm_actor))
        init_states = states
    obs_l, act_l, logp_l, val_l, rew_l, start_l, done_l = [], [], [], [], [], [], []
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        start = True
        for _ in range(n_steps):
            o = torch.as_tensor(vn.normalize_obs(obs)[None], dtype=torch.float32)
            if recurrent:
                a, v, lp, states = policy(o, states, torch.tensor([float(start)]))
            else:
                a, v, lp = policy(o)
            obs, r, term, trunc, _ = env.step(np.clip(a[0].numpy(), low, high))
            obs_l.append(o[0]); act_l.append(a[0]); logp_l.append(lp.item()); val_l.append(v.item())
            rew_l.append(float(vn.normalize_reward(np.array([r]))[0]))
            start_l.append(float(start)); done_l.append(term or trunc)
            start = term or trunc
            if start:
                obs, _ = env.reset()
    env.close()

    returns, g = np.zeros(n_steps), 0.0  # discounted return, restarted at each episode end
    for t in reversed(range(n_steps)):
        g = rew_l[t] + gamma * g * (not done_l[t])
        returns[t] = g
    adv = returns - np.array(val_l)
    batch = {"obs": torch.stack(obs_l), "actions": torch.stack(act_l),
             "old_logp": torch.tensor(logp_l), "starts": torch.tensor(start_l),
             "returns": torch.tensor(returns, dtype=torch.float32),
             "adv": torch.tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32)}
    if recurrent:
        batch["lstm_states"] = init_states
    return batch


def ppo_loss(policy, batch, config_load):
    """Forward pass through the policy and the PPO loss: clipped policy term + value term - entropy term."""
    if "lstm_states" in batch:
        values, logp, entropy = policy.evaluate_actions(batch["obs"], batch["actions"],
                                                        batch["lstm_states"], batch["starts"])
    else:
        values, logp, entropy = policy.evaluate_actions(batch["obs"], batch["actions"])
    ratio = torch.exp(logp - batch["old_logp"])
    c, adv = config_load["clip_range"], batch["adv"]
    policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - c, 1 + c) * adv).mean()
    value_loss = torch.nn.functional.mse_loss(values.flatten(), batch["returns"])
    loss = policy_loss + config_load["vf_coef"] * value_loss - config_load["ent_coef"] * entropy.mean()
    return loss, policy_loss.item(), value_loss.item(), entropy.mean().item()


def show_forward_backward(model, config_load, wrappers, vn, key):
    print("\nForward and backward pass - " + f"{key.upper()}:\n")
    policy = model.policy
    batch = forward_backward_batch(model, wrappers, vn, config_load["gamma"])
    print(f"input: {tuple(batch['obs'].shape)} normalised observations; "
          f"output: {tuple(batch['actions'].shape)} actions, one value per step")

    policy.set_training_mode(True)
    policy.zero_grad(set_to_none=True)
    loss, pl, vl, ent = ppo_loss(policy, batch, config_load) # forward
    print(f"loss = policy {pl:.4f} + {config_load['vf_coef']} * value {vl:.4f} - {config_load['ent_coef']} * entropy {ent:.4f}"
          f" = {loss.item():.4f}")
    loss.backward() # backward

    print(f"gradient norm per part of the network (after loss.backward()):")
    total = 0.0
    for name, child in policy.named_children():
        sq = sum(p.grad.norm().item() ** 2 for p in child.parameters() if p.grad is not None)
        total += sq
        print(f"  {name:28s} {sq ** 0.5:10.4e}")
    total = total ** 0.5
    clip = config_load["max_grad_norm"]
    print(f"total {total:.4f}; PPO clips it to max_grad_norm {clip}"
          + (f" (scale {clip / total:.3f})" if total > clip else " (no clipping needed)"))
    policy.zero_grad(set_to_none=True)   # nothing is trained or saved here
    policy.set_training_mode(False)


def print_table(columns, seeds, split):
    print(f"\n{len(seeds)} episodes per seed, seeds {seeds[0]}-{seeds[-1]}, {split} ground, deterministic")
    print(f"{'':16s}" + "".join(f"{name:>20s}" for name in columns))
    for k, label in ROWS:
        print(f"{label:16s}" + "".join(f"{c[k]:20.2f}" for c in columns.values()))
    if split == "test":
        print("note: toe peak is not informative on the test ground (set at touchdown, before the policy acts)")


def mean_rows(rows):
    """Average the per-seed summaries (every seed uses the same episodes)."""
    return {k: float(np.mean([r[k] for r in rows])) for k, _ in ROWS}

# Main

def main():
    p = argparse.ArgumentParser(description="Demo of the STANCE networks (no training).")
    p.add_argument("--episodes", type=int, default=200, help="evaluation episodes per ground and seed")
    p.add_argument("--submission", default="trained_networks", help="folder with the run folders")
    a = p.parse_args()

    # One list of run folders (seed0, seed1, ...) per network.
    runs = {}
    for d in sorted(Path(a.submission).glob("*/*/")):
        if all((d / f).exists() for f in REQUIRED_FILES):
            runs.setdefault(json.loads((d / "config.json").read_text())["network"], []).append(d)
    if not runs:
        print(f"No run folder with {', '.join(REQUIRED_FILES)} found under '{a.submission}/'. See README.md.")
        return

    seeds = EVAL_SEEDS[: a.episodes]
    loaded = {key: [load_run(d) for d in dirs] for key, dirs in runs.items()}

    # Forward and backward pass: shown for the first seed of each network only.
    for key, group in loaded.items():
        net, config_load, wrappers, model, vn = group[0]
        show_forward_backward(model, config_load, wrappers, vn, key)

    print("\nComparison: The Networks (mean over their seeds) and The Best Fixed Action (please give it some time)\n")
    for split in ("train", "test"):
        per_seed = {}                       # network name -> one summary per seed
        for group in loaded.values():
            for net, config_load, wrappers, model, vn in group:
                env = make_env(split, wrappers)
                per_seed.setdefault(net.NAME, []).append(summarise(run_episodes(net.POLICY(model, vn), env, seeds)))
                env.close()
        cols = {name: mean_rows(rows) for name, rows in per_seed.items()}
        env = make_env(split)
        cols["fixed action"] = summarise(run_episodes(FixedPolicy(to_norm_action(env, *REFERENCE)), env, seeds))
        env.close()
        print_table(cols, seeds, split)
        for name, rows in per_seed.items():
            print(f"  return per seed, {name}: " + " / ".join(f"{r['mean_return']:.2f}" for r in rows))
    print(f"\n(fixed action = {REFERENCE[0]:g} deg, K {REFERENCE[1]:g}, B {REFERENCE[2]:g})")

# Run main
if __name__ == "__main__":
    main()