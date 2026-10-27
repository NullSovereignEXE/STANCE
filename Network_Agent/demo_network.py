"""Demo of the two STANCE networks: load the saved weights, show the design, evaluate.

Nothing is trained. For each run folder under submission/ (networkA_mlp = Network A, the
MLP by P1; networkB_lstm = Network B, the LSTM by P2) it prints, in order:
    1. Inputs and preprocessing   observation shape, frames and sensors, normalisation statistics
    2. Data flow                  the layers (PyTorch print) and parameters per part
    3. Outputs                    what the three action values mean and their physical ranges
    4. One real observation       one forward pass: action and value estimate (LSTM: memory check)
    5. Weights                    shape, mean, std, min, max per layer; histograms to demo_output/
    6. Learning curve             survival and toe peak against training steps, to demo_output/
    7. Evaluation                 the network next to the best fixed action on the same seeds
    8. Forward and backward pass  a short stochastic rollout, the PPO loss, its gradients per
                                  tensor, and 10 Adam steps on a copy (saved weights untouched)
With two networks it ends with a side-by-side table.

A run folder holds four files written by agent/train.py: model.zip, vecnormalize.pkl,
config.json and curve.csv. The model is rebuilt from config.json exactly as
`python -m agent.train --eval-only` does it: the network's own wrappers, then the residual
action if the config says so, then the network's ALGO.load and VecNormalize.load.

Commands (run from the Network_Agent folder, with STANCE_GYM_PATH set; see README_network.md)
    python demo_network.py                              every network under submission/
    python demo_network.py --networks mlp --episodes 200
    python demo_network.py --split test --no-plots

Uses: common.env_setup (make_env, to_physical, to_norm_action, SENSOR_NAMES), common.evaluate
(EVAL_SEEDS, REFERENCE, run_episodes, summarise), common.policies.FixedPolicy,
agent.actions.ResidualAction, sb3_contrib's RNNStates, and the network modules
networkA_mlp.network and networkB_lstm.network. Plots need matplotlib.
"""
import argparse
import copy
import csv
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agent.actions import ResidualAction
from common.env_setup import SENSOR_NAMES, make_env, to_norm_action, to_physical
from common.evaluate import EVAL_SEEDS, REFERENCE, run_episodes, summarise
from common.policies import FixedPolicy

NETWORKS = {"mlp": "networkA_mlp.network", "lstm": "networkB_lstm.network"}
LABELS = {"mlp": "Network A (MLP, P1)", "lstm": "Network B (LSTM, P2)"}
REQUIRED_FILES = ("model.zip", "vecnormalize.pkl", "config.json")
OUT = Path("demo_output")
ROWS = [("survival_pct", "survival (%)"), ("toe_peak_bw", "toe peak (BW)"),
        ("mean_theta_deg", "mean angle (deg)"), ("mean_K", "mean K (N.m/rad)"),
        ("mean_B", "mean B (N.m.s/rad)"), ("mean_return", "mean return")]


def header(text):
    print(f"\n--- {text} " + "-" * max(0, 70 - len(text)))


def fmt(values, digits=4):
    """A short printable list of numbers."""
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in np.ravel(values)) + "]"


# ---------------------------------------------------------------- loading
def find_runs(submission, wanted):
    """Return {key: folder} for the run folders under `submission` that hold the required files.

    The key ("mlp" or "lstm") is read from each folder's config.json. `wanted` limits the keys.
    """
    runs = {}
    for d in sorted(Path(submission).glob("*/")):
        if all((d / f).exists() for f in REQUIRED_FILES):
            key = json.loads((d / "config.json").read_text())["network"]
            if key in NETWORKS and (not wanted or key in wanted):
                runs[key] = d
    return runs


def load_run(run_dir):
    """Load one run folder the same way `agent.train --eval-only` does.

    1. config.json gives the network key and the full config (agent and network settings).
    2. The wrappers are the network's input selection, then ResidualAction if cfg["residual"].
    3. The model is loaded with the network's own ALGO; VecNormalize holds the observation
       statistics the network was trained with and is frozen (training=False).
    Returns (net, cfg, wrappers, model, vecnormalize).
    """
    saved = json.loads((run_dir / "config.json").read_text())
    cfg = saved["config"]
    net = importlib.import_module(NETWORKS[saved["network"]])
    wrappers = list(net.wrappers(cfg))
    if cfg["residual"]:
        wrappers.append(lambda e: ResidualAction(e, tuple(cfg["anchor"]), tuple(cfg["scale"])))
    model = net.ALGO.load(run_dir / "model.zip", device="cpu")
    vn = VecNormalize.load(str(run_dir / "vecnormalize.pkl"),
                           DummyVecEnv([lambda: make_env("train", wrappers)]))
    vn.training = False
    return net, cfg, wrappers, model, vn


# ---------------------------------------------------------------- sections 1-3
def show_inputs(cfg, env, vn):
    """1. What the network sees: the cut observation and the saved normalisation statistics."""
    header("1. Inputs and preprocessing")
    drop = cfg.get("drop_sensors", [])
    kept = [n for i, n in enumerate(SENSOR_NAMES) if i not in drop]
    print(f"observation shape after the network's wrappers: {env.observation_space.shape}")
    print(f"frames: newest {cfg['frames']} of 10   sensors kept: {len(kept)} of {len(SENSOR_NAMES)}"
          f"   dropped: {drop or 'none'}")
    print("layout: frames oldest first, each frame = the kept sensors in this order:")
    print("  " + ", ".join(kept))
    rms = vn.obs_rms
    print(f"normalisation (VecNormalize, running statistics from training): count = {rms.count:,.0f}")
    print("  newest frame:   sensor                 mean          variance")
    for name, m, v in zip(kept, rms.mean[-len(kept):], rms.var[-len(kept):]):
        print(f"                  {name:20s} {m:12.4f} {v:14.4f}")
    print("  each value is fed in as (x - mean) / sqrt(variance), clipped to [-10, 10]")


def show_layers(model):
    """2. The layers, widths and activations, and the parameters in each part of the policy."""
    header("2. Data flow (layers and parameters)")
    print(model.policy)
    total = sum(p.numel() for p in model.policy.parameters())
    listed = 0
    print("\nparameters by part")
    for name, child in model.policy.named_children():
        n = sum(p.numel() for p in child.parameters())
        listed += n
        print(f"  {name:28s} {n:>10,}")
    print(f"  {'other (action spread)':28s} {total - listed:>10,}")
    print(f"  {'total':28s} {total:>10,}")


def show_outputs(cfg, env):
    """3. The three action values, their physical ranges, and the residual mapping."""
    header("3. Outputs")
    lo, hi = to_physical(env, [-1, -1, -1]), to_physical(env, [1, 1, 1])
    print("actor: three numbers in [-1, 1], used as the means of a Gaussian (deterministic here)")
    print(f"  target ankle angle theta_d : {lo[0]:7.1f} to {hi[0]:7.1f} deg      (linear)")
    print(f"  ankle stiffness K          : {lo[1]:7.1f} to {hi[1]:7.1f} N.m/rad  (log-spaced)")
    print(f"  ankle damping B            : {lo[2]:7.2f} to {hi[2]:7.2f} N.m.s/rad (log-spaced)")
    print("critic: one number, the estimated value of the state (used only in training)")
    if cfg["residual"]:
        anchor = to_norm_action(env, *cfg["anchor"])
        scale = np.asarray(cfg["scale"])
        a_lo = to_physical(env, np.clip(anchor - scale, -1, 1))
        a_hi = to_physical(env, np.clip(anchor + scale, -1, 1))
        print(f"residual action: env action = clip(anchor + scale * output, -1, 1)")
        print(f"  anchor {cfg['anchor']} (normalised {fmt(anchor, 3)}), scale {cfg['scale']}")
        print(f"  reachable: theta {a_lo[0]:.1f} to {a_hi[0]:.1f} deg, K {a_lo[1]:.1f} to {a_hi[1]:.1f},"
              f" B {a_lo[2]:.2f} to {a_hi[2]:.2f}")


# ---------------------------------------------------------------- section 4
def residual_of(env):
    """Return the ResidualAction wrapper inside `env`, or None."""
    while hasattr(env, "env"):
        if isinstance(env, ResidualAction):
            return env
        env = env.env
    return None


def value_of(model, obs_norm, recurrent):
    """The critic's value estimate for one normalised observation (LSTM: fresh critic state)."""
    obs_t = model.policy.obs_to_tensor(obs_norm)[0]
    with torch.no_grad():
        if not recurrent:
            return model.policy.predict_values(obs_t).item()
        lstm = model.policy.lstm_critic
        zeros = torch.zeros(lstm.num_layers, 1, lstm.hidden_size)
        return model.policy.predict_values(obs_t, (zeros, zeros), torch.ones(1)).item()


def show_one_step(net, cfg, env, model, vn):
    """4. Reset the environment, run the network once, print the action and the value."""
    header("4. One real observation (seed 1000, first step)")
    obs, _ = env.reset(seed=1000)
    policy = net.POLICY(model, vn)
    policy.reset()
    out = np.asarray(policy.act(obs), dtype=np.float32)
    res = residual_of(env)
    env_action = res.action(out) if res is not None else out
    th, K, B = to_physical(env, env_action)
    recurrent = hasattr(model.policy, "lstm_actor")
    n_kept = len(SENSOR_NAMES) - len(cfg.get("drop_sensors", []))
    print(f"newest frame of the observation : {fmt(obs[-n_kept:])}")
    print(f"network output (normalised)     : {fmt(out)}")
    if res is not None:
        print(f"env action after residual       : {fmt(env_action)}")
    print(f"physical action                 : theta_d {th:.2f} deg, K {K:.1f} N.m/rad, B {B:.2f} N.m.s/rad")
    print(f"value estimate                  : {value_of(model, vn.normalize_obs(obs), recurrent):.3f}")
    if recurrent:
        # Same observation fed five times: a changing action shows the LSTM state is used.
        policy.reset()
        acts = [np.asarray(policy.act(obs)) for _ in range(5)]
        print("memory check: the same observation fed 5 times in a row")
        for i in (0, 4):
            print(f"  step {i + 1}: output {fmt(acts[i])}")
        print(f"  largest change step 1 -> 5: {np.abs(acts[4] - acts[0]).max():.4f}"
              f" ({'the action depends on memory' if np.abs(acts[4] - acts[0]).max() > 1e-6 else 'no change'})")


# ---------------------------------------------------------------- sections 5-6
def show_weights(model, key, plots):
    """5. Weight statistics per parameter tensor; histograms of the weight matrices."""
    header("5. Weights")
    params = list(model.policy.named_parameters())
    print(f"  {'parameter':42s} {'shape':>14s} {'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s}")
    for name, p in params:
        w = p.detach().numpy().ravel()
        print(f"  {name:42s} {str(tuple(p.shape)):>14s} {w.mean():9.4f} {w.std():9.4f}"
              f" {w.min():9.4f} {w.max():9.4f}")
    if plots is None:
        return
    mats = [(n, p.detach().numpy().ravel()) for n, p in params if p.dim() >= 2]
    cols = 4
    rows = -(-len(mats) // cols)
    fig, axes = plots.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
    for ax, (name, w) in zip(axes.flat, mats):
        ax.hist(w, bins=60)
        ax.set_title(name, fontsize=8)
    for ax in list(axes.flat)[len(mats):]:
        ax.axis("off")
    fig.suptitle(f"{LABELS[key]}: weight histograms")
    fig.tight_layout()
    path = OUT / f"{key}_weights.png"
    fig.savefig(path, dpi=110)
    plots.close(fig)
    print(f"saved {path}")


def show_curve(run_dir, key, plots):
    """6. Survival and toe peak against training steps, from curve.csv (seeds 2000-2049)."""
    header("6. Learning curve")
    path = run_dir / "curve.csv"
    if not path.exists():
        print("no curve.csv in the run folder; skipped")
        return
    rows = list(csv.DictReader(open(path)))
    print(f"{len(rows)} points; last: {rows[-1]['timesteps']} steps, survival {rows[-1]['survival_pct']}%,"
          f" toe peak {rows[-1]['toe_peak_bw']} BW")
    if plots is None:
        return
    t = [float(r["timesteps"]) / 1000 for r in rows]
    fig, (a1, a2) = plots.subplots(1, 2, figsize=(11, 4))
    a1.plot(t, [float(r["survival_pct"]) for r in rows], marker="o")
    a1.set(xlabel="training steps (thousands)", ylabel="survival (%)", ylim=(0, 100), title="Survival")
    a2.plot(t, [float(r["toe_peak_bw"]) for r in rows], marker="o", color="tab:red")
    a2.set(xlabel="training steps (thousands)", ylabel="toe peak (body weights)", title="Toe peak force")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.suptitle(f"{LABELS[key]}: learning curve (seeds 2000-2049)")
    fig.tight_layout()
    out = OUT / f"{key}_curve.png"
    fig.savefig(out, dpi=110)
    plots.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------- section 7
def evaluate(net, wrappers, model, vn, seeds, split):
    """7. The network (its own adapter, net.POLICY) on `seeds` and `split`; returns the summary."""
    env = make_env(split, wrappers)
    s = summarise(run_episodes(net.POLICY(model, vn), env, seeds))
    env.close()
    return s


def reference(seeds, split):
    """The best fixed action (common.evaluate.REFERENCE) on the same seeds; nothing is cached."""
    env = make_env(split)
    s = summarise(run_episodes(FixedPolicy(to_norm_action(env, *REFERENCE)), env, seeds))
    env.close()
    return s


def table(columns, seeds, split):
    """Print one row per metric and one column per policy."""
    print(f"{len(seeds)} episodes, seeds {seeds[0]}-{seeds[-1]}, {split} ground, deterministic")
    print(f"{'':20s}" + "".join(f"{name:>22s}" for name in columns))
    for k, label in ROWS:
        print(f"{label:20s}" + "".join(f"{c[k]:22.2f}" for c in columns.values()))
    if split == "test":
        print("note: toe peak is not informative on the test ground; it is set at touchdown,"
              " before the policy can act")


# ---------------------------------------------------------------- section 8
ROLLOUT_STEPS = 128
ROLLOUT_SEED = 0
ADAM_STEPS = 10


def checksum(policy):
    """sha256 of every parameter and buffer of the policy, to show the weights did not change."""
    h = hashlib.sha256()
    for name, t in policy.state_dict().items():
        h.update(name.encode())
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def rollout(model, cfg, wrappers, vn, n_steps=ROLLOUT_STEPS, seed=ROLLOUT_SEED):
    """Sample `n_steps` from the saved policy (stochastic) on the train ground; return the batch.

    Observations are normalised with the saved VecNormalize and rewards scaled by it, as in
    training. Actions are the policy's raw Gaussian samples; the environment gets them clipped
    to [-1, 1], as SB3 does. A time-limit end bootstraps with the critic's value of the last
    observation, as SB3 does. Advantages are GAE(gamma, gae_lambda), normalised; returns are
    the unnormalised advantages plus the values. For the LSTM the batch also holds the initial
    (zero) LSTM states and the episode starts that evaluate_actions needs to replay the memory.
    """
    policy = model.policy
    policy.set_training_mode(False)
    recurrent = hasattr(policy, "lstm_actor")
    env = make_env("train", wrappers)
    low, high = env.action_space.low, env.action_space.high
    if recurrent:
        def zeros(lstm):
            z = torch.zeros(lstm.num_layers, 1, lstm.hidden_size)
            return z, z.clone()
        states = RNNStates(zeros(policy.lstm_actor), zeros(policy.lstm_critic or policy.lstm_actor))
        init = states

    def value(o, start):
        if not recurrent:
            return policy.predict_values(o).item()
        return policy.predict_values(o, states.vf, torch.tensor([float(start)])).item()

    buf = {k: [] for k in ("obs", "actions", "logp", "values", "rewards", "starts", "dones")}
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
            r = float(vn.normalize_reward(np.array([r]))[0])
            if trunc and not term:
                r += cfg["gamma"] * value(torch.as_tensor(vn.normalize_obs(obs)[None], dtype=torch.float32), False)
            for k, x in zip(buf, (o[0], a[0], lp.item(), v.item(), r, float(start), float(term or trunc))):
                buf[k].append(x)
            start = term or trunc
            if start:
                obs, _ = env.reset()
        last = value(torch.as_tensor(vn.normalize_obs(obs)[None], dtype=torch.float32), start)
    env.close()

    values, rewards, dones = (np.array(buf[k]) for k in ("values", "rewards", "dones"))
    adv, gae = np.zeros(n_steps), 0.0
    for t in reversed(range(n_steps)):
        next_value = last if t == n_steps - 1 else values[t + 1]
        delta = rewards[t] + cfg["gamma"] * next_value * (1 - dones[t]) - values[t]
        gae = delta + cfg["gamma"] * cfg["gae_lambda"] * (1 - dones[t]) * gae
        adv[t] = gae
    batch = {
        "obs": torch.stack(buf["obs"]), "actions": torch.stack(buf["actions"]),
        "old_logp": torch.tensor(buf["logp"]), "starts": torch.tensor(buf["starts"]),
        "returns": torch.tensor(adv + values, dtype=torch.float32),
        "adv": torch.tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32),
        "episodes": int(sum(buf["starts"])),
    }
    if recurrent:
        batch["lstm_states"] = init
    return batch


def ppo_loss(policy, batch, cfg):
    """PPO loss on `batch` (forward pass through evaluate_actions); returns (loss, parts)."""
    if "lstm_states" in batch:
        values, logp, entropy = policy.evaluate_actions(batch["obs"], batch["actions"],
                                                        batch["lstm_states"], batch["starts"])
    else:
        values, logp, entropy = policy.evaluate_actions(batch["obs"], batch["actions"])
    ratio = torch.exp(logp - batch["old_logp"])
    c, adv = cfg["clip_range"], batch["adv"]
    policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - c, 1 + c) * adv).mean()
    value_loss = torch.nn.functional.mse_loss(values.flatten(), batch["returns"])
    entropy = entropy.mean()
    loss = policy_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * entropy
    return loss, {"policy": policy_loss.item(), "value": value_loss.item(), "entropy": entropy.item(),
                  "ratio_dev": (ratio - 1).abs().max().item()}


def gradients(policy, batch, cfg):
    """One forward pass in training mode and loss.backward(); returns (loss, parts, grads).

    grads: (name, shape, gradient or None) per trainable tensor. The gradients are cleared
    from the policy afterwards; no weight is changed.
    """
    policy.set_training_mode(True)
    policy.zero_grad(set_to_none=True)
    loss, parts = ppo_loss(policy, batch, cfg)
    loss.backward()
    grads = [(n, tuple(p.shape), None if p.grad is None else p.grad.detach().clone())
             for n, p in policy.named_parameters() if p.requires_grad]
    policy.zero_grad(set_to_none=True)
    policy.set_training_mode(False)
    return loss.item(), parts, grads


def adam_steps(clone, batch, cfg, n=ADAM_STEPS):
    """Take `n` Adam steps on `clone`, a deepcopy of the policy, on the same batch; return (before, after).

    Make the copy before any backward pass: afterwards the policy holds the last action
    distribution, whose tensors are part of the autograd graph and cannot be deep-copied. The optimiser is the one the policy was trained with (class and kwargs, e.g. eps), at the
    config's learning_rate, with the gradient clipped to max_grad_norm as in PPO.
    """
    clone.set_training_mode(True)
    opt = clone.optimizer_class(clone.parameters(), lr=cfg["learning_rate"], **clone.optimizer_kwargs)
    with torch.no_grad():
        before = ppo_loss(clone, batch, cfg)[0].item()
    for _ in range(n):
        opt.zero_grad()
        ppo_loss(clone, batch, cfg)[0].backward()
        torch.nn.utils.clip_grad_norm_(clone.parameters(), cfg["max_grad_norm"])
        opt.step()
    with torch.no_grad():
        after = ppo_loss(clone, batch, cfg)[0].item()
    return before, after


def flag(g):
    """Why a gradient is unusable, or None if it is finite and non-zero."""
    if g is None:
        return "missing: the tensor is not on the path from the loss (not used by evaluate_actions)"
    if not torch.isfinite(g).all():
        return "non-finite: an overflow or a log/exp of an extreme value in the forward pass"
    if not g.abs().sum() > 0:
        return "zero: the tensor is used but every contribution cancels (e.g. saturated tanh)"
    return None


def show_forward_backward(model, cfg, wrappers, vn):
    """8. Rollout, PPO loss, gradients per tensor, then Adam steps on a copy."""
    header("8. Forward and backward pass")
    policy = model.policy
    before_sum = checksum(policy)
    batch = rollout(model, cfg, wrappers, vn)
    clone = copy.deepcopy(policy)
    print(f"rollout: {ROLLOUT_STEPS} steps, stochastic actions, seed {ROLLOUT_SEED}, train ground,"
          f" {batch['episodes']} episode(s) started")
    print(f"GAE: gamma {cfg['gamma']}, lambda {cfg['gae_lambda']}, rewards scaled by the saved"
          f" VecNormalize; advantages normalised to mean 0, std 1")
    loss, parts, grads = gradients(policy, batch, cfg)
    print(f"forward (training mode, evaluate_actions): max |ratio - 1| = {parts['ratio_dev']:.2e}"
          " (the rollout's log-probs are reproduced)")
    print(f"loss = policy {parts['policy']:.4f} + {cfg['vf_coef']} * value {parts['value']:.4f}"
          f" - {cfg['ent_coef']} * entropy {parts['entropy']:.4f} = {loss:.4f}")
    print("  (the policy term is 0 before any update: the ratio is 1 and the advantages have mean 0;"
          " its gradient is not)")
    print(f"gradients after loss.backward():\n  {'parameter':42s} {'shape':>14s} {'grad norm':>11s}")
    for name, shape, g in grads:
        print(f"  {name:42s} {str(shape):>14s} {'-' if g is None else f'{g.norm().item():11.4e}':>11s}")
    total = torch.norm(torch.stack([g.norm() for _, _, g in grads if g is not None])).item()
    clip = cfg["max_grad_norm"]
    print(f"total gradient norm {total:.4f}; max_grad_norm {clip}:"
          + (f" PPO would scale it by {clip / total:.3f}" if total > clip else " no clipping needed"))
    flagged = [(n, flag(g)) for n, _, g in grads if flag(g)]
    for name, why in flagged:
        print(f"  FLAG {name}: {why}")
    if not flagged:
        print("flags: none; every trainable tensor has a finite, non-zero gradient")
    before, after = adam_steps(clone, batch, cfg)
    print(f"{ADAM_STEPS} Adam steps on a copy (lr {cfg['learning_rate']}, clip {clip}), same batch:"
          f" loss {before:.4f} -> {after:.4f}")
    after_sum = checksum(policy)
    print(f"saved weights unchanged: {'yes' if after_sum == before_sum else 'NO'}"
          f" (sha256 {before_sum[:16]} before, {after_sum[:16]} after)")


# ---------------------------------------------------------------- main
def main():
    """Parse the arguments, then run the eight sections for each network found."""
    p = argparse.ArgumentParser(description="Demo of the STANCE networks (no training).")
    p.add_argument("--networks", nargs="+", choices=sorted(NETWORKS), help="default: all found")
    p.add_argument("--episodes", type=int, default=20, help="evaluation episodes, seeds 1000 and up")
    p.add_argument("--split", choices=["train", "test"], default="train", help="ground to evaluate on")
    p.add_argument("--no-plots", action="store_true", help="do not write demo_output/*.png")
    p.add_argument("--submission", default="submission", help="folder holding the run folders")
    a = p.parse_args()

    runs = find_runs(a.submission, a.networks)
    if not runs:
        print(f"No usable run folder found under '{a.submission}/'.\n"
              f"Put each network's files in {a.submission}/networkA_mlp/ or {a.submission}/networkB_lstm/:\n"
              f"  {', '.join(REQUIRED_FILES)} (and curve.csv for the learning curve),\n"
              f"copied from runs/<network>/<run>/. See README_network.md.")
        return

    plots = None
    if not a.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plots
            OUT.mkdir(exist_ok=True)
        except ImportError:
            print("matplotlib is not installed; plots are skipped")

    seeds = EVAL_SEEDS[: a.episodes]
    results = {}
    for key, run_dir in runs.items():
        print("\n" + "=" * 78 + f"\n{LABELS[key]}   run folder: {run_dir}\n" + "=" * 78)
        net, cfg, wrappers, model, vn = load_run(run_dir)
        env = make_env("train", wrappers)
        show_inputs(cfg, env, vn)
        show_layers(model)
        show_outputs(cfg, env)
        show_one_step(net, cfg, env, model, vn)
        env.close()
        show_weights(model, key, plots)
        show_curve(run_dir, key, plots)
        header("7. Evaluation")
        results[net.NAME] = evaluate(net, wrappers, model, vn, seeds, a.split)
        ref = reference(seeds, a.split)
        table({net.NAME: results[net.NAME], "fixed action": ref}, seeds, a.split)
        show_forward_backward(model, cfg, wrappers, vn)

    if len(results) > 1:
        print("\n" + "=" * 78 + "\nSide by side\n" + "=" * 78)
        table({**results, "fixed action": ref}, seeds, a.split)
    print(f"\n(fixed action = {REFERENCE[0]:g} deg, K {REFERENCE[1]:g}, B {REFERENCE[2]:g})")


if __name__ == "__main__":
    main()
