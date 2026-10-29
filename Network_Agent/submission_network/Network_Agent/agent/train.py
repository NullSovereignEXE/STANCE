"""
Train one network (MLP or LSTM) with PPO and save it, so demo.py can load it.
The run is saved to runs/<network>/seed<N>/ as model.zip, vecnormalize.pkl and config.json.
"""
import argparse
import importlib
import json
import time
from pathlib import Path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agent.actions import ResidualAction
from common.env_setup import make_env
from common.evaluate import EVAL_SEEDS, run_episodes, summarise

NETWORKS = {"mlp": "networkA_mlp.network", "lstm": "networkB_lstm.network"}
RUNS = Path(__file__).resolve().parent.parent / "runs"

# PPO settings (config v1), the same for both networks.
TRAIN = {
    "total_steps": 300_000,   # Environment steps to train for
    "n_envs": 8,              # Environments stepped in parallel
    "n_steps": 256,           # Steps per environment per update (8 * 256 = 2048 samples per update)
    "batch_size": 512,        # Minibatch size inside an update
    "n_epochs": 10,           # Passes over each batch of samples
    "learning_rate": 3e-4,    # Adam step size
    "gamma": 0.99,            # Discount factor
    "gae_lambda": 0.95,       # Smoothing of the advantage estimate
    "clip_range": 0.2,        # PPO limit on how far one update may change the policy
    "ent_coef": 0.0,          # Weight of the entropy bonus in the loss
    "vf_coef": 0.5,           # Weight of the value error in the loss
    "max_grad_norm": 0.5,     # Gradients are clipped to this length before each Adam step
    "residual": True,         # The policy outputs an offset from `anchor` instead of the action itself
    "anchor": [-14.0, 250.6, 6.38],  # The action the offset starts from: degrees, N.m/rad, N.m.s/rad
    "scale": [0.3, 0.3, 0.3],        # Largest offset per action dimension, in normalised units
}


def make_wrappers(cfg, net):
    """The network's input selection, then the residual action."""
    wrappers = list(net.wrappers(cfg))
    if cfg["residual"]:
        wrappers.append(lambda e: ResidualAction(e, tuple(cfg["anchor"]), tuple(cfg["scale"])))
    return wrappers


def make_venv(cfg, wrappers, seed):
    """n_envs training environments (train ground), with observation and reward normalisation."""
    def env_fn(rank):
        def _init():
            env = make_env("train", wrappers)
            env.action_space.seed(seed + rank)
            return Monitor(env)
        return _init

    venv = DummyVecEnv([env_fn(rank) for rank in range(cfg["n_envs"])])
    venv.seed(seed)
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg["gamma"])


def main():
    p = argparse.ArgumentParser(description="Train a network with PPO.")
    p.add_argument("--network", required=True, choices=sorted(NETWORKS), help="which network to train")
    p.add_argument("--seed", type=int, default=0, help="seed for the environments and the network")
    p.add_argument("--steps", type=int, help="training steps")
    a = p.parse_args()

    # Dynamically loads the selected network module, merges the training configurations builds environment wrappers using the final configuration.
    net = importlib.import_module(NETWORKS[a.network])
    assert not set(TRAIN) & set(net.CONFIG), "network CONFIG repeats a PPO setting"
    cfg = {**TRAIN, **net.CONFIG}
    if a.steps:
        cfg["total_steps"] = a.steps
    wrappers = make_wrappers(cfg, net)

    run_dir = RUNS / net.NAME / f"seed{a.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"network": a.network, "seed": a.seed, "config": cfg}, indent=2))

    venv = make_venv(cfg, wrappers, a.seed)
    model = net.ALGO(
        net.POLICY_NAME, venv, policy_kwargs=net.policy_kwargs(cfg),
        n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"], gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"], ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"], seed=a.seed, device="cpu", verbose=0)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"{net.NAME}: {n_params:,} parameters, {cfg['total_steps']:,} steps")

    t0 = time.time()
    model.learn(cfg["total_steps"])
    print(f"trained in {(time.time() - t0) / 60:.1f} min") # how long it takes to train the model
    model.save(run_dir / "model.zip")
    venv.save(str(run_dir / "vecnormalize.pkl"))

    # Final check on the train ground, 200 episodes (seeds 1000-1199).
    env = make_env("train", wrappers)
    s = summarise(run_episodes(net.POLICY(model, venv), env, EVAL_SEEDS))
    env.close()
    print(f"train ground, {len(EVAL_SEEDS)} episodes: survival {s['survival_pct']:.1f}%, "
          f"toe peak {s['toe_peak_bw']:.2f} BW, mean return {s['mean_return']:.2f}")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()