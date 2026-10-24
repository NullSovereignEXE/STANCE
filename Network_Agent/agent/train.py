"""The learning agent: the main program that trains and evaluates a network. OWNER: P3.

In the lecture's structure the agent sits between the network and the environment. It
asks the environment for a new episode (`env.reset()`), asks the network for an action
given the observation, applies it (`env.step()`), records the experience, and every
`n_steps` steps uses it to update the network's weights. Here Stable-Baselines3's PPO
runs that loop inside `model.learn()`. This file builds the pieces around it, starts it,
and handles saving, logging and evaluation.

What this file does, in order
    1. `load_network` picks the network module (A or B) and checks it follows the contract.
    2. `merged_config` combines agent/config.py with the network's CONFIG, applies
       --smoke, then your --set overrides.
    3. `make_wrappers` builds the wrapper list: the network's input selection, the
       optional residual action, then the optional reweighted impact penalty. The same
       list is used for training and evaluation.
    4. `make_venv` builds parallel environments with observation/reward normalisation.
    5. `build_model` creates the PPO model around the network.
    6. `main` trains with a learning-curve logger, saves, then evaluates with the shared
       protocol in common/evaluate.py.

The network contract (what a file such as networkA_mlp/network.py must provide)
    NAME           short name; used for the runs folder and results/<NAME>.csv
    CONFIG         the network's own settings (must not repeat an agent/config.py key)
    ALGO           the SB3 class: PPO or sb3_contrib.RecurrentPPO
    POLICY_NAME    "MlpPolicy" or "MlpLstmPolicy"
    POLICY         the evaluation adapter class from common/policies.py
    wrappers(cfg)       list of environment wrappers (input selection)
    policy_kwargs(cfg)  keyword arguments describing the network layers

Commands
    python -m agent.train --network mlp --smoke           tiny run that checks the pipeline
    python -m agent.train --network lstm --run-name base  full run, seed 0
    python -m agent.train --network mlp --summary         print the layers and parameter counts
    python -m agent.train --eval-only runs/<NAME>/<run> --test-split
    add --set "key=value" ... to change any config value for one run

Uses: stable_baselines3 (PPO, VecNormalize, make_vec_env), sb3_contrib (RecurrentPPO),
and the shared code in common/.
"""
import argparse
import ast
import importlib
import json
import time
from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from agent.actions import ResidualAction
from agent.reward import ReweightedImpact
from agent.config import AGENT_CONFIG, SMOKE
from agent.curve import CurveCallback
from common.env_setup import make_env
from common.evaluate import evaluate_and_record

NETWORKS = {"mlp": "networkA_mlp.network", "lstm": "networkB_lstm.network"}
REQUIRED = ("NAME", "CONFIG", "ALGO", "POLICY_NAME", "POLICY", "wrappers", "policy_kwargs")
RUNS = Path(__file__).resolve().parent.parent / "runs"


def load_network(key):
    """Import the network module registered under `key` and check it has every required name."""
    if key not in NETWORKS:
        raise SystemExit(f"unknown network '{key}'; choose from {sorted(NETWORKS)}")
    module = importlib.import_module(NETWORKS[key])
    missing = [n for n in REQUIRED if not hasattr(module, n)]
    if missing:
        raise SystemExit(f"{NETWORKS[key]} is missing {missing}")
    return module


def merged_config(net, smoke=False, overrides=()):
    """Return the full config: agent settings, network settings, then smoke, then overrides.

    overrides: strings such as "gamma=0.995". Values are read as Python literals
    (numbers, True/False, lists); anything else stays a string. Unknown keys and keys
    present in both config files raise an error instead of being silently ignored.
    """
    clash = set(AGENT_CONFIG) & set(net.CONFIG)
    if clash:
        raise SystemExit(f"{net.NAME}.CONFIG repeats agent keys: {sorted(clash)}")
    cfg = {**AGENT_CONFIG, **net.CONFIG}
    if smoke:
        cfg.update(SMOKE)
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        key, value = item.split("=", 1)
        if key not in cfg:
            raise SystemExit(f"--set: unknown key '{key}'. Known keys: {sorted(cfg)}")
        try:
            cfg[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            cfg[key] = value
    return cfg


def make_wrappers(cfg, net):
    """The wrapper list used everywhere: input selection, residual action, reweighted impact."""
    wrappers = list(net.wrappers(cfg))
    if cfg["residual"]:
        wrappers.append(lambda e: ResidualAction(e, tuple(cfg["anchor"]), tuple(cfg["scale"])))
    if cfg.get("w_impact") is not None and cfg.get("lambda_bw") is not None:
        wrappers.append(lambda e: ReweightedImpact(e, cfg["w_impact"], cfg["lambda_bw"]))
    return wrappers


def make_venv(cfg, wrappers, seed):
    """Parallel training environments wrapped in VecNormalize (observation and reward scaling)."""
    venv = make_vec_env(lambda: make_env("train", wrappers), n_envs=cfg["n_envs"], seed=seed,
                        vec_env_cls=SubprocVecEnv if cfg["subproc"] else DummyVecEnv)
    return VecNormalize(venv, norm_obs=cfg["norm_obs"], norm_reward=cfg["norm_reward"],
                        clip_obs=10.0, gamma=cfg["gamma"])


def build_model(cfg, net, venv, seed):
    """Create the PPO model: the network's layers plus the agent's learning settings."""
    return net.ALGO(
        net.POLICY_NAME, venv, policy_kwargs=net.policy_kwargs(cfg),
        n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"], gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"], ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"], seed=seed, device="cpu", verbose=0)


def count_params(model):
    """Total number of trainable parameters in the model's network."""
    return sum(p.numel() for p in model.policy.parameters())


def print_architecture(model, venv, net):
    """Print the network's layers and parameter counts, the PyTorch equivalent of Keras summary().

    The "other" row is parameters that belong to no layer, such as the learned action spread.
    """
    print(f"network: {net.NAME}")
    print(f"observation: {venv.observation_space.shape}   action: {venv.action_space.shape}")
    print(model.policy)
    total, listed = count_params(model), 0
    print("\nparameters by part")
    for name, child in model.policy.named_children():
        n = sum(p.numel() for p in child.parameters())
        listed += n
        print(f"  {name:28s} {n:>10,}")
    print(f"  {'other (e.g. action spread)':28s} {total - listed:>10,}")
    print(f"  {'total':28s} {total:>10,}")


def main():
    """Parse the command line, then train and/or evaluate."""
    p = argparse.ArgumentParser(description="Train and evaluate a network with the learning agent.")
    p.add_argument("--network", choices=sorted(NETWORKS), help="which network to train")
    p.add_argument("--run-name", default="base", help="label for this run")
    p.add_argument("--seed", type=int, default=0, help="seed for the environments and the network")
    p.add_argument("--smoke", action="store_true", help="tiny run to check everything works")
    p.add_argument("--summary", action="store_true", help="print the architecture and exit")
    p.add_argument("--eval-only", metavar="RUN_DIR", help="skip training; evaluate a saved run")
    p.add_argument("--test-split", action="store_true", help="also evaluate on the held-out ground")
    p.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override config values")
    a = p.parse_args()

    if a.eval_only:
        run_dir = Path(a.eval_only)
        saved = json.loads((run_dir / "config.json").read_text())
        cfg, seed = saved["config"], saved["seed"]
        net = load_network(saved["network"])
        wrappers = make_wrappers(cfg, net)
        model = net.ALGO.load(run_dir / "model.zip", device="cpu")
        vn = VecNormalize.load(str(run_dir / "vecnormalize.pkl"),
                               DummyVecEnv([lambda: make_env("train", wrappers)]))
        vn.training = False
        run_name = run_dir.name
    else:
        if not a.network:
            p.error("--network is required unless --eval-only is given")
        net = load_network(a.network)
        cfg = merged_config(net, a.smoke, a.set)
        wrappers = make_wrappers(cfg, net)
        seed = a.seed
        if a.summary:
            venv = make_venv(dict(cfg, n_envs=1, subproc=False), wrappers, seed)
            print_architecture(build_model(cfg, net, venv, seed), venv, net)
            return
        run_name = ("smoke_" if a.smoke else "") + f"{a.run_name}_s{seed}"
        run_dir = RUNS / net.NAME / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "curve.csv").unlink(missing_ok=True)
        saved = {"network": a.network, "seed": seed, "config": cfg}
        (run_dir / "config.json").write_text(json.dumps(saved, indent=2))
        venv = make_venv(cfg, wrappers, seed)
        model = build_model(cfg, net, venv, seed)
        print(f"{net.NAME}/{run_name}: {count_params(model):,} parameters, "
              f"{cfg['total_steps']:,} steps")
        t0 = time.time()
        model.learn(cfg["total_steps"],
                    callback=CurveCallback(cfg, wrappers, net.POLICY, run_dir / "curve.csv"))
        print(f"trained in {(time.time() - t0) / 60:.1f} min")
        model.save(run_dir / "model.zip")
        venv.save(str(run_dir / "vecnormalize.pkl"))
        vn = venv

    extra = {"seed": seed, "steps": cfg["total_steps"], "params": count_params(model)}
    seeds = list(range(1000, 1000 + cfg["eval_episodes"]))
    for split in (["train", "test"] if a.test_split else ["train"]):
        env = make_env(split, wrappers)
        evaluate_and_record(net.NAME, run_name, net.POLICY(model, vn), env, split,
                            seeds=seeds, run_dir=run_dir, extra=extra)
        env.close()


if __name__ == "__main__":
    main()
