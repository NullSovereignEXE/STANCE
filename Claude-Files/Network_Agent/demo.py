"""Demo of the learning agent: load or train a network, then compare it with the best fixed action.

What this file does, in order
    1. Parses the command line (see the examples below).
    2. Gets a model: loads a saved run folder (--model) or trains one from scratch (--train)
       into a temporary folder. Both reuse agent/train.py, so the model is built exactly
       as in the experiments.
    3. Prints the network's layers and parameter count.
    4. Plays --episodes evaluation episodes (seeds 1000 and up) with the trained policy and
       with the best fixed action (common.evaluate.REFERENCE), and prints both side by side.

Nothing is written to results/ or runs/; a trained model goes to a temporary folder.

Commands (run from the Network_Agent folder, with STANCE_GYM_PATH set; see README_agent.md)
    python demo.py --train --smoke --episodes 3       quick check that everything works
    python demo.py --train                            full training (300k steps, a few minutes)
    python demo.py --model runs/networkA_mlp/<run>    evaluate a saved run

Uses: agent.train (network loading, config, wrappers, model building, architecture print),
agent.curve.CurveCallback, common.env_setup, common.evaluate and common.policies.
"""
import argparse
import json
import tempfile
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agent.curve import CurveCallback
from agent.train import (load_network, merged_config, make_wrappers, make_venv, build_model,
                         count_params, print_architecture)
from common.env_setup import make_env, to_norm_action
from common.evaluate import EVAL_SEEDS, REFERENCE, run_episodes, summarise
from common.policies import FixedPolicy

ROWS = [("survival_pct", "survival (%)"), ("toe_peak_bw", "toe peak (BW)"),
        ("mean_theta_deg", "mean angle (deg)"), ("mean_K", "mean K (N.m/rad)"),
        ("mean_B", "mean B (N.m.s/rad)"), ("mean_return", "mean return")]


def load_run(run_dir):
    """Load a saved run folder the same way `agent.train --eval-only` does.

    Returns (net, model, vecnormalize, wrappers). The VecNormalize statistics are frozen
    (training=False) so evaluation does not change them.
    """
    run_dir = Path(run_dir)
    saved = json.loads((run_dir / "config.json").read_text())
    cfg = saved["config"]
    net = load_network(saved["network"])
    wrappers = make_wrappers(cfg, net)
    model = net.ALGO.load(run_dir / "model.zip", device="cpu")
    vn = VecNormalize.load(str(run_dir / "vecnormalize.pkl"),
                           DummyVecEnv([lambda: make_env("train", wrappers)]))
    vn.training = False
    return net, model, vn, wrappers


def train_run(network, smoke, seed=0):
    """Train a network from scratch with the agent's settings and save it to a temporary folder.

    Same steps as `agent.train` without --eval-only: merged config, wrappers, normalised
    parallel environments, PPO model, and `model.learn` with the learning-curve logger.
    The folder holds config.json, model.zip, vecnormalize.pkl and curve.csv, so it can be
    passed back to this demo with --model.
    Returns (net, model, vecnormalize, wrappers).
    """
    net = load_network(network)
    cfg = merged_config(net, smoke)
    wrappers = make_wrappers(cfg, net)
    run_dir = Path(tempfile.mkdtemp(prefix=f"stance_demo_{net.NAME}_"))
    (run_dir / "config.json").write_text(
        json.dumps({"network": network, "seed": seed, "config": cfg}, indent=2))
    venv = make_venv(cfg, wrappers, seed)
    model = build_model(cfg, net, venv, seed)
    print(f"training {net.NAME} for {cfg['total_steps']:,} steps; run folder: {run_dir}")
    model.learn(cfg["total_steps"], callback=CurveCallback(cfg, wrappers, net.POLICY,
                                                           run_dir / "curve.csv"))
    model.save(run_dir / "model.zip")
    venv.save(str(run_dir / "vecnormalize.pkl"))
    venv.training = False
    return net, model, venv, wrappers


def main():
    """Parse the arguments, get a model, print its architecture, evaluate it and the reference."""
    p = argparse.ArgumentParser(description="Demo of the STANCE learning agent.")
    p.add_argument("--network", choices=["mlp", "lstm"], default="mlp", help="network to train")
    p.add_argument("--model", metavar="DIR", help="saved run folder (model.zip, vecnormalize.pkl, config.json)")
    p.add_argument("--episodes", type=int, default=20, help="evaluation episodes (seeds 1000 and up)")
    p.add_argument("--train", action="store_true", help="train from scratch instead of loading")
    p.add_argument("--smoke", action="store_true", help="tiny training run, only to check the pipeline")
    a = p.parse_args()

    # 1. Get a model: load a saved run, or train one.
    if a.model:
        net, model, vn, wrappers = load_run(a.model)
    elif a.train:
        net, model, vn, wrappers = train_run(a.network, a.smoke)
    else:
        print("Nothing to do. Choose one of:\n"
              "  python demo.py --model runs/networkA_mlp/<run>   evaluate a saved run\n"
              "  python demo.py --train [--smoke]                 train from scratch, then evaluate\n"
              "See README_agent.md.")
        return

    # 2. The network's layers and parameter count.
    print()
    print_architecture(model, vn, net)
    print(f"parameters: {count_params(model):,}")

    # 3. Evaluate the policy and the best fixed action on the same seeds.
    seeds = EVAL_SEEDS[: a.episodes]
    env = make_env("train", wrappers)
    agent = summarise(run_episodes(net.POLICY(model, vn), env, seeds))
    env.close()
    env = make_env("train")
    reference = summarise(run_episodes(FixedPolicy(to_norm_action(env, *REFERENCE)), env, seeds))
    env.close()

    # 4. Side-by-side table.
    print(f"\n{len(seeds)} episodes, seeds {seeds[0]}-{seeds[-1]}, deterministic policy")
    print(f"{'':20s} {net.NAME:>14s} {'fixed action':>14s}")
    for key, label in ROWS:
        print(f"{label:20s} {agent[key]:14.2f} {reference[key]:14.2f}")
    print(f"(fixed action = {REFERENCE[0]:g} deg, K {REFERENCE[1]:g}, B {REFERENCE[2]:g})")


if __name__ == "__main__":
    main()
