"""Normal and blind (observation = VecNormalize mean) evaluation of the six w6 runs, seeds 1000-1199.
Writes eps_<net>_<run>_<split>_<mode>.csv and prints one summary line per case."""
import sys, json, csv
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from common.env_setup import make_env
from common.evaluate import run_episodes, summarise, EVAL_SEEDS
from agent.train import load_network, make_wrappers

OUT = Path(sys.argv[1])

class ConstantObs(gym.ObservationWrapper):
    """Replaces every observation with a fixed vector."""
    def __init__(self, env, value):
        super().__init__(env)
        self.value = np.asarray(value, dtype=np.float32)
    def observation(self, obs):
        return self.value.copy()

for netdir in ("networkA_mlp", "networkB_lstm"):
    for s in range(3):
        run_dir = Path("runs") / netdir / f"w6_s{s}"
        saved = json.loads((run_dir / "config.json").read_text())
        cfg = saved["config"]
        assert cfg["w_impact"] == 6 and cfg["lambda_bw"] == 1.58, cfg
        net = load_network(saved["network"])
        wrappers = make_wrappers(cfg, net)
        model = net.ALGO.load(run_dir / "model.zip", device="cpu")
        vn = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), DummyVecEnv([lambda: make_env("train", wrappers)]))
        vn.training = False
        mean = vn.obs_rms.mean
        for split in ("train", "test"):
            for mode in ("normal", "blind"):
                env = make_env(split, wrappers)
                if mode == "blind":
                    env = ConstantObs(env, mean)
                eps = run_episodes(net.POLICY(model, vn), env, EVAL_SEEDS)
                env.close()
                with open(OUT / f"eps_{netdir}_w6_s{s}_{split}_{mode}.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(eps[0])); w.writeheader(); w.writerows(eps)
                m = summarise(eps)
                print(f"{netdir} w6_s{s} {split} {mode} surv {m['survival_pct']:.1f} toe {m['toe_peak_bw']:.3f} "
                      f"ret {m['mean_return']:.2f} theta {m['mean_theta_deg']:.2f} K {m['mean_K']:.1f} B {m['mean_B']:.3f}", flush=True)
