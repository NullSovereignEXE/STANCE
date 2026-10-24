"""Every setting that belongs to the learning agent.

Keys
    total_steps      environment steps to train for
    n_envs           environments stepped in parallel
    n_steps          steps per environment per update; one update uses n_envs * n_steps samples
    batch_size       minibatch size inside an update
    n_epochs         passes over each batch of samples
    learning_rate    Adam step size
    gamma            discount factor; how much later reward and the fall penalty count
    gae_lambda       smoothing of the advantage estimate
    clip_range       PPO's limit on how far one update may change the policy
    ent_coef         weight of the entropy bonus in the loss (more exploration)
    vf_coef          weight of the value-function error in the loss
    max_grad_norm    gradients are clipped to this length before each Adam step
    norm_obs         normalise observations with running statistics (VecNormalize)
    norm_reward      scale rewards by a running estimate of their size (VecNormalize)
    subproc          True runs each environment in its own process (faster with many cores)
    residual         True (default since config v1) makes the policy output an offset from
                     `anchor` (see actions.py); False gives plain PPO
    anchor           the fixed action the residual starts from: degrees, N.m/rad, N.m.s/rad
                     (config v1: -14.0 deg, 250.6, 6.38, inside the action range)
    scale            largest offset per action dimension, in normalised units
    w_impact         new weight on the squared toe-force excess (env: 2.94); see reward.py
    lambda_bw        new toe-force threshold in body weights (env: 1.58). None for either
                     keeps the environment's reward unchanged
    curve_every      steps between learning-curve points (written to curve.csv)
    curve_episodes   episodes per learning-curve point, on seeds 2000 and up
    eval_episodes    episodes in the final evaluation, on seeds starting at 1000

The loss PPO minimises, which the lecture assigns to the network side but which the
library computes, is:
    loss = -(clipped policy objective) + vf_coef * (value error)^2 - ent_coef * entropy
"""

AGENT_CONFIG = {
    "total_steps": 300_000,
    "n_envs": 8,
    "n_steps": 256,
    "batch_size": 512,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "norm_obs": True,
    "norm_reward": True,
    "subproc": False,
    "residual": True,
    "anchor": [-14.0, 250.6, 6.38],
    "scale": [0.3, 0.3, 0.3],
    "w_impact": None,
    "lambda_bw": None,
    "curve_every": 20_000,
    "curve_episodes": 50,
    "eval_episodes": 200,
}

SMOKE = {
    "total_steps": 4096,
    "n_envs": 2,
    "n_steps": 256,
    "batch_size": 256,
    "curve_every": 2048,
    "curve_episodes": 3,
    "eval_episodes": 5,
}

