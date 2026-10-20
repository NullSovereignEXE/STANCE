"""Every setting that belongs to the learning agent. OWNER: P3.

The agent is the main program (train.py). The network files define the network
only; everything about HOW it is trained lives here: the PPO hyperparameters, the
loss weights, normalisation, the optional residual action, and the logging budget.
A network file's own CONFIG must not repeat any key from this file (train.py checks).

Any value can be overridden for one run without editing this file:
    python -m agent.train --network mlp --set learning_rate=1e-4 gamma=0.995

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
    residual         True makes the policy output an offset from `anchor` (see actions.py)
    anchor           the fixed action the residual starts from: degrees, N.m/rad, N.m.s/rad
    scale            largest offset per action dimension, in normalised units
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
    "residual": False,
    "anchor": [-20.0, 500.0, 10.0],
    "scale": [0.3, 0.3, 0.3],
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
"""Settings applied by --smoke: a tiny run that only checks the pipeline works end to end."""
