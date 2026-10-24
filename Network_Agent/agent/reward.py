"""The reweighted impact penalty. OWNER: P3.

The frozen environment charges, every step,

    impact = -W_IMPACT * max(0, toe_force_bw - LAMBDA_BW) ** 2

with W_IMPACT 2.94 and LAMBDA_BW 1.58 (Gym/stance_env/ankle_gym_env.py, AnkleEnv._reward),
where toe_force_bw is the step's peak toe force in body weights (info["toe_force_bw"]).
`ReweightedImpact` swaps that term for one with new weights and leaves the rest of the
reward (alive bonus, smoothness, fall penalty) untouched:

    reward_new = reward_env + (impact_original - impact_new)

where both impact terms are the positive penalties W * max(0, toe_force_bw - L) ** 2.
The environment's own reward is kept in info["reward_original"]. With w_impact and
lambda_bw set in config.py, train.py appends this wrapper last, so training and
evaluation both score with the new weights.

Uses: gymnasium.Wrapper and the W_IMPACT / LAMBDA_BW constants of the environment module.
"""
import sys

import gymnasium as gym


def impact_penalty(toe_force_bw, w_impact, lambda_bw):
    """The environment's impact penalty (a positive number) for one step's toe force."""
    return w_impact * max(0.0, toe_force_bw - lambda_bw) ** 2


class ReweightedImpact(gym.Wrapper):
    """Replaces the impact term of the reward with w_impact * max(0, toe_bw - lambda_bw)^2."""

    def __init__(self, env, w_impact, lambda_bw):
        super().__init__(env)
        module = sys.modules[type(env.unwrapped).__module__]
        self.w_orig, self.lambda_orig = module.W_IMPACT, module.LAMBDA_BW
        self.w_impact, self.lambda_bw = float(w_impact), float(lambda_bw)

    def step(self, action):
        """Step the environment and return the reweighted reward."""
        obs, r, term, trunc, info = self.env.step(action)
        toe = float(info["toe_force_bw"])
        info["reward_original"] = r
        r = r + impact_penalty(toe, self.w_orig, self.lambda_orig) \
              - impact_penalty(toe, self.w_impact, self.lambda_bw)
        return obs, r, term, trunc, info
