"""
THE GYM ENVIRONMENT -- owned by Person 3.

The Gymnasium contract and the RL cast: observation, action, reward,
termination. The ground is delegated to GroundModel, rendering to Recorder,
so this file is purely the RL side.

One episode = one stance phase: heel strike through mid-stance, 0.5 s at 200 Hz.
"""
from pathlib import Path
import collections

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from stance_env.ground import GroundModel

# ----------------------------------------------- geometry (must match leg.xml)
HEEL_OFFSET = np.array([-0.07, -0.05])     # (x, z) in the foot's own frame
TOE_OFFSET = np.array([0.19, -0.05])

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "leg.xml"

# ----------------------------------------------- actuator
ANKLE_RANGE = (-0.349, 0.349)              # rad, +-20 deg
K_RANGE = (5.0, 500.0)                     # N.m/rad
B_RANGE = (0.5, 20.0)                      # N.m.s/rad
TAU_LIMIT = 150.0                          # N.m

# ----------------------------------------------- episode
HISTORY_LEN = 10                           # stacked steps -> 50 ms of history
N_SENSORS = 11
FRAME_SKIP = 5                             # 1 ms physics -> 200 Hz control
MAX_STEPS = 100                            # 0.5 s

# ----------------------------------------------- failure
MAX_TILT = np.deg2rad(15.0)
COLLAPSE_FRACTION = 0.70
STANDING_HEIGHT = 0.85                     # ankle-to-mass length from leg.xml

# ----------------------------------------------- reward
LAMBDA_BW = 2.0                            # impact threshold, body weights
W_IMPACT = 10.0                            # w1  <- the main tuning knob
W_SMOOTH = 0.5                             # w2

G = 9.81


class AnkleEnv(gym.Env):
    """Prosthetic ankle learning to set its own impedance on unknown ground."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 200}

    def __init__(self, ground_split="train", render_mode=None,
                 history_len=HISTORY_LEN):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        self.ground = GroundModel(split=ground_split, n_points=2)
        self.ground_split = ground_split
        self.history_len = history_len     # varied for the history ablation

        S, J, B = (mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_JOINT,
                   mujoco.mjtObj.mjOBJ_BODY)
        nid = lambda t, n: mujoco.mj_name2id(self.model, t, n)
        self.sid = [nid(S, "heel_site"), nid(S, "toe_site")]
        self.foot_bid = nid(B, "foot")
        self.body_bid = nid(B, "body_mass")

        # qpos/qvel indices, in leg.xml joint order
        self.IX, self.IZ, self.IPITCH, self.IANKLE, self.IPYLON = 0, 1, 2, 3, 4

        self.total_mass = float(self.model.body_subtreemass[self.foot_bid])
        self.body_weight = self.total_mass * G

        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (history_len * N_SENSORS,), np.float32)

        self.render_mode = render_mode
        self._recorder = None

    # ===================================================== Gym: reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)           # seeds self.np_random -- always use it
        mujoco.mj_resetData(self.model, self.data)

        self.ground.sample(self.np_random)
        self._set_initial_pose()
        mujoco.mj_forward(self.model, self.data)

        self.history = collections.deque(
            [np.zeros(N_SENSORS)] * self.history_len, maxlen=self.history_len)
        self.step_count = 0
        self.prev_action = np.zeros(3)
        self.prev_site_xpos = np.array(
            [self.data.site_xpos[i].copy() for i in self.sid])

        return self._get_obs(), {}

    def _set_initial_pose(self):
        """Place the leg with the HEEL exactly touching, carrying approach velocity.

        Rotating a foot-frame offset (x, z) about +y by phi gives world offset
            (x cos phi + z sin phi,  -x sin phi + z cos phi)
        so putting the heel's world height at zero requires
            foot_z = x_heel sin(phi) - z_heel cos(phi)
        At phi = 0 this gives 0.05 m, exactly the site's z offset -- a useful
        check that the geometry constants still match leg.xml.
        """
        rng = self.np_random
        phi = rng.uniform(np.deg2rad(-8), np.deg2rad(4))
        ankle_q = rng.uniform(np.deg2rad(-5), np.deg2rad(5))
        v_fwd = rng.uniform(0.8, 1.4)
        v_down = rng.uniform(0.10, 0.50)

        foot_z = HEEL_OFFSET[0] * np.sin(phi) - HEEL_OFFSET[1] * np.cos(phi)

        self.data.qpos[:] = [0.0, foot_z, phi, ankle_q, 0.0]
        self.data.qvel[:] = [v_fwd, -v_down, 0.0, 0.0, 0.0]

    # ===================================================== Gym: step
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        theta_d, K, B = self._rescale(action)

        for _ in range(FRAME_SKIP):
            self._apply_ground_forces()
            # Impedance law, recomputed EVERY physics step -- it depends on
            # angle and velocity, which change at 1 ms.
            q = self.data.qpos[self.IANKLE]
            qd = self.data.qvel[self.IANKLE]
            self.data.ctrl[0] = np.clip(K * (theta_d - q) - B * qd,
                                        -TAU_LIMIT, TAU_LIMIT)
            mujoco.mj_step(self.model, self.data)
            self.data.qfrc_applied[:] = 0.0    # our forces are per-step, not sticky

        self.history.append(self._sensors())
        self.step_count += 1

        reward = self._reward(action)
        terminated = self._failed()
        truncated = self.step_count >= MAX_STEPS
        self.prev_action = action.copy()

        info = {
            "normal_force_bw": self.ground.last_normal / self.body_weight,
            "leg_tilt_deg": float(np.rad2deg(self.data.qpos[self.IPITCH])),
            "body_height": float(self.data.xpos[self.body_bid][2]),
            **self.ground.describe(),
        }
        return self._get_obs(), reward, bool(terminated), bool(truncated), info

    @staticmethod
    def _rescale(a):
        """[-1, 1] -> physical units.

        K and B are mapped LOGARITHMICALLY. K spans 5-500 N.m/rad, a factor of
        100; a linear map would squeeze 5-15 into the last 2% of the action
        range, making compliant behaviour nearly unreachable.
        """
        theta_d = ANKLE_RANGE[0] + (a[0] + 1) / 2 * (ANKLE_RANGE[1] - ANKLE_RANGE[0])

        def logspan(rng, x):
            lo, hi = np.log10(rng[0]), np.log10(rng[1])
            return 10 ** (lo + (x + 1) / 2 * (hi - lo))

        return theta_d, logspan(K_RANGE, a[1]), logspan(B_RANGE, a[2])

    # ===================================================== ground coupling
    def _site_velocity(self, i):
        """Finite difference. Simpler and harder to get subtly wrong than
        mj_objectVelocity, whose 6-vector component order is easy to misread."""
        cur = self.data.site_xpos[self.sid[i]]
        v = (cur - self.prev_site_xpos[i]) / self.model.opt.timestep
        self.prev_site_xpos[i] = cur.copy()
        return v

    def _apply_ground_forces(self):
        self.data.qfrc_applied[:] = 0.0

        positions = np.array([self.data.site_xpos[i] for i in self.sid])
        velocities = np.array([self._site_velocity(i) for i in range(len(self.sid))])

        forces = self.ground.forces(positions, velocities)

        for i, f in enumerate(forces):
            if not np.any(f):
                continue
            mujoco.mj_applyFT(self.model, self.data, f, np.zeros(3),
                              positions[i], self.foot_bid, self.data.qfrc_applied)

    # ===================================================== observation
    def _sensors(self):
        """The 11 values, in a FIXED order. Agree it as a team; never reorder.

        Every one exists on real prosthetic hardware. Ground properties are
        deliberately absent -- inferring them is the agent's job.
        """
        d = self.data
        return np.array([
            d.qpos[self.IANKLE],          # 0  ankle angle        (encoder)
            d.qvel[self.IANKLE],          # 1  ankle rate         (encoder)
            d.ctrl[0],                    # 2  ankle torque       (motor current)
            self.ground.last_normal,      # 3  vertical GRF       (pylon load cell)
            d.qpos[self.IPYLON],          # 4  pylon compression  (linear sensor)
            d.qpos[self.IPITCH],          # 5  shank tilt         (IMU)
            d.qvel[self.IPITCH],          # 6  shank tilt rate    (IMU)
            d.qacc[self.IZ],              # 7  vertical accel     (IMU)
            *self.prev_action,            # 8,9,10  previous action
        ], dtype=np.float64)

    def _get_obs(self):
        return np.concatenate(list(self.history)).astype(np.float32)

    # ===================================================== reward / failure
    def _reward(self, action):
        """Three terms: survival, impact, smoothness.

        TODO(P3): these weights are a STARTING POINT, not an answer. The one
        ratio that matters is W_IMPACT against the survival bonus: how many
        steps of successful support is one body weight of excess force worth?
        Expect the first policy to sit at maximum stiffness -- that is the
        baseline's behaviour, and the fix is to raise W_IMPACT.
        """
        survival = 1.0

        excess = max(0.0, self.ground.last_normal / self.body_weight - LAMBDA_BW)
        impact = -W_IMPACT * excess ** 2

        smooth = -W_SMOOTH * float(np.sum((action - self.prev_action) ** 2))

        return survival + impact + smooth

    def _failed(self):
        """Terminal failure: toppled, or collapsed.

        Returns `terminated`, NOT `truncated`. Running out of steps is a time
        limit, not a failure; conflating them teaches the critic that surviving
        is worthless.
        """
        toppled = abs(self.data.qpos[self.IPITCH]) > MAX_TILT
        collapsed = (self.data.xpos[self.body_bid][2]
                     < COLLAPSE_FRACTION * STANDING_HEIGHT)
        return toppled or collapsed

    # ===================================================== rendering
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._recorder is None:
            from stance_env.viz import Recorder
            self._recorder = Recorder(self.model)
        self.ground.update_visual(self.model)     # show the dent
        return self._recorder.capture(self.data)

    def close(self):
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None
