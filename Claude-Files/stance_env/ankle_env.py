"""
STANCE: powered prosthetic ankle on unknown compliant ground.

A Gymnasium environment. One episode = one stance phase (heel strike through
mid-stance, 0.5 s at 200 Hz).

The plumbing is done: Gym API, seeding, reset geometry, site velocities,
action rescaling, rendering. The two methods marked TODO are YOUR design
decisions -- they are where the project's argument lives.
"""
from pathlib import Path
import collections

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from stance_env.ground import normal_force, friction_force

# --------------------------------------------------------------- geometry
# (x, z) of each contact site in the foot's own frame. Must match leg.xml.
HEEL_OFFSET = np.array([-0.07, -0.05])
TOE_OFFSET = np.array([0.19, -0.05])

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "leg.xml"

# --------------------------------------------------------------- actuator
ANKLE_RANGE = (-0.349, 0.349)      # rad, +-20 deg
K_RANGE = (5.0, 500.0)             # N.m/rad
B_RANGE = (0.5, 20.0)              # N.m.s/rad
TAU_LIMIT = 150.0                  # N.m

# --------------------------------------------------------------- episode
HISTORY_LEN = 10                   # stacked timesteps -> 50 ms of history
N_SENSORS = 11                     # values per timestep
FRAME_SKIP = 5                     # 1 ms physics -> 200 Hz control
MAX_STEPS = 100                    # 0.5 s episode

# --------------------------------------------------------------- failure
MAX_TILT = np.deg2rad(15.0)        # leg pitch beyond this = topple
COLLAPSE_FRACTION = 0.70           # body below this x standing height = collapse

# --------------------------------------------------------------- reward
LAMBDA_BW = 2.0                    # impact threshold, multiples of body weight
W_IMPACT = 10.0                    # w1
W_SMOOTH = 0.5                     # w2

G = 9.81

# ----------------------------------------------------- ground distributions
# Training and held-out ranges from the proposal. The test ranges do NOT
# overlap the training ranges -- that is the anti-memorisation design.
GROUND_RANGES = {
    "train": dict(
        k0=[(5e3, 1e5)],
        c=(20.0, 400.0),
        alpha=(0.0, 15.0),
        f_yield=(300.0, 3000.0),
        mu=(0.30, 0.90),
    ),
    "test": dict(
        k0=[(2e3, 5e3), (1e5, 2e5)],      # two disjoint bands
        c=(400.0, 800.0),
        alpha=(15.0, 25.0),
        f_yield=(150.0, 300.0),
        mu=(0.15, 0.30),
    ),
}


class AnkleEnv(gym.Env):
    """Prosthetic ankle learning to set its own impedance on unknown ground."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 200}

    def __init__(self, ground_split="train", render_mode=None):
        if ground_split not in GROUND_RANGES:
            raise ValueError(f"ground_split must be 'train' or 'test', got {ground_split}")
        self.ground_split = ground_split

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        # Cache ids once. Looking these up every step is needlessly slow.
        S, J, B = mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY
        nid = lambda t, n: mujoco.mj_name2id(self.model, t, n)
        self.sid = [nid(S, "heel_site"), nid(S, "toe_site")]
        self.foot_bid = nid(B, "foot")
        self.body_bid = nid(B, "body_mass")
        self.ankle_jid = nid(J, "ankle_hinge")
        self.pylon_jid = nid(J, "pylon_slide")

        # qpos/qvel indices, in XML order
        self.IX, self.IZ, self.IPITCH, self.IANKLE, self.IPYLON = 0, 1, 2, 3, 4

        self.total_mass = float(self.model.body_subtreemass[self.foot_bid])
        self.body_weight = self.total_mass * G

        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (HISTORY_LEN * N_SENSORS,), np.float32
        )

        self.render_mode = render_mode
        self._renderer = None
        self._standing_height = None

    # ====================================================== Gym: reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)            # seeds self.np_random -- always use it
        mujoco.mj_resetData(self.model, self.data)

        self._sample_ground()
        self._set_initial_pose()
        mujoco.mj_forward(self.model, self.data)

        if self._standing_height is None:
            self._standing_height = 0.85    # ankle-to-mass length from leg.xml

        self.history = collections.deque(
            [np.zeros(N_SENSORS)] * HISTORY_LEN, maxlen=HISTORY_LEN
        )
        self.step_count = 0
        self.prev_action = np.zeros(3)
        self.prev_site_xpos = np.array(
            [self.data.site_xpos[i].copy() for i in self.sid]
        )
        self._last_normal = 0.0

        return self._get_obs(), {}

    def _sample_ground(self):
        """Draw per-contact-point material properties for this episode."""
        rng = self.np_random
        r = GROUND_RANGES[self.ground_split]
        n = 2                                # heel, toe -- drawn independently

        # k0 may come from several disjoint bands (the test split has two)
        bands = r["k0"]
        pick = rng.integers(0, len(bands), n)
        self.k0 = np.array([
            10 ** rng.uniform(np.log10(bands[p][0]), np.log10(bands[p][1]))
            for p in pick
        ])
        self.c_damp = rng.uniform(*r["c"], n)
        self.alpha = rng.uniform(*r["alpha"], n)
        self.f_yield = rng.uniform(*r["f_yield"], n)
        self.mu = rng.uniform(*r["mu"], n)

        self.surface_z = np.zeros(n)         # dent depth, grows as ground yields

    def _set_initial_pose(self):
        """Place the leg with the HEEL exactly touching, carrying approach velocity.

        Rotating a foot-frame offset (x, z) about +y by phi gives world offset
            (x cos phi + z sin phi,  -x sin phi + z cos phi)
        Setting the heel's world height to zero therefore requires
            foot_z = x_heel sin(phi) - z_heel cos(phi)
        At phi = 0 this gives 0.05 m, which is the site offset -- a useful check.
        """
        rng = self.np_random
        phi = rng.uniform(np.deg2rad(-8), np.deg2rad(4))
        ankle_q = rng.uniform(np.deg2rad(-5), np.deg2rad(5))
        v_fwd = rng.uniform(0.8, 1.4)
        v_down = rng.uniform(0.10, 0.50)

        foot_z = HEEL_OFFSET[0] * np.sin(phi) - HEEL_OFFSET[1] * np.cos(phi)

        self.data.qpos[:] = [0.0, foot_z, phi, ankle_q, 0.0]
        self.data.qvel[:] = [v_fwd, -v_down, 0.0, 0.0, 0.0]

    # ====================================================== Gym: step
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        theta_d, K, B = self._rescale(action)

        for _ in range(FRAME_SKIP):
            self._apply_ground_forces()
            # Impedance law. Recomputed EVERY physics step, not once per control
            # step -- it depends on angle and velocity, which change at 1 ms.
            q = self.data.qpos[self.IANKLE]
            qd = self.data.qvel[self.IANKLE]
            self.data.ctrl[0] = np.clip(K * (theta_d - q) - B * qd,
                                        -TAU_LIMIT, TAU_LIMIT)
            mujoco.mj_step(self.model, self.data)
            self.data.qfrc_applied[:] = 0.0   # our forces are per-step, not sticky

        self.history.append(self._sensors())
        self.step_count += 1

        reward = self._reward(action)
        terminated = self._failed()
        truncated = self.step_count >= MAX_STEPS
        self.prev_action = action.copy()

        info = {
            "normal_force_bw": self._last_normal / self.body_weight,
            "leg_tilt_deg": np.rad2deg(self.data.qpos[self.IPITCH]),
            "body_height": self.data.xpos[self.body_bid][2],
            "dent_mm": -self.surface_z * 1000.0,
        }
        return self._get_obs(), reward, bool(terminated), bool(truncated), info

    @staticmethod
    def _rescale(a):
        """[-1, 1] -> physical units.

        K and B are mapped LOGARITHMICALLY. K spans 5-500 N.m/rad, a factor of
        100; a linear map would bury the soft end (5-50) in the last 10% of the
        action range, making compliant behaviour nearly unreachable.
        """
        theta_d = ANKLE_RANGE[0] + (a[0] + 1) / 2 * (ANKLE_RANGE[1] - ANKLE_RANGE[0])

        def logspan(rng, x):
            lo, hi = np.log10(rng[0]), np.log10(rng[1])
            return 10 ** (lo + (x + 1) / 2 * (hi - lo))

        return theta_d, logspan(K_RANGE, a[1]), logspan(B_RANGE, a[2])

    # ================================================= ground interaction
    def _site_velocity(self, i):
        """Finite difference of site position.

        Simpler and harder to get subtly wrong than mj_objectVelocity, whose
        6-vector component order is easy to misread.
        """
        cur = self.data.site_xpos[self.sid[i]]
        v = (cur - self.prev_site_xpos[i]) / self.model.opt.timestep
        self.prev_site_xpos[i] = cur.copy()
        return v

    def _apply_ground_forces(self):
        """Compute and apply our own contact forces. MuJoCo's floor is disabled."""
        self.data.qfrc_applied[:] = 0.0
        total_normal = 0.0

        for i in range(2):
            pos = self.data.site_xpos[self.sid[i]]
            vel = self._site_velocity(i)
            depth = self.surface_z[i] - pos[2]

            Fn, extra_dent = normal_force(
                depth, -vel[2], self.k0[i], self.c_damp[i],
                self.alpha[i], self.f_yield[i]
            )
            if extra_dent is not None:
                self.surface_z[i] -= extra_dent      # surface stays dented

            if Fn <= 0.0:
                continue

            Ft = friction_force(Fn, vel[0], self.mu[i])
            total_normal += Fn

            mujoco.mj_applyFT(
                self.model, self.data,
                np.array([Ft, 0.0, Fn]), np.zeros(3),
                pos, self.foot_bid, self.data.qfrc_applied,
            )

        self._last_normal = total_normal

    # ================================================= observation
    def _sensors(self):
        """The 11 values, in a FIXED order. Agree it as a team; never reorder.

        Every one exists on real prosthetic hardware. Ground properties are
        deliberately absent -- inferring them is the agent's job.
        """
        d = self.data
        shank_vacc = d.qacc[self.IZ] if d.qacc is not None else 0.0
        return np.array([
            d.qpos[self.IANKLE],       # 0  ankle angle        (encoder)
            d.qvel[self.IANKLE],       # 1  ankle rate         (encoder)
            d.ctrl[0],                 # 2  ankle torque       (motor current)
            self._last_normal,         # 3  vertical GRF       (pylon load cell)
            d.qpos[self.IPYLON],       # 4  pylon compression  (linear sensor)
            d.qpos[self.IPITCH],       # 5  shank tilt         (IMU)
            d.qvel[self.IPITCH],       # 6  shank tilt rate    (IMU)
            shank_vacc,                # 7  vertical accel     (IMU)
            *self.prev_action,         # 8,9,10  previous action
        ], dtype=np.float64)

    def _get_obs(self):
        return np.concatenate(list(self.history)).astype(np.float32)

    # ================================================= reward / termination
    def _reward(self, action):
        """Three terms: survival, impact, smoothness.

        TODO(B): these weights are a STARTING POINT, not an answer. The single
        ratio that matters is W_IMPACT against the survival bonus: how many
        steps of successful support is one body weight of excess force worth?
        Expect the first policy to sit at maximum stiffness -- that is the
        baseline's behaviour, and the fix is to raise W_IMPACT.
        """
        survival = 1.0

        excess = max(0.0, self._last_normal / self.body_weight - LAMBDA_BW)
        impact = -W_IMPACT * excess ** 2

        smooth = -W_SMOOTH * float(np.sum((action - self.prev_action) ** 2))

        return survival + impact + smooth

    def _failed(self):
        """Terminal failure: toppled, or collapsed.

        Note this returns `terminated`, NOT `truncated`. Running out of steps is
        a time limit, not a failure, and Gym's TimeLimit wrapper handles that.
        Conflating the two teaches the critic that surviving is worthless.
        """
        toppled = abs(self.data.qpos[self.IPITCH]) > MAX_TILT
        collapsed = (self.data.xpos[self.body_bid][2]
                     < COLLAPSE_FRACTION * self._standing_height)
        return toppled or collapsed

    # ================================================= rendering
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        # Move the visual floor down to show the dent, so the leg does not
        # appear to sink into nothing.
        self.model.geom("floor").pos[2] = float(self.surface_z.min())
        self._renderer.update_scene(self.data, camera="side")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
