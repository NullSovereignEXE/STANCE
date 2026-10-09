from pathlib import Path
import collections

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from stance_env.ground import GroundModel # This will be pulled from Mukul's work

# Leg geometry is defined in the MuJoCo XML file, but we need to know some of the dimensions here for computing the ankle torque and leg tilt.
HEEL_OFFSET = np.array([-0.07, -0.05])  # heel position (x, z) in the FOOT's own frame
TOE_OFFSET = np.array([0.19, -0.05])    # toe position (x, z) in the FOOT's own frame

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "leg.xml" # Path to the MuJoCo XML model file for the leg

# Actuator limits for the ankle joint (The ankle joint is the only actuator in this environment)
ANKLE_RANGE = (-np.deg2rad(20.0), np.deg2rad(20.0))  # The range of motion for the ankle joint, in radians (-20 to +20 degrees or -0.349 to +0.349 radians)
K_RANGE = (5.0, 500.0)         # The stiffness range for the ankle joint, in N.m/rad (5 to 500 N.m/rad)
B_RANGE = (0.5, 20.0)          # The damping range for the ankle joint, in N.m.s/rad (0.5 to 20 N.m.s/rad)
TAU_LIMIT = 150.0              # The maximum torque for the ankle joint, in N.m (150 N.m)

# Episode information for the environment
MAX_EPISODE_LENGTH = 100  # The maximum number of steps per episode
HISTORY_LENGTH = 10        # The number of past observations to keep in the observation space
FRAME_SKIP = 5             # The number of "physics simulation" steps per actual program environment step. T
                           # This is used to speed up the simulation and reduce the computational load.
SENSOR_INPUTS = 11         # The number of sensor inputs for the agent
'''
Sensor inputs:
1. Ankle angle (encoder)
2. Ankle angular velocity (encoder)
3. Ankle torque (motor current)
4. Vertical ground reaction force (pylon load cell)
5. Pylon compression (pylon load cell)
6. Shank tilt angle (IMU)
7. Shank tilt rate (IMU)
8. Shank vertical acceleration (IMU)
9, 10, 11. Previous actions (3 inputs: ankle angle, stiffness, damping)
'''

# Episode failure conditions for the environment
FALL_ANGLE = np.deg2rad(15.0)  # The maximum tilt angle for the leg before the episode is considered a failure, in radians (15 degrees or 0.262 radians)
COLLAPSE_FRACTION = 0.70       # The fraction of the standing height below which the episode is considered a failure (70% of standing height)

# Reward weights for the environment
W_SMOOTH  = 0.5     # smoothness penalty weight
W_IMPACT = 1.2     # weight for impact penalty. # This was set to 2.0 after running the smoke test, providing the longest survival time.
LAMBDA_BW = 2.0     # impact threshold given in multiples of body weights
'''
LAMBDA_BW is the threshold for the vertical ground reaction force (GRF) above which the agent is penalized. 
The GRF is normalized by the body weight of the leg, and if it exceeds this threshold, a penalty is applied to the reward function. 
This encourages the agent to avoid high impact forces during stance.
'''

G = 9.81 # Gravity constant in m/s^2

class AnkleEnv(gym.Env):
    ''' 
    Agent learning is independent of the ground model. 
    The agent is trained on a variety of ground conditions, and evaluated on a separate set of test specific ground conditions. 
    '''
    metadata = {"render_modes": ["rgb_array"], "render_fps": 200}

    def __init__(self, ground_split="train", render_mode=None, history_len=HISTORY_LENGTH):

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        
        self.ground = GroundModel(split=ground_split, n_points=2)  # Initialize the ground model with the specified split (train or test) and number of contact points (2)
        self.ground_split = ground_split
        self.history_len = history_len 

        S = mujoco.mjtObj.mjOBJ_SITE 
        J = mujoco.mjtObj.mjOBJ_JOINT  
        B = mujoco.mjtObj.mjOBJ_BODY   

        nid = lambda t, n: mujoco.mj_name2id(self.model, t, n) # Helper function to get the ID of a MuJoCo object (site, joint, or body) by name
        self.sid = [nid(S, "heel_site"), nid(S, "toe_site")]
        self.foot_bid = nid(B, "foot")
        self.body_bid = nid(B, "body_mass")
        self.shank_bid = nid(B, "shank") 

        # Get the indices of the joint positions and velocities for the ankle joint and other relevant joints in the MuJoCo model.
        def jadr(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint {name!r} not found in {MODEL_PATH.name}")
            return int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])

        self.IX,     self.VX     = jadr("foot_x")
        self.IZ,     self.VZ     = jadr("foot_z")
        self.IPITCH, self.VPITCH = jadr("foot_pitch")
        self.IANKLE, self.VANKLE = jadr("ankle_hinge")
        self.IPYLON, self.VPYLON = jadr("pylon_slide")

        self.total_mass = float(self.model.body_subtreemass[self.foot_bid])
        self.body_weight = self.total_mass * G
        self.standing_height = self._measure_standing_height() # Measure the standing height of the leg in the MuJoCo simulation directly.

        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (history_len * SENSOR_INPUTS,), np.float32)

        self.render_mode = render_mode
        self._recorder = None

    def _measure_standing_height(self):
        """
        World height of the body mass in the model's neutral pose.
        Uses a scratch MjData so the live simulation state is untouched.
        """
        scratch = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, scratch)
        mujoco.mj_forward(self.model, scratch)
        return float(scratch.xpos[self.body_bid][2])

### Reset function: This is where the environment is reset to its initial state at the beginning of each episode.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.ground.sample(self.np_random)
        self._set_initial_pose()
        mujoco.mj_forward(self.model, self.data) 
        # ^^^ Computes the forward dynamics of the MuJoCo model to update the simulation state based on the current positions and velocities
        
        '''
        We need to keep a history of the last N sensor readings to provide temporal context to the agent.
        This is important for the agent to learn the dynamics of the system and make better decisions based on past observations. 
        The history is stored in a deque (double-ended queue) with a maximum length of `history_len`, which is specified during the environment initialization.
        '''
        self.history = collections.deque(
            [np.zeros(SENSOR_INPUTS)] * self.history_len, maxlen=self.history_len)
        self.step_count = 0
        self.peak_force = 0.0
        self.prev_action = np.zeros(3)
        self.prev_site_xpos = np.array(
            [self.data.site_xpos[i].copy() for i in self.sid])

        return self._get_obs(), {}
    
    def _set_initial_pose(self):
            '''
            Place the leg with the heel ALREADY touching the ground but carrying approach velocity.
            We need to set the initial position of the leg such that the heel is already touching the ground, but the leg is still moving downward with some velocity.
            The leg tilt angle (phi) and the ankle joint angle (ankle_q) are randomly sampled within specified ranges to provide variability in the initial conditions. 
            The forward and downward velocities (v_fwd and v_down) are also randomly sampled to simulate different approach speeds.
            '''
            rng = self.np_random
            phi     = rng.uniform(np.deg2rad(-8), np.deg2rad(4))
            ankle_q = rng.uniform(np.deg2rad(-5), np.deg2rad(5))
            v_fwd   = rng.uniform(0.8, 1.4)
            v_down  = rng.uniform(0.10, 0.50)

            # Height that puts the HEEL exactly on the ground, minus the foot
            # body's own offset in the XML (slide qpos is relative, not absolute)
            want_z = HEEL_OFFSET[0] * np.sin(phi) - HEEL_OFFSET[1] * np.cos(phi)
            foot_z = want_z - self.model.body_pos[self.foot_bid][2]

            self.data.qpos[self.IZ]     = foot_z
            self.data.qpos[self.IPITCH] = phi
            self.data.qpos[self.IANKLE] = ankle_q
            self.data.qvel[self.VX]     = v_fwd
            self.data.qvel[self.VZ]     = -v_down

### Step function: This is where the agent takes an action and the environment responds with the next state, reward, and done signal.

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) # Clip the action values to be within the range [-1, 1]
        theta_d, K, B = self._rescale(action) # Rescaled using the _rescale function.

        '''
        For each environment step, we need to apply the ground forces and update the simulation state for a number of physics steps (FRAME_SKIP).
        This is done to simulate the dynamics of the system over a longer time period and to reduce the computational load by skipping some of the physics steps.
        The ankle torque is computed using an impedance control law based on the desired ankle angle (theta_d), stiffness (K), and damping (B). 
        The torque is clipped to be within the actuator limits (TAU_LIMIT) to ensure that the actuator does not exceed its maximum torque capacity. 
        '''
        peak_force = 0.0 # Tracking the peak ground reaction force during the step for reward calculation and info logging.

        for _ in range(FRAME_SKIP):
            self._apply_ground_forces()
            peak_force = max(peak_force, self.ground.last_normal) # Update the peak ground reaction force based on the last normal force computed by the ground model.
            q = self.data.qpos[self.IANKLE]
            qd = self.data.qvel[self.VANKLE]
            self.data.ctrl[0] = np.clip(K * (theta_d - q) - B * qd,
                                        -TAU_LIMIT, TAU_LIMIT)
            mujoco.mj_step(self.model, self.data)
            self.data.qfrc_applied[:] = 0.0    # our forces are per-step, not sticky

        self.peak_force = peak_force
        self.history.append(self._sensors())
        self.step_count += 1

        '''
        The reward is computed based on the current state of the simulation and the action taken by the agent.
        Designed to encourage maintaining balance and avoiding falls.
        Penalizes high impact forces and promotes smoothness in the ankle torque.
        '''
        reward = self._reward(action) 
        terminated = self._failed()
        truncated = self.step_count >= MAX_EPISODE_LENGTH
        self.prev_action = action.copy()

        info = {
            "normal_force_bw": self.peak_force / self.body_weight,
            "leg_tilt_deg": float(np.rad2deg(self._leg_tilt())),
            "body_height": float(self.data.xpos[self.body_bid][2]),
            **self.ground.describe(),
        }
        
        truncated = (self.step_count >= MAX_EPISODE_LENGTH) and not terminated
        return self._get_obs(), reward, bool(terminated), bool(truncated), info
    
    @staticmethod

    def _rescale(a):
            '''
            Rescale the action values from the range [-1, 1] to the actual physical ranges for the ankle joint angle, stiffness, and damping.
            The ankle joint angle is rescaled to the range defined by ANKLE_RANGE, the stiffness is rescaled to the range defined by K_RANGE, and the damping is rescaled to the range defined by B_RANGE. 
            This is done to ensure that the agent's actions are mapped to valid physical values that can be applied to the system.
            '''
            theta_d = ANKLE_RANGE[0] + (a[0] + 1) / 2 * (ANKLE_RANGE[1] - ANKLE_RANGE[0])
    
            def logspan(rng, x):
                lo, hi = np.log10(rng[0]), np.log10(rng[1])
                return 10 ** (lo + (x + 1) / 2 * (hi - lo))
    
            return theta_d, logspan(K_RANGE, a[1]), logspan(B_RANGE, a[2])


### Ground coupling: we need to know the world-frame positions and velocities of the heel and toe sites to compute the ground reaction forces.

    def _site_velocity(self, i):
        '''
        Compute the velocity of a contact site (heel or toe) based on its current and previous positions.
        This is used to compute the ground reaction forces (next def) based on the contact site velocities.
        '''
        cur = self.data.site_xpos[self.sid[i]]
        v = (cur - self.prev_site_xpos[i]) / self.model.opt.timestep
        self.prev_site_xpos[i] = cur.copy()
        return v

    def _apply_ground_forces(self):
        '''
        Apply the ground reaction forces to the contact sites (heel and toe)
        The ground reaction forces are computed using the GroundModel class, which takes into account the ground properties and the contact site dynamics. 
        The forces are applied to the MuJoCo simulation using the `mj_applyFT` function, which applies a force and torque to a body at a specified position.  
        '''
        self.data.qfrc_applied[:] = 0.0

        positions = np.array([self.data.site_xpos[i] for i in self.sid])
        velocities = np.array([self._site_velocity(i) for i in range(len(self.sid))])

        forces = self.ground.forces(positions, velocities)

        '''
        Apply the computed ground reaction forces to the MuJoCo simulation.
        The forces are applied to the foot body at the contact site positions using the `mj_applyFT` function. 
        This updates the simulation state based on the applied forces, which affects the dynamics of the leg and the ground interaction.
        '''
        for i, f in enumerate(forces):
            if not np.any(f):
                continue                         # Force,   torque,    position,      body_id,      applied_force_array
            mujoco.mj_applyFT(self.model, self.data, f, np.zeros(3), positions[i], self.foot_bid, self.data.qfrc_applied)

### Observation functions: we need to provide the agent with a set of observations that it can use to make decisions.
    
    def _leg_tilt(self):
        """
        MuJoCo joint angles are relative to the parent body. foot_pitch is
        the foot's angle relative to the world, but the shank hangs off the
        foot via ankle_hinge -- so foot_pitch alone measures the FOOT, not
        the leg. Reading the shank's world rotation matrix asks MuJoCo where
        the shank actually is, which cannot drift from the model.
        """
        R = self.data.xmat[self.shank_bid].reshape(3, 3)
        return np.arctan2(R[0, 2], R[2, 2])

    def _leg_tilt_rate(self):
        """
        Shank angular velocity about the pitch axis, rad/s.
        """
        return self.data.qvel[self.VPITCH] + self.data.qvel[self.VANKLE]
    
    def _sensors(self):    
        '''
        Get the current sensor readings from the MuJoCo simulation and return them as a numpy array.
        These sensor readings are used to provide the agent with information about the current state of the system, to make decisions about the next action choice.
        '''  
        d = self.data
        return np.array([
            d.qpos[self.IANKLE],          # 0  ankle angle        (encoder)
            d.qvel[self.VANKLE],          # 1  ankle rate         (encoder)
            d.ctrl[0],                    # 2  ankle torque       (motor current)
            self.ground.last_normal,      # 3  vertical GRF       (pylon load cell)
            d.qpos[self.IPYLON],          # 4  pylon compression  (linear sensor)
            self._leg_tilt(),             # 5  shank tilt         (IMU)
            self._leg_tilt_rate(),        # 6  shank tilt rate    (IMU)
            d.qacc[self.VZ],              # 7  vertical accel     (IMU)
            *self.prev_action,            # 8,9,10  previous action (Memory)
        ], dtype=np.float64)

    def _get_obs(self):
        return np.concatenate(list(self.history)).astype(np.float32)

    # ===================================================== reward / failure
    def _reward(self, action):
        survival = 1.0 # Reward for surviving the step. 

        excess = max(0.0, self.peak_force / self.body_weight - LAMBDA_BW) # Calculate the excess force relative to the body weight.
        
        impact = -W_IMPACT * excess ** 2 # Penalize the agent for exceeding the impact threshold (LAMBDA_BW) in terms of body weights. 
                                         # The penalty is quadratic to encourage the agent to avoid high impact forces.
        
        smooth = -W_SMOOTH * float(np.sum((action - self.prev_action) ** 2)) # Penalize the agent for large changes in the action values.

        return survival + impact + smooth

    def _failed(self):
        '''
        Terminal failure: toppled, or collapsed.
        Should return TERMINATED rather than TRUNCATED. 
        Running out of steps is a time limit, not a failure condition. 
        This way the agent actually learns to avoid falling over, rather than just learning to survive for a fixed number of steps.
        '''
        toppled = abs(self._leg_tilt()) > FALL_ANGLE
        collapsed = (self.data.xpos[self.body_bid][2]
                     < COLLAPSE_FRACTION * self.standing_height)
        return toppled or collapsed

    # ===================================================== rendering
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._recorder is None:
            from stance_env.viz import Recorder # from Muthu's work, this is the class that handles rendering the MuJoCo simulation to an RGB array
            self._recorder = Recorder(self.model)
        self.ground.update_visual(self.model)     # show the dent
        return self._recorder.capture(self.data)

    def close(self):
        if self._recorder is not None: 
            self._recorder.close()
            self._recorder = None
        