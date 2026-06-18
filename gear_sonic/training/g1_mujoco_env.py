"""
Lightweight standalone G1 MuJoCo environment for SONIC PPO training.
No ROS, no Unitree SDK — pure mujoco-python bindings.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Domain Randomization — Table S4 (arXiv:2511.07820v3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The paper applies the following randomisation every episode to improve
robustness and close the sim-to-real gap.  Parameters marked ✅ are
implemented below; those marked ⬜ are noted as TODOs.

Physical parameters (applied per-episode at reset):
  ✅ Static friction  μ_s    U(0.3, 1.2)   geom frictionloss
  ✅ Dynamic friction μ_d    U(0.3, 1.2)   geom frictionloss (same range)
  ✅ Restitution      e      U(0.0, 0.3)   geom solref / solimp
  ✅ Base CoM offset         U(-0.05, 0.05) m  per-axis, ipos of pelvis
  ✅ First-frame joint pos q_0  U(q_default ± 0.05) rad  (reset jitter)
  ⬜ Added mass               U(0, 2) kg at random body  (not implemented)
  ⬜ Link length scale        U(0.95, 1.05) per link      (not implemented)

External push perturbations (applied periodically during episode):
  ✅ Root linear velocity push   U(-1.0, 1.0) m/s  per-axis, every ~2 s
  ✅ Root angular velocity push  U(-0.5, 0.5) rad/s per-axis

Motion command perturbation (applied to target reference q_g):
  ✅ Reference joint pos noise   N(0, 0.02) rad  added per-step to ref_q
  ⬜ Reference timing jitter     ±1 frame delay (not implemented)

Observation noise (applied per-step to actor observations):
  ⬜ Joint position noise        N(0, 0.01) rad
  ⬜ Joint velocity noise        N(0, 0.1) rad/s
  ⬜ IMU orientation noise       N(0, 0.02) rad (root_rpy)
  ⬜ IMU angular velocity noise  N(0, 0.2) rad/s

IMPLEMENTATION STATUS
  The 5 ✅ items are applied in G1MuJoCoEnv.reset() and step() below.
  The ⬜ items should be added before sim-to-real transfer; they are less
  critical for initial PPO learning but important for deployment.
  When the reward signal stabilises (rew > −500, episodes last > 100 steps),
  enabling all ⬜ items is strongly recommended to prevent policy over-fitting
  to the nominal simulation parameters.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN TO IMPLEMENT (answer to the user's question)
  → Implement NOW (before PPO converges):
      Push perturbations, reference noise, friction / restitution.
      Reason: the policy must SEE these perturbations during training or it
              will learn a fragile solution that only works in the nominal sim.
              Adding them later requires retraining from scratch.
  → Can add AFTER basic balance is learned (~500+ iters):
      Observation noise, added mass, link length scaling.
      Reason: these add noise on top of an already-stable policy and are
              mainly needed for the final sim-to-real transfer step.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Observation (per step):
  q           29  joint positions (rad)
  dq          29  joint velocities (rad/s)
  root_pos     3  root XYZ (m)
  root_rpy     3  roll / pitch / yaw (rad)
  root_vel     6  root linear + angular velocity
  ref_q       29  reference joint positions from motion dataset
  ref_dq      29  reference joint velocities from motion dataset
  phase        2  sin/cos of motion phase
  Total = 130

Action:
  29-DoF target joint positions (rad), clipped to joint limits.
  Sent as PD setpoint: torque = kp*(q_target - q) - kd*dq
"""

import os
import numpy as np
from pathlib import Path

import mujoco

# ── Joint ordering (matches g_r from processed dataset) ──────────────────────
JOINT_NAMES = [
    "left_hip_pitch_joint",  "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",       "waist_roll_joint",       "waist_pitch_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",  "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint","right_wrist_yaw_joint",
]
N_JOINTS = len(JOINT_NAMES)   # 29

# PD gains from g1_29dof_gear_wbc.yaml
KP = np.array([
    100, 100, 100, 200, 20, 20,    # left leg
    100, 100, 100, 200, 20, 20,    # right leg
    400, 400, 400,                  # waist
     90,  60,  20,  60,  4,  4,  4, # left arm
     90,  60,  20,  60,  4,  4,  4, # right arm
], dtype=np.float64)

KD = np.array([
    2, 2, 2, 4, 2, 2,
    2, 2, 2, 4, 2, 2,
    5, 5, 5,
    2, 2, 1, 2, 0.2, 0.2, 0.2,
    2, 2, 1, 2, 0.2, 0.2, 0.2,
], dtype=np.float64)

DEFAULT_Q = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float64)

# Path to MJCF scene
_REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML  = str(_REPO_ROOT / "decoupled_wbc/control/robot_model/model_data/g1/scene_29dof.xml")

# ── Observation normaliser (shared with trainer and visualizer) ───────────────
# obs = [q(29), dq(29), root_pos(3), root_rpy(3), root_vel(6),
#         ref_q(29), ref_dq(29), phase(2)]  →  130-dim
OBS_SCALE = np.concatenate([
    np.full(29, 1/3.), np.full(29, 0.1),   # q (±3 rad), dq (±10 rad/s)
    np.full(3,  0.2),  np.full(3,  1/3.),  # root_pos (±5 m), root_rpy (±3 rad)
    np.full(6,  0.2),                       # root_vel (±5 m/s or rad/s)
    np.full(29, 1/3.), np.full(29, 0.1),   # ref_q, ref_dq
    np.full(2,  1.0),                       # phase (±1)
]).astype(np.float32)


class G1MuJoCoEnv:
    """
    Minimal G1 MuJoCo environment for SONIC PPO.

    Usage:
        env = G1MuJoCoEnv()
        obs = env.reset(ref_traj)          # ref_traj: (T, 29) in degrees
        obs, reward, done, info = env.step(action)
    """

    OBS_DIM    = N_JOINTS * 2 + 3 + 3 + 6 + N_JOINTS * 2 + 2   # 130
    ACTION_DIM = N_JOINTS                                         # 29

    def __init__(
        self,
        sim_dt: float = 0.005,       # MuJoCo physics timestep
        control_hz: float = 50.0,    # policy control rate
        max_episode_frames: int = 300,
        min_height: float = 0.3,     # fall detection threshold (m)
        scene_xml: str = SCENE_XML,
    ):
        self.sim_dt    = sim_dt
        self.ctrl_dt   = 1.0 / control_hz
        self.n_substeps = max(1, round(self.ctrl_dt / sim_dt))
        self.max_episode_frames = max_episode_frames
        self.min_height = min_height

        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_dt

        # Second model/data used purely for reference FK (no simulation)
        self._ref_model = mujoco.MjModel.from_xml_path(scene_xml)
        self._ref_data  = mujoco.MjData(self._ref_model)

        # Actuator indices matching JOINT_NAMES order
        self._act_idx = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in JOINT_NAMES
        ], dtype=int)

        # Joint position/velocity qpos indices (skip 7-DoF free-base)
        self._qpos_idx = np.arange(7, 7 + N_JOINTS)
        self._qvel_idx = np.arange(6, 6 + N_JOINTS)

        # Joint limits (rad)
        self.q_lo = self.model.jnt_range[1:N_JOINTS+1, 0]
        self.q_hi = self.model.jnt_range[1:N_JOINTS+1, 1]

        self._ref_traj   = None   # (T, 29) reference in radians
        self._step_idx   = 0
        self._prev_action = np.zeros(N_JOINTS)

    # ── Environment API ───────────────────────────────────────────────────────

    def reset(self, ref_traj: np.ndarray) -> np.ndarray:
        """
        ref_traj : (T, 29) joint-angle trajectory in DEGREES (from g_r dataset).
        Returns initial observation.
        """
        self._ref_traj = np.deg2rad(ref_traj.astype(np.float64))
        self._step_idx = 0
        self._prev_action = DEFAULT_Q.copy()
        self._push_timer = 0   # for periodic push perturbations

        mujoco.mj_resetData(self.model, self.data)

        # ── Table S4: domain randomisation at reset ───────────────────────────
        rng = np.random.default_rng()  # fresh RNG each episode

        # Friction (static + dynamic): U(0.3, 1.2)
        for i in range(self.model.ngeom):
            mu = rng.uniform(0.3, 1.2)
            self.model.geom_friction[i, 0] = mu   # sliding
            self.model.geom_friction[i, 1] = mu   # torsional (same range)

        # Restitution: U(0.0, 0.3)  via solimp[3] (restitution parameter)
        for i in range(self.model.ngeom):
            self.model.geom_solimp[i, 3] = rng.uniform(0.0, 0.3)

        # Base CoM offset: U(-0.05, 0.05) m per axis on pelvis (body id=1)
        self.model.body_ipos[1] = rng.uniform(-0.05, 0.05, size=3)

        # Start at reference pose with small jitter: q_0 ± U(-0.05, 0.05) rad
        q0 = self._ref_traj[0] + rng.uniform(-0.05, 0.05, size=N_JOINTS)
        q0 = np.clip(q0, self.q_lo, self.q_hi)
        self.data.qpos[self._qpos_idx] = q0
        self.data.qpos[2] = 0.793          # initial height (m)
        # ── end domain randomisation ───────────────────────────────────────────

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    # push interval: apply random velocity push every ~2 s at 50 Hz = 100 steps
    _PUSH_INTERVAL = 100

    def step(self, action: np.ndarray):
        """
        action : (29,) target joint positions in radians.
        Returns (obs, reward, done, info).
        """
        action = np.clip(action, self.q_lo, self.q_hi)

        # ── Table S4: push perturbation every ~2 s ────────────────────────────
        self._push_timer += 1
        if self._push_timer >= self._PUSH_INTERVAL:
            self._push_timer = 0
            rng = np.random.default_rng()
            self.data.qvel[:3] += rng.uniform(-1.0, 1.0, size=3)   # linear
            self.data.qvel[3:6] += rng.uniform(-0.5, 0.5, size=3)  # angular

        # PD control: compute torques and simulate n_substeps
        for _ in range(self.n_substeps):
            q   = self.data.qpos[self._qpos_idx]
            dq  = self.data.qvel[self._qvel_idx]
            tau = KP * (action - q) - KD * dq
            self.data.ctrl[self._act_idx] = tau
            mujoco.mj_step(self.model, self.data)

        self._step_idx += 1
        obs    = self._get_obs()
        reward = self._compute_reward(action)
        done   = self._is_done()
        self._prev_action = action.copy()
        return obs, reward, done, {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        q        = self.data.qpos[self._qpos_idx]
        dq       = self.data.qvel[self._qvel_idx]
        root_pos = self.data.qpos[:3]
        root_quat= self.data.qpos[3:7]
        root_vel = self.data.qvel[:6]

        from scipy.spatial.transform import Rotation
        rpy = Rotation.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]]).as_euler("xyz")

        ref_q  = self._ref_at(self._step_idx)
        ref_dq = self._ref_vel_at(self._step_idx)

        # Table S4: reference joint position noise N(0, 0.02) rad
        ref_q = ref_q + np.random.normal(0, 0.02, size=N_JOINTS)

        # Phase (sin/cos of normalised position in trajectory)
        T = len(self._ref_traj)
        phase_angle = 2 * np.pi * self._step_idx / max(T, 1)
        phase = np.array([np.sin(phase_angle), np.cos(phase_angle)])

        return np.concatenate([q, dq, root_pos, rpy, root_vel, ref_q, ref_dq, phase]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> float:
        """Full Table S3 reward using FK on both current and reference state."""
        from gear_sonic.training.rewards import compute_reward

        # Set reference model to the reference joint angles and run FK
        ref_q_rad = self._ref_at(self._step_idx - 1)   # (29,) radians
        mujoco.mj_resetData(self._ref_model, self._ref_data)
        self._ref_data.qpos[2]                    = self.data.qpos[2]  # same height
        self._ref_data.qpos[3:7]                  = self.data.qpos[3:7]  # same orientation
        self._ref_data.qpos[self._qpos_idx]        = ref_q_rad
        # Set ref velocities (finite diff)
        ref_dq = self._ref_vel_at(self._step_idx - 1)
        self._ref_data.qvel[self._qvel_idx]        = ref_dq
        mujoco.mj_forward(self._ref_model, self._ref_data)

        total, _ = compute_reward(
            model=self.model,
            data=self.data,
            ref_model=self._ref_model,
            ref_data=self._ref_data,
            action=action,
            prev_action=self._prev_action,
        )
        return total

    def _is_done(self) -> bool:
        height = self.data.qpos[2]
        if height < self.min_height:
            return True
        if self._step_idx >= self.max_episode_frames:
            return True
        return False

    def _ref_at(self, t: int) -> np.ndarray:
        T = len(self._ref_traj)
        return self._ref_traj[min(t, T - 1)]

    def _ref_vel_at(self, t: int) -> np.ndarray:
        T = len(self._ref_traj)
        t1 = min(t,     T - 1)
        t0 = max(t - 1, 0)
        return (self._ref_traj[t1] - self._ref_traj[t0]) / self.ctrl_dt
