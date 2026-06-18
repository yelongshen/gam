"""
Lightweight standalone G1 MuJoCo environment for SONIC PPO training.
No ROS, no Unitree SDK — pure mujoco-python bindings.

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

        mujoco.mj_resetData(self.model, self.data)
        # Start at reference pose
        q0 = self._ref_traj[0]
        self.data.qpos[self._qpos_idx] = q0
        self.data.qpos[2] = 0.793          # initial height (m)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray):
        """
        action : (29,) target joint positions in radians.
        Returns (obs, reward, done, info).
        """
        action = np.clip(action, self.q_lo, self.q_hi)

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

        # Phase (sin/cos of normalised position in trajectory)
        T = len(self._ref_traj)
        phase_angle = 2 * np.pi * self._step_idx / max(T, 1)
        phase = np.array([np.sin(phase_angle), np.cos(phase_angle)])

        return np.concatenate([q, dq, root_pos, rpy, root_vel, ref_q, ref_dq, phase]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> float:
        from gear_sonic.training.rewards import compute_reward
        q    = self.data.qpos[self._qpos_idx]
        dq   = self.data.qvel[self._qvel_idx]
        return compute_reward(
            q=q, dq=dq,
            ref_q=self._ref_at(self._step_idx - 1),
            root_vel=self.data.qvel[:6],
            action=action,
            prev_action=self._prev_action,
        )

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
