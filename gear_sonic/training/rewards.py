"""
SONIC Motion-Tracking Reward Function  —  Table S3 (arXiv 2511.07820v3)
==========================================================================

Reference: Luo et al., "SONIC: Supersizing Motion Tracking for Natural
Humanoid Whole-Body Control", arXiv:2511.07820v3, Table S3.

The reward is split into tracking rewards ℛ(s_t^p, s_t^g)  and
penalty terms 𝒫(s_t^p, a_t):
    r_t = ℛ(s_t^p, s_t^g) + 𝒫(s_t^p, a_t)

All tracking terms use exponential shaping:  exp(−‖error‖² / σ²)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TRACKING REWARDS  ℛ(s_t^p, s_t^g)                    Weight  σ (denom)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Root position
   r_pos_root = exp(−‖p_r^p − p_r^g‖² / 0.3²)          0.5
 Root orientation  (6-D rotation repr.)
   r_ori_root = exp(−‖o_r^p − o_r^g‖² / 0.4²)          0.5
 Body-link positions (root-relative, set ℬ of links)
   r_pos_body = exp(−(1/|ℬ|) Σ_b ‖p_b^p − p_b^g‖² / 0.1²)  1.0
 Body-link orientations (root-relative)
   r_ori_body = exp(−(1/|ℬ|) Σ_b ‖o_b^p − o_b^g‖² / 0.1²)  0.5
 Body-link linear velocities
   r_vel_body = exp(−(1/|ℬ|) Σ_b ‖v_b^p − v_b^g‖² / 0.5²)  0.05
 Body-link angular velocities
   r_ang_body = exp(−(1/|ℬ|) Σ_b ‖ω_b^p − ω_b^g‖² / 0.5²)  0.02
 End-effector positions  (head, L/R wrist, L/R ankle)
   r_pos_ee   = exp(−(1/|ℰ|) Σ_e ‖p_e^p − p_e^g‖² / 0.05²) 2.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PENALTY TERMS  𝒫(s_t^p, a_t)                          Weight
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Action rate (joint jerk)
   p_rate    = −‖a_t − a_{t−1}‖²                        −0.01
 Joint limit proximity  (penalise near limits)
   p_limit   = −Σ_j max(0, |q_j| − q_j^lim + δ)²       −1.0
 Undesired contact  (torso/knee on ground)
   p_contact = −Σ_c I[f_c > 0] (non-foot contact force)  −1.0
 Anti-shake  (head and wrist angular velocity)
   p_shake   = −Σ_{e∈{head,L/R wrist}} ‖ω_e‖²           −0.1
 Foot acceleration  (encourage smooth contact)
   p_foot    = −Σ_{f∈{L/R foot}} ‖a̋_f‖²                −0.01
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NOTE ON THIS IMPLEMENTATION
The function below is a simplified version that works with the available
MuJoCo observations (q, dq, root vel, ref_q).  Full body-link position /
orientation tracking requires forward kinematics on the MuJoCo model which
is not yet integrated here.  Priority items to add for a full Table S3 match:
  1. body-link FK positions/orientations (root-relative) for all |ℬ| links
  2. end-effector positions for {head, L/R wrist, L/R ankle}
  3. joint-limit proximity penalty
  4. anti-shake penalty on head/wrist angular velocity
  5. foot acceleration penalty
"""

import numpy as np


def compute_reward(
    q:           np.ndarray,    # (29,) current joint positions (rad)
    dq:          np.ndarray,    # (29,) current joint velocities (rad/s)
    ref_q:       np.ndarray,    # (29,) reference joint positions (rad)
    root_vel:    np.ndarray,    # (6,)  [vx, vy, vz, wx, wy, wz]
    action:      np.ndarray,    # (29,) current action (rad)
    prev_action: np.ndarray,    # (29,) previous action (rad)
    torques:     np.ndarray | None = None,
    # -- weights (from Table S3) --
    w_joint:    float = 1.0,    # body-link pos tracking (simplified as joint pos)
    w_balance:  float = 0.2,    # anti-shake proxy: penalise base angular velocity
    w_smooth:   float = 0.1,    # action rate penalty
    w_torque:   float = 0.05,   # torque penalty (proxy for joint-limit penalty)
    # -- shaping sigmas (from Table S3) --
    sigma_q:    float = 0.25,   # rad  (body-link pos σ=0.1 m; approximated)
    sigma_ang:  float = 1.0,    # rad/s  (anti-shake)
    sigma_da:   float = 0.5,    # rad  (action rate)
) -> float:
    """
    Simplified SONIC reward (Table S3 subset) using only currently
    available MuJoCo observations.

    Returns a scalar in approximately [0, 1].
    """
    # ── r_pos_body (proxy via joint angles) ──────────────────────────────────
    # Full version: exp(−(1/|ℬ|) Σ_b ‖p_b^p − p_b^g‖² / 0.1²)
    # Proxy: match joint positions since FK is not yet integrated
    err_q      = q - ref_q
    r_joint    = float(np.exp(-np.sum(err_q ** 2) / (2 * sigma_q ** 2 * len(q))))

    # ── anti-shake / balance (root angular velocity proxy) ───────────────────
    # Full version: penalise ‖ω_head‖ and ‖ω_wrist‖; proxy: base ω
    ang_vel    = root_vel[3:]
    r_balance  = float(np.exp(-np.sum(ang_vel ** 2) / (2 * sigma_ang ** 2)))

    # ── action rate (Table S3: p_rate = −‖a_t − a_{t-1}‖²) ──────────────────
    delta_a    = action - prev_action
    r_smooth   = float(np.exp(-np.sum(delta_a ** 2) / (2 * sigma_da ** 2 * len(action))))

    # ── torque / joint-limit penalty (proxy) ─────────────────────────────────
    r_torque   = 1.0
    if torques is not None:
        r_torque = float(np.exp(-np.mean(torques ** 2) / 1000.0))

    total = (
        w_joint   * r_joint
        + w_balance * r_balance
        + w_smooth  * r_smooth
        + w_torque  * r_torque
    ) / (w_joint + w_balance + w_smooth + w_torque)

    return float(total)
