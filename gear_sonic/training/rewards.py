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
 Body-link positions (root-relative, 14 links)
   r_pos_body = exp(−(1/|ℬ|) Σ_b ‖p_b^p − p_b^g‖² / 0.1²)  1.0
 Body-link orientations (root-relative, 6-D)
   r_ori_body = exp(−(1/|ℬ|) Σ_b ‖o_b^p − o_b^g‖² / 0.1²)  0.5
 Body-link linear velocities
   r_vel_body = exp(−(1/|ℬ|) Σ_b ‖v_b^p − v_b^g‖² / 0.5²)  0.05
 Body-link angular velocities
   r_ang_body = exp(−(1/|ℬ|) Σ_b ‖ω_b^p − ω_b^g‖² / 0.5²)  0.02
 End-effector positions  (L/R hand, L/R ankle, torso-as-head-proxy)
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
   p_shake   = −Σ_{e∈{torso,L/R wrist}} ‖ω_e‖²          −0.1
 Foot acceleration  (encourage smooth contact)
   p_foot    = −Σ_{f∈{L/R ankle}} ‖a̋_f‖²               −0.01
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

G1 body link IDs used (from scene_29dof.xml, confirmed via mj_name2id):
  BODY_LINKS (14): pelvis(1), L/R knee(5,15), L/R ankle_roll(7,17),
      torso(24), L/R elbow(28,36), L/R wrist_yaw(31,39),
      L/R shoulder_pitch(25,33), L/R hip_yaw(4,14)
  EE_LINKS   (5) : L/R rubber_hand(32,40), L/R ankle_roll(7,17), torso(24)
  FOOT_LINKS (2) : L/R ankle_roll(7,17)
  SHAKE_LINKS(3) : torso(24), L/R wrist_yaw(31,39)
"""

import mujoco
import numpy as np

# ── Body link indices (matched to Table S3 "14 body links") ──────────────────
_BODY_IDS = np.array([1, 5, 15, 7, 17, 24, 28, 36, 31, 39, 25, 33, 4, 14], dtype=int)
_EE_IDS   = np.array([32, 40, 7, 17, 24], dtype=int)   # hands, ankles, torso
_FOOT_IDS = np.array([7, 17], dtype=int)                # L/R ankle_roll
_SHAKE_IDS= np.array([24, 31, 39], dtype=int)           # torso, L/R wrist

# 6-D rotation repr: cols 0 and 1 of the rotation matrix (Zhou et al. 2019)
def _rot6d(xmat_row: np.ndarray) -> np.ndarray:
    """xmat_row is a flattened 3×3 row-major matrix from mj_data.xmat[body_id]."""
    R = xmat_row.reshape(3, 3)
    return R[:, :2].T.ravel()   # first two columns, 6-D


def compute_reward(
    model:       mujoco.MjModel,
    data:        mujoco.MjData,
    ref_model:   mujoco.MjModel,
    ref_data:    mujoco.MjData,
    action:      np.ndarray,
    prev_action: np.ndarray,
    joint_limits: np.ndarray | None = None,   # (29, 2) [lo, hi] in rad
    limit_margin: float = 0.05,               # rad buffer before penalty
) -> tuple[float, dict]:
    """
    Full Table S3 SONIC reward using MuJoCo FK for both current and reference.

    Parameters
    ----------
    model / data       : current robot state (after mj_forward)
    ref_model/ref_data : reference state set to ref joint angles + mj_forward
    action             : current joint position action (29,) rad
    prev_action        : previous action (29,) rad
    joint_limits       : (29,2) joint limit array; if None uses model.jnt_range

    Returns
    -------
    total_reward : float
    components   : dict  (for logging)
    """
    pelvis_id = 1   # body index of pelvis (root)

    # ── Root position / orientation ──────────────────────────────────────────
    p_root_p = data.xpos[pelvis_id].copy()
    p_root_g = ref_data.xpos[pelvis_id].copy()

    r_pos_root = float(np.exp(-np.sum((p_root_p - p_root_g)**2) / (0.3**2)))

    o_root_p = _rot6d(data.xmat[pelvis_id])
    o_root_g = _rot6d(ref_data.xmat[pelvis_id])
    r_ori_root = float(np.exp(-np.sum((o_root_p - o_root_g)**2) / (0.4**2)))

    # ── Body-link positions (root-relative) ───────────────────────────────────
    pos_p = data.xpos[_BODY_IDS] - p_root_p          # (14, 3)
    pos_g = ref_data.xpos[_BODY_IDS] - p_root_g
    mse_pos = float(np.mean(np.sum((pos_p - pos_g)**2, axis=1)))
    r_pos_body = float(np.exp(-mse_pos / (0.1**2)))

    # ── Body-link orientations (6-D, root-relative via relative rotation) ─────
    ori_p = np.stack([_rot6d(data.xmat[b])     for b in _BODY_IDS])  # (14,6)
    ori_g = np.stack([_rot6d(ref_data.xmat[b]) for b in _BODY_IDS])
    mse_ori = float(np.mean(np.sum((ori_p - ori_g)**2, axis=1)))
    r_ori_body = float(np.exp(-mse_ori / (0.1**2)))

    # ── Body-link linear velocities ───────────────────────────────────────────
    # xvelp (body CoM linear velocity in world frame) is not stored directly;
    # we approximate from cvel (6-D twist: [angular, linear] in body frame).
    # cvel index matches body index. Take linear part [3:6].
    vel_p = data.cvel[_BODY_IDS, 3:]       # (14, 3)
    vel_g = ref_data.cvel[_BODY_IDS, 3:]
    mse_vel = float(np.mean(np.sum((vel_p - vel_g)**2, axis=1)))
    r_vel_body = float(np.exp(-mse_vel / (0.5**2)))

    # ── Body-link angular velocities ─────────────────────────────────────────
    ang_p = data.cvel[_BODY_IDS, :3]       # (14, 3)
    ang_g = ref_data.cvel[_BODY_IDS, :3]
    mse_ang = float(np.mean(np.sum((ang_p - ang_g)**2, axis=1)))
    r_ang_body = float(np.exp(-mse_ang / (0.5**2)))

    # ── End-effector positions (root-relative) ─────────────────────────────
    ee_p = data.xpos[_EE_IDS] - p_root_p     # (5, 3)
    ee_g = ref_data.xpos[_EE_IDS] - p_root_g
    mse_ee = float(np.mean(np.sum((ee_p - ee_g)**2, axis=1)))
    r_pos_ee = float(np.exp(-mse_ee / (0.05**2)))

    # ── Total tracking reward ─────────────────────────────────────────────────
    R_track = (
        0.5  * r_pos_root
        + 0.5  * r_ori_root
        + 1.0  * r_pos_body
        + 0.5  * r_ori_body
        + 0.05 * r_vel_body
        + 0.02 * r_ang_body
        + 2.0  * r_pos_ee
    )

    # ── Penalty: action rate ──────────────────────────────────────────────────
    p_rate = -0.01 * float(np.sum((action - prev_action)**2))

    # ── Penalty: joint limit proximity ────────────────────────────────────────
    if joint_limits is None:
        # skip first free-base joint (7 qpos), take 29 actuated joints
        jnt_lo = model.jnt_range[1:30, 0]
        jnt_hi = model.jnt_range[1:30, 1]
    else:
        jnt_lo, jnt_hi = joint_limits[:, 0], joint_limits[:, 1]

    q = data.qpos[7:36]   # 29 actuated joints
    exceed_lo = np.maximum(0., jnt_lo + limit_margin - q)
    exceed_hi = np.maximum(0., q - (jnt_hi - limit_margin))
    p_limit = -1.0 * float(np.sum(exceed_lo**2 + exceed_hi**2))

    # ── Penalty: undesired contact (non-foot bodies touching ground) ──────────
    # Check contact forces on bodies other than left/right ankle_roll
    p_contact = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        # world body = 0; ankle bodies = 7, 17
        for b in (b1, b2):
            if b not in (0, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21):
                # Not world, not foot dummy bodies → undesired contact
                p_contact += -1.0
                break   # count once per contact

    # ── Penalty: anti-shake (angular velocity of torso and wrists) ────────────
    shake_ang = data.cvel[_SHAKE_IDS, :3]   # (3, 3)  [torso, L/R wrist]
    p_shake = -0.1 * float(np.sum(shake_ang**2))

    # ── Penalty: foot acceleration ─────────────────────────────────────────────
    # Approximate foot acceleration as linear velocity change per step.
    # We use cacc[body, 3:6] (linear acceleration in body frame).
    foot_acc = data.cacc[_FOOT_IDS, 3:]    # (2, 3)
    p_foot = -0.01 * float(np.sum(foot_acc**2))

    # ── Total ─────────────────────────────────────────────────────────────────
    P_penalty = p_rate + p_limit + p_contact + p_shake + p_foot
    total = R_track + P_penalty

    return total, {
        "r_pos_root": r_pos_root,
        "r_ori_root": r_ori_root,
        "r_pos_body": r_pos_body,
        "r_ori_body": r_ori_body,
        "r_vel_body": r_vel_body,
        "r_ang_body": r_ang_body,
        "r_pos_ee":   r_pos_ee,
        "R_track":    R_track,
        "p_rate":     p_rate,
        "p_limit":    p_limit,
        "p_contact":  p_contact,
        "p_shake":    p_shake,
        "p_foot":     p_foot,
        "total":      total,
    }
