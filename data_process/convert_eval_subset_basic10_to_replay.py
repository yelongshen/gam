"""
Convert the 10 "basic locomotion" eval_subset clips (robot/*.pkl, IsaacLab
motion_lib schema: root_trans_offset/root_rot/dof/pose_aa/smpl_joints) into the
gear_sonic_deploy/reference/ motion-replay CSV format expected by
convert_motions.py (joint_pos/joint_vel/body_pos_w/body_quat_w/body_lin_vel_w/
body_ang_vel_w), by running forward kinematics through the G1 MuJoCo model.

WHY THIS IS NEEDED
------------------
convert_motions.py expects a dict of
  motion_name -> {joint_pos (T,29), joint_vel (T,29),
                   body_pos_w (T,14,3), body_quat_w (T,14,4),
                   body_lin_vel_w (T,14,3), body_ang_vel_w (T,14,3)}
(this is what /home/grease/GR00T-WholeBodyControl/data/smpl_filtered/*.pkl
already contains, per model_eval/build_eval_set.sh's existing usage).

The eval_subset clips (/home/grease/ego_dataset/eval_subset/robot/*.pkl) are in
a DIFFERENT (upstream, motion_lib) schema:
  {root_trans_offset (T,3), root_rot (T,4, xyzw), dof (T,29), pose_aa (T,30,3),
   smpl_joints (T,24,3), fps}
missing joint_vel and all body_* arrays entirely. This script fills that gap:
  - joint_vel: finite-difference of `dof` at the clip's fps.
  - body_pos_w / body_quat_w: forward-kinematics via MuJoCo (free-base G1
    model, g1_29dof_with_hand.xml) at each frame, reading `xpos`/`xquat` for a
    chosen set of 14 representative bodies (see BODY_NAMES below -- NOT
    guaranteed to be the same 14 bodies/order used to generate the original
    smpl_filtered pkls, since that indexing is internal to the external
    GR00T-WholeBodyControl motion_lib code and not available here; this is a
    self-consistent, independently-chosen substitute good enough for motion
    replay/reference purposes).
  - body_lin_vel_w / body_ang_vel_w: finite-difference of the FK'd positions
    and (via rotvec) orientations.

ASSUMPTIONS (flagged explicitly, verify before relying on this for anything
beyond basic visual/replay sanity checks)
-----------------------------------------
  - `dof` is already in the same 29-DOF ordering as MJ_JOINT_NAMES /
    g1_params.JOINT_NAMES (the "hardware order" used throughout this repo's
    sim2real tooling) -- this matches the convention seen in
    sim2real_phaseC1_onestep_prediction.py and g1_params.py.
  - `root_rot` is stored as an (x, y, z, w) quaternion (the IsaacGym/PhysX/
    motion_lib convention), converted here to MuJoCo's (w, x, y, z) free-joint
    convention.

USAGE
-----
  .venv_sim/bin/python data_process/convert_eval_subset_basic10_to_replay.py
"""
import os
import sys

import joblib
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_params import JOINT_NAMES, ISAACLAB_TO_MUJOCO, DEFAULT_ANGLES  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BODY_XML = os.path.join(REPO, 'gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml')
SRC_DIR = '/home/grease/ego_dataset/eval_subset/robot'
OUT_DIR = os.path.join(REPO, 'gear_sonic_deploy/reference/evaluation_set/Basic_Locomotion_10')
CONVERT_SCRIPT = os.path.join(REPO, 'gear_sonic_deploy/reference/convert_motions.py')

MOTION_NAMES = [
    'arc_walk_left_loop_001__A030',
    'change_idle_right_to_idle_001__A037',
    'jog_ff_loop_180_R_004__A079',
    'jog_ff_loop_360_004__A182',
    'jog_forward_stop_001__A038_M',
    'turn_jog_360_R_001__A234_M',
    'walk_180_R_003__A332_M',
    'walk_backward_start_001__A030_M',
    'walk_hands_on_back_loop_003__A053_M',
    'walk_sideway_045_stop_005__A042_M',
]

# Simplest 4 of the 10 (no turning/arc/loop/backward-locomotion content per
# clip name), used as the default --clips selection when SIMPLE_ONLY is set.
SIMPLE_MOTION_NAMES = [
    'change_idle_right_to_idle_001__A037',   # literally idle -> idle
    'walk_backward_start_001__A030_M',       # starts from rest ("start")
    'jog_forward_stop_001__A038_M',          # ends at rest ("stop")
    'walk_sideway_045_stop_005__A042_M',     # ends at rest ("stop")
]

# Standing-height used for the settle pad (matches the observed pelvis height
# across these clips, ~0.79m for standing/idle frames).
SETTLE_ROOT_HEIGHT = 0.793

# 14 representative bodies (self-consistent choice; see docstring caveat)
BODY_NAMES = [
    'pelvis',
    'torso_link',
    'left_hip_pitch_link', 'right_hip_pitch_link',
    'left_knee_link', 'right_knee_link',
    'left_ankle_roll_link', 'right_ankle_roll_link',
    'left_shoulder_pitch_link', 'right_shoulder_pitch_link',
    'left_elbow_link', 'right_elbow_link',
    'left_wrist_yaw_link', 'right_wrist_yaw_link',
]
assert len(BODY_NAMES) == 14

MJ_JOINT_NAMES = [n.lower().replace('L_', 'left_').replace('R_', 'right_') for n in JOINT_NAMES]
# g1_params.JOINT_NAMES uses short names (L_hip_pitch); MuJoCo XML uses full
# joint names (left_hip_pitch_joint). Build the explicit mapping instead of
# guessing via string munging, to avoid silent mismatches:
MJ_JOINT_FULL_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
    'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint',
    'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]
assert len(MJ_JOINT_FULL_NAMES) == 29


def xyzw_to_wxyz(q):
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def yaw_only_quat_wxyz(q_wxyz):
    """Extract the yaw (about world z) component of a (w,x,y,z) quaternion,
    dropping roll/pitch -- used to make settle-pad frames stand upright while
    still facing the same direction as the real motion's first/last frame."""
    w, x, y, z = q_wxyz
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def smoothstep(n):
    t = np.linspace(0.0, 1.0, n)
    return 3 * t ** 2 - 2 * t ** 3


def add_settle_pad(dof_mj, root_trans, root_rot_wxyz, fps, settle_seconds):
    """Prepend/append a smooth blend to/from a neutral standing pose
    (DEFAULT_ANGLES, upright yaw-only orientation, SETTLE_ROOT_HEIGHT), so the
    replay genuinely starts and ends standing regardless of what the raw
    clip's own first/last frame looks like (rather than just holding the
    first frame, as stream_clip_mode2.py's --settle does)."""
    n = max(1, int(round(settle_seconds * fps)))
    w_in = smoothstep(n)[:, None]     # (n,1), 0->1
    w_out = 1.0 - smoothstep(n)[:, None]  # (n,1), 1->0

    yaw0 = yaw_only_quat_wxyz(root_rot_wxyz[0])
    yawN = yaw_only_quat_wxyz(root_rot_wxyz[-1])
    root_stand0 = np.array([root_trans[0, 0], root_trans[0, 1], SETTLE_ROOT_HEIGHT])
    root_standN = np.array([root_trans[-1, 0], root_trans[-1, 1], SETTLE_ROOT_HEIGHT])

    dof_in = DEFAULT_ANGLES[None, :] * (1 - w_in) + dof_mj[0][None, :] * w_in
    dof_out = dof_mj[-1][None, :] * w_out + DEFAULT_ANGLES[None, :] * (1 - w_out)

    root_trans_in = root_stand0[None, :] * (1 - w_in) + root_trans[0][None, :] * w_in
    root_trans_out = root_trans[-1][None, :] * w_out + root_standN[None, :] * (1 - w_out)

    # slerp-free linear+renormalize blend is fine here since yaw0/yawN and the
    # real first/last-frame orientation are typically close (settle pad is
    # only correcting residual roll/pitch lean, not a large rotation)
    root_rot_in = yaw0[None, :] * (1 - w_in) + root_rot_wxyz[0][None, :] * w_in
    root_rot_in /= np.linalg.norm(root_rot_in, axis=1, keepdims=True)
    root_rot_out = root_rot_wxyz[-1][None, :] * w_out + yawN[None, :] * (1 - w_out)
    root_rot_out /= np.linalg.norm(root_rot_out, axis=1, keepdims=True)

    dof_padded = np.concatenate([dof_in, dof_mj, dof_out], axis=0)
    root_trans_padded = np.concatenate([root_trans_in, root_trans, root_trans_out], axis=0)
    root_rot_padded = np.concatenate([root_rot_in, root_rot_wxyz, root_rot_out], axis=0)
    return dof_padded, root_trans_padded, root_rot_padded


def quat_to_rotvec(q_wxyz):
    """Small-angle-safe quaternion (w,x,y,z) -> rotation vector, batched."""
    w = np.clip(q_wxyz[..., 0], -1.0, 1.0)
    xyz = q_wxyz[..., 1:]
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.clip(1.0 - w * w, 1e-12, None))
    axis = xyz / sin_half[..., None]
    return axis * angle[..., None]


def build_fk_model():
    return mujoco.MjModel.from_xml_path(BODY_XML)


def fk_convert(motion_name, settle_seconds=0.0):
    src_path = os.path.join(SRC_DIR, motion_name + '.pkl')
    raw = joblib.load(src_path)[motion_name]
    root_trans = raw['root_trans_offset'].astype(np.float64)   # (T,3)
    root_rot_xyzw = raw['root_rot'].astype(np.float64)          # (T,4)
    dof_isaaclab = raw['dof'].astype(np.float64)                 # (T,29), IsaacLab joint order
    # `dof` comes from IsaacLab's motion_lib and is in IsaacLab joint order,
    # NOT the hardware/MuJoCo order used by MJ_JOINT_FULL_NAMES below (this is
    # the exact same reordering g1_params.action_to_q_target() applies to
    # `action` -- see g1_params.py's module docstring). Without this, arm/hand
    # joints in particular end up visibly wrong since IsaacLab's joint
    # ordering interleaves legs/waist/arms differently from hardware order.
    dof = dof_isaaclab[:, ISAACLAB_TO_MUJOCO]                    # (T,29), hardware/MuJoCo order
    fps = float(raw['fps'])
    dt = 1.0 / fps

    root_rot_wxyz = xyzw_to_wxyz(root_rot_xyzw)

    if settle_seconds > 0:
        dof, root_trans, root_rot_wxyz = add_settle_pad(dof, root_trans, root_rot_wxyz, fps, settle_seconds)

    T = dof.shape[0]

    model = build_fk_model()
    data = mujoco.MjData(model)

    dof_adr = {}
    for jname in MJ_JOINT_FULL_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dof_adr[jname] = model.jnt_qposadr[jid]

    body_id = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in BODY_NAMES]

    body_pos_w = np.zeros((T, 14, 3))
    body_quat_w = np.zeros((T, 14, 4))   # wxyz, matching MuJoCo's convention

    for t in range(T):
        data.qpos[0:3] = root_trans[t]
        data.qpos[3:7] = root_rot_wxyz[t]
        for j, jname in enumerate(MJ_JOINT_FULL_NAMES):
            data.qpos[dof_adr[jname]] = dof[t, j]
        mujoco.mj_forward(model, data)
        for k, bid in enumerate(body_id):
            body_pos_w[t, k] = data.xpos[bid]
            body_quat_w[t, k] = data.xquat[bid]   # already wxyz

    # finite-difference velocities (central difference, edge frames use one-sided)
    joint_vel = np.gradient(dof, dt, axis=0)
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0)

    body_ang_vel_w = np.zeros_like(body_lin_vel_w)
    rotvec = quat_to_rotvec(body_quat_w)          # (T,14,3), NOT unwrap-safe across large rotations
    body_ang_vel_w[1:-1] = (rotvec[2:] - rotvec[:-2]) / (2 * dt)
    body_ang_vel_w[0] = (rotvec[1] - rotvec[0]) / dt
    body_ang_vel_w[-1] = (rotvec[-1] - rotvec[-2]) / dt

    return {
        'joint_pos': dof.astype(np.float32),
        'joint_vel': joint_vel.astype(np.float32),
        'body_pos_w': body_pos_w.astype(np.float32),
        'body_quat_w': body_quat_w.astype(np.float32),
        'body_lin_vel_w': body_lin_vel_w.astype(np.float32),
        'body_ang_vel_w': body_ang_vel_w.astype(np.float32),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--simple-only', action='store_true',
                     help='convert only the 4 simplest clips (SIMPLE_MOTION_NAMES) instead of all 10')
    ap.add_argument('--settle-seconds', type=float, default=1.0,
                     help='seconds to blend to/from a neutral standing pose at the start/end '
                          'of each clip (0 disables settle padding)')
    ap.add_argument('--out-dir', type=str, default=OUT_DIR)
    args = ap.parse_args()

    names = SIMPLE_MOTION_NAMES if args.simple_only else MOTION_NAMES
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    tmp_pkl_dir = os.path.join(out_dir, '_tmp_converted_pkls')
    os.makedirs(tmp_pkl_dir, exist_ok=True)

    for name in names:
        print(f"=== {name} (settle={args.settle_seconds}s) ===")
        motion_data = fk_convert(name, settle_seconds=args.settle_seconds)
        tmp_pkl = os.path.join(tmp_pkl_dir, name + '.pkl')
        joblib.dump({name: motion_data}, tmp_pkl)
        print(f"  wrote intermediate pkl: {tmp_pkl}")

        # reuse convert_motions.py's CSV writer via subprocess, so the output
        # format is byte-for-byte identical to the existing evaluation_set/*
        os.system(f"{sys.executable} {CONVERT_SCRIPT} {tmp_pkl} {out_dir}")

    print(f"\nDone. Output in: {out_dir}/")
    print("(intermediate per-motion pkls kept in _tmp_converted_pkls/ for inspection/rerun)")


if __name__ == '__main__':
    main()
