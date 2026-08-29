"""
Side-by-side visualization of a SOMA-BVH -> motion_lib .pkl conversion:
   LEFT  : source SOMA-skeleton BVH motion (e.g. smpl_filtered_to_bvh/<name>.bvh)
   RIGHT : retargeted Unitree G1 (29 DOF) motion, loaded from the motion_lib
           `.pkl` produced by `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`
           (e.g. smpl_filtered_to_bvh_csv_robot/smpl_filtered_to_bvh_csv/<name>.pkl)

This is a thin wrapper around `visualize_soma_retarget.py`: it reuses that
module's BVH parser, G1 real-MJCF forward-kinematics (`g1_fk_batch`), and
rendering code UNCHANGED, but replaces its `pd.read_csv(csv_path)` DataFrame
source with a synthetic DataFrame built directly from the motion_lib pkl's
`dof` (T,29 radians, MuJoCo/MJCF actuator order) + `root_rot` (T,4 xyzw) +
`root_trans_offset` (T,3 meters), so `g1_fk()`'s existing
`row['root_translateX']` / `row[f'{jname}_dof']` lookups work unmodified.

Usage:
  .venv_sim/bin/python visualize_soma_retarget_robot_pkl.py \
      --bvh_dir /home/grease/ego_dataset/smpl_filtered_to_bvh \
      --pkl_dir /home/grease/ego_dataset/smpl_filtered_to_bvh_csv_robot/smpl_filtered_to_bvh_csv \
      --name "amass__ACCAD__Female1General_c3d__A1_-_Stand_poses" \
      --out data_visualization/soma_robot_pkl_check/A1_-_Stand.gif \
      --fps 50 --frame_step 4
"""
import argparse
import io
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import scipy.spatial.transform as sT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visualize_soma_retarget as V  # reuse BVH parser, G1 FK, rendering


# Same column order used by convert_soma_csv_to_motion_lib.py's
# BONES_CSV_JOINT_NAMES (MuJoCo/MJCF actuator order) -- `dof` column i
# corresponds to G1_JOINT_NAMES[i] + "_dof".
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def load_robot_pkl_as_df(pkl_path, name):
    """motion_lib .pkl entry -> DataFrame matching the retargeter CSV schema
    that `visualize_soma_retarget.g1_fk()` expects (root_translate{X,Y,Z} in
    cm, root_rotate{X,Y,Z} in degrees (intrinsic 'xyz' euler, matching how
    convert_soma_csv_to_motion_lib.load_bones_csv() builds the quat from
    that same convention), and "<joint>_dof" in degrees)."""
    d = joblib.load(pkl_path)
    # Files may be saved either as {name: entry} or as the entry directly.
    entry = d[name] if (isinstance(d, dict) and name in d) else (
        d if 'dof' in d else next(iter(d.values())))

    root_trans = np.asarray(entry['root_trans_offset'], dtype=np.float64)  # (T,3) meters
    root_quat_xyzw = np.asarray(entry['root_rot'], dtype=np.float64)       # (T,4) xyzw
    dof_rad = np.asarray(entry['dof'], dtype=np.float64)                  # (T,29) radians

    euler_deg = sT.Rotation.from_quat(root_quat_xyzw).as_euler('xyz', degrees=True)
    dof_deg = np.rad2deg(dof_rad)

    T = len(root_trans)
    cols = {
        'root_translateX': root_trans[:, 0] * 100.0,   # m -> cm
        'root_translateY': root_trans[:, 1] * 100.0,
        'root_translateZ': root_trans[:, 2] * 100.0,
        'root_rotateX': euler_deg[:, 0],
        'root_rotateY': euler_deg[:, 1],
        'root_rotateZ': euler_deg[:, 2],
    }
    n_dof = min(dof_deg.shape[1], len(G1_JOINT_NAMES))
    for i in range(n_dof):
        cols[f'{G1_JOINT_NAMES[i]}_dof'] = dof_deg[:, i]

    df = pd.DataFrame(cols, index=range(T))
    fps = entry.get('fps', None)
    return df, fps


def process_one(name, bvh_dir, pkl_dir, out_path, fps, frame_step):
    bvh_path = os.path.join(bvh_dir, f'{name}.bvh')
    pkl_path = os.path.join(pkl_dir, f'{name}.pkl')
    print(f'\n=== {name} ===')

    joints, offsets, channels, parents, frames = V.parse_bvh(bvh_path)
    df_g1, pkl_fps = load_robot_pkl_as_df(pkl_path, name)
    if pkl_fps:
        fps = pkl_fps
    n = min(len(frames), len(df_g1))
    sampled = list(range(0, n, frame_step))
    print(f'  BVH frames: {len(frames)}  robot-pkl rows: {len(df_g1)}  '
          f'sampled: {len(sampled)}  fps: {fps}')

    pos_soma = V.fk_batch(V.VIZ_JOINTS, joints, offsets, channels, parents, frames, sampled)
    foot_idx = [V.VIZ_JOINTS.index(n_) for n_ in ('LeftFoot', 'RightFoot')]
    pos_soma[:, :, 1] -= pos_soma[:, foot_idx, 1].min()

    pos_g1 = V.g1_fk_batch(V.G1_VIZ, df_g1, sampled)
    foot_g1_idx = [V.G1_VIZ.index('left_ankle_roll_link'), V.G1_VIZ.index('right_ankle_roll_link')]
    pos_g1[:, :, 1] -= pos_g1[:, foot_g1_idx, 1].min()

    lim_soma = V.axis_limits([pos_soma])
    lim_g1 = V.axis_limits([pos_g1])

    def render(fi):
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        V.draw_skeleton(ax1, pos_soma[fi], V.BONES, V.VIZ_JOINTS, V.C_SOMA)
        V.draw_skeleton(ax2, pos_g1[fi], V.G1_BONES, V.G1_VIZ, V.C_G1)
        V.style_ax(ax1, 'Source SOMA BVH', *lim_soma)
        V.style_ax(ax2, 'Retargeted G1 (motion_lib .pkl)', *lim_g1)
        frame_num = sampled[fi]
        plt.suptitle(f'{name}  ·  frame {frame_num}/{n}  ·  t={frame_num / fps:.2f}s',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        return fig

    N = len(sampled)
    gif_frames = []
    t0 = time.time()
    for fi in range(N):
        fig = render(fi)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=85, bbox_inches='tight')
        buf.seek(0)
        gif_frames.append(Image.open(buf).copy())
        plt.close(fig)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:],
                        duration=50, loop=0, optimize=False)
    print(f'  saved -> {out_path}  ({N} frames, {time.time() - t0:.0f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bvh_dir', required=True)
    ap.add_argument('--pkl_dir', required=True,
                     help="dir containing <name>.pkl motion_lib files "
                          "(output of convert_soma_csv_to_motion_lib.py)")
    ap.add_argument('--name', required=True)
    ap.add_argument('--out', required=True, help='output .gif path')
    ap.add_argument('--fps', type=float, default=50.0,
                     help='fallback fps if not present inside the pkl entry')
    ap.add_argument('--frame_step', type=int, default=4)
    args = ap.parse_args()

    process_one(args.name, args.bvh_dir, args.pkl_dir, args.out, args.fps, args.frame_step)


if __name__ == '__main__':
    main()
