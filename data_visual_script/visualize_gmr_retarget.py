"""
Visualize a GMR (General Motion Retargeting) retargeted G1 `.pkl` file, e.g.
    /home/grease/GMR/retargeted_g1_output/<name>.pkl

Unlike the `motion_lib` format (dof/root_trans_offset/root_rot, requiring a
full G1-MJCF forward-kinematics pass -- see `visualize_soma_retarget.py`),
GMR's own output already stores PRE-COMPUTED body positions directly:

    fps            : int
    root_pos       : (T, 3)      world root position, meters
    root_rot       : (T, 4)      world root quaternion (xyzw, scipy convention
                                  -- verified unit-norm)
    dof_pos        : (T, 29)     joint angles, radians (MuJoCo/MJCF order)
    local_body_pos : (T, 38, 3)  ROOT-LOCAL body positions, meters, Z-up
                                  (verified: feet sit at local z ~ -0.75m
                                  relative to pelvis at the local origin --
                                  i.e. already the exact MJCF world axis
                                  convention X=forward,Y=left,Z=up, just
                                  re-centered on the root -- no FK needed!)
    link_body_list : list[str]   38 G1 link names, in `local_body_pos`'s
                                  column order (a SUPERSET of the 30-body
                                  G1_CHAIN_MJCF used elsewhere in this repo:
                                  also includes toe/mocap/imu/hand sites).

Since `local_body_pos` is already computed, this script does NOT need any
FK -- it just draws the stored positions directly, floor-aligned (feet at
z=0) and heading-fixed to frame 0 (matching the convention used throughout
`visualize_soma_retarget.py`/`visualize_soma_bvh.py`).

Usage:
    .venv_sim/bin/python visualize_gmr_retarget.py \
        --pkl /home/grease/GMR/retargeted_g1_output/CMU__CMU__29__29_24_stageii.pkl \
        --out data_visualization/gmr_check/CMU__29_24_stageii.gif \
        --frame_step 4
"""
import argparse
import os
import time

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

C_G1 = '#66bb6a'

# Bones inferred from G1's real kinematic tree (superset of G1_CHAIN_MJCF in
# visualize_soma_retarget.py, plus GMR's extra toe/mocap/imu/hand sites).
BONES = [
    ('pelvis', 'left_hip_pitch_link'), ('left_hip_pitch_link', 'left_hip_roll_link'),
    ('left_hip_roll_link', 'left_hip_yaw_link'), ('left_hip_yaw_link', 'left_knee_link'),
    ('left_knee_link', 'left_ankle_pitch_link'), ('left_ankle_pitch_link', 'left_ankle_roll_link'),
    ('left_ankle_roll_link', 'left_toe_link'),
    ('pelvis', 'right_hip_pitch_link'), ('right_hip_pitch_link', 'right_hip_roll_link'),
    ('right_hip_roll_link', 'right_hip_yaw_link'), ('right_hip_yaw_link', 'right_knee_link'),
    ('right_knee_link', 'right_ankle_pitch_link'), ('right_ankle_pitch_link', 'right_ankle_roll_link'),
    ('right_ankle_roll_link', 'right_toe_link'),
    ('pelvis', 'pelvis_contour_link'),
    ('pelvis', 'waist_yaw_link'), ('waist_yaw_link', 'waist_roll_link'),
    ('waist_roll_link', 'torso_link'),
    ('torso_link', 'head_link'), ('head_link', 'head_mocap'), ('torso_link', 'imu_in_torso'),
    ('torso_link', 'left_shoulder_pitch_link'), ('left_shoulder_pitch_link', 'left_shoulder_roll_link'),
    ('left_shoulder_roll_link', 'left_shoulder_yaw_link'), ('left_shoulder_yaw_link', 'left_elbow_link'),
    ('left_elbow_link', 'left_wrist_roll_link'), ('left_wrist_roll_link', 'left_wrist_pitch_link'),
    ('left_wrist_pitch_link', 'left_wrist_yaw_link'), ('left_wrist_yaw_link', 'left_rubber_hand'),
    ('torso_link', 'right_shoulder_pitch_link'), ('right_shoulder_pitch_link', 'right_shoulder_roll_link'),
    ('right_shoulder_roll_link', 'right_shoulder_yaw_link'), ('right_shoulder_yaw_link', 'right_elbow_link'),
    ('right_elbow_link', 'right_wrist_roll_link'), ('right_wrist_roll_link', 'right_wrist_pitch_link'),
    ('right_wrist_pitch_link', 'right_wrist_yaw_link'), ('right_wrist_yaw_link', 'right_rubber_hand'),
]


def _quat_to_yaw(q_xyzw):
    x, y, z, w = q_xyzw
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def draw_skeleton(ax, pos_frame, names, color, lw=2.5):
    idx = {n: i for i, n in enumerate(names)}
    for b0, b1 in BONES:
        if b0 in idx and b1 in idx:
            p0, p1 = pos_frame[idx[b0]], pos_frame[idx[b1]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, lw=lw)
    ax.scatter(pos_frame[:, 0], pos_frame[:, 1], pos_frame[:, 2],
               color=color, s=15, depthshade=False)


def style_ax(ax, title, lim):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, 2 * lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('X (fwd)', fontsize=6); ax.set_ylabel('Y (left)', fontsize=6)
    ax.set_zlabel('Z (up)', fontsize=6)
    ax.tick_params(labelsize=5)
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.view_init(elev=12, azim=45)


def process_one(pkl_path, out_path, frame_step):
    name = os.path.splitext(os.path.basename(pkl_path))[0]
    print(f'\n=== {name} ===')
    d = joblib.load(pkl_path)
    fps = d.get('fps', 30)
    names = d['link_body_list']
    pos = d['local_body_pos'].astype(np.float64)  # (T, 38, 3), root-local, Z-up
    root_rot = d['root_rot']  # (T,4) xyzw
    T = len(pos)
    print(f'  frames: {T}  fps: {fps}  bodies: {len(names)}')

    # Floor-align: feet (toe links) at z=0.
    foot_idx = [names.index(n) for n in ('left_toe_link', 'right_toe_link')]
    pos = pos.copy()
    pos[:, :, 2] -= pos[:, foot_idx, 2].min()

    # Heading-fix to frame 0 (rotate every frame's LOCAL positions by the
    # SAME constant yaw, so the body's own turning is visible as motion
    # rather than being cancelled by a rotating "camera" -- same rationale
    # as g1_fk_batch()'s fixed_yaw in visualize_soma_retarget.py).
    yaw0 = _quat_to_yaw(root_rot[0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    Ryaw = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    pos = pos @ Ryaw.T

    sampled = list(range(0, T, frame_step))
    print(f'  sampled: {len(sampled)}')
    lim = float(np.percentile(np.abs(pos), 95)) * 1.3

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    canvas = fig.canvas
    suptitle = fig.suptitle('', fontsize=11, fontweight='bold')
    layout_done = [False]

    def render(fi):
        ax.clear()
        frame_idx = sampled[fi]
        draw_skeleton(ax, pos[frame_idx], names, C_G1)
        style_ax(ax, 'GMR Retargeted G1', lim)
        suptitle.set_text(f'{name}  ·  frame {frame_idx}/{T}  ·  t={frame_idx / fps:.2f}s')
        if not layout_done[0]:
            plt.tight_layout()
            layout_done[0] = True
        canvas.draw()
        w, h = canvas.get_width_height()
        renderer = canvas.get_renderer()
        rgba = np.asarray(renderer.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4).copy()
        return Image.fromarray(rgba, mode='RGBA').convert('RGB')

    N = len(sampled)
    gif_frames = []
    t0 = time.time()
    for fi in range(N):
        gif_frames.append(render(fi))
        if (fi + 1) % 20 == 0 or fi + 1 == N:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed if elapsed > 0 else 0
            eta = (N - fi - 1) / rate if rate > 0 else 0
            print(f'    [render] {fi + 1}/{N} frames  ({rate:.1f} fps, ETA {eta:.0f}s)',
                  flush=True)
    plt.close(fig)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:],
                        duration=50, loop=0, optimize=False)
    print(f'  saved -> {out_path}  ({N} frames, {time.time() - t0:.0f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', help='single GMR retargeted_g1_output/<name>.pkl')
    ap.add_argument('--out', help='output .gif path (used with --pkl)')
    ap.add_argument('--pkl_dir', help='directory of .pkl files (renders ALL of them)')
    ap.add_argument('--out_dir', help='output directory (used with --pkl_dir)')
    ap.add_argument('--frame_step', type=int, default=4)
    args = ap.parse_args()

    if args.pkl:
        if not args.out:
            ap.error('--out is required with --pkl')
        process_one(args.pkl, args.out, args.frame_step)
    elif args.pkl_dir:
        if not args.out_dir:
            ap.error('--out_dir is required with --pkl_dir')
        import glob
        pkl_files = sorted(glob.glob(os.path.join(args.pkl_dir, '*.pkl')))
        print(f'Found {len(pkl_files)} .pkl files in {args.pkl_dir}')
        n_ok, n_fail = 0, 0
        for i, pkl_path in enumerate(pkl_files):
            name = os.path.splitext(os.path.basename(pkl_path))[0]
            out_path = os.path.join(args.out_dir, f'{name}.gif')
            print(f'\n[{i + 1}/{len(pkl_files)}]', end='')
            try:
                process_one(pkl_path, out_path, args.frame_step)
                n_ok += 1
            except Exception as e:
                print(f'  [!] FAILED: {name}: {e}')
                n_fail += 1
        print(f'\n\nDone: {n_ok} succeeded, {n_fail} failed, out of {len(pkl_files)} total')
    else:
        ap.error('must specify either --pkl (single file) or --pkl_dir (batch)')


if __name__ == '__main__':
    main()
