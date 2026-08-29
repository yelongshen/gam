"""
Standalone, SINGLE-PANEL visualization of a `motion_lib`-format robot
`.pkl` (dof/root_trans_offset/root_rot), with NO paired BVH or SMPL source
required -- fills the gap that no other script in this repo covers (see
inventory below).

Existing scripts and why none of them do this:
  - visualize_soma_retarget.py            requires --bvh_dir + retargeter CSV
  - visualize_soma_retarget_robot_pkl.py   still requires --bvh_dir (left panel)
  - visualize_smpl_robot_pair.py           requires a matching smpl/<name>.pkl
  - visualize_gmr_retarget.py              standalone, but for GMR's OWN raw
                                           schema (local_body_pos/root_pos/
                                           dof_pos), NOT motion_lib's own
                                           (dof/root_trans_offset/root_rot)
  - gear_sonic_deploy/visualize_motion.py  interactive MuJoCo viewer, expects
                                           the CSV reference-motion folder
                                           format (joint_pos.csv etc), not .pkl

Useful for isolating whether a "looks wrong" issue is in the G1 FK/data
itself, vs. bugs in a paired-comparison script's own plotting code (this is
exactly how the axis-swizzle bug in visualize_smpl_robot_pair.py's G1 panel
was isolated).

Reuses the SAME validated pipeline as visualize_smpl_robot_pair.py:
  motion_lib .pkl -> load_robot_pkl_as_df() -> V.g1_fk_batch() -> draw_g1_skeleton()

Usage (single file):
  .venv_sim/bin/python visualize_robot_pkl.py \
      --pkl /home/grease/ego_dataset/amass_trainset/robot/CMU__CMU__143__143_30_stageii.pkl \
      --out data_visualization/robot_pkl_check/CMU__CMU__143__143_30_stageii.gif \
      --frame_step 4

  # Batch: every .pkl in a directory
  .venv_sim/bin/python visualize_robot_pkl.py \
      --pkl_dir /home/grease/ego_dataset/amass_trainset/robot \
      --out_dir data_visualization/robot_pkl_check \
      --sample_n 10 --seed 0 --frame_step 4
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visualize_soma_retarget as V  # reuse G1_CHAIN_MJCF/G1_VIZ/G1_BONES/g1_fk_batch
import visualize_soma_retarget_robot_pkl as VR  # reuse load_robot_pkl_as_df()

C_G1 = '#66bb6a'

_G1_NAME_TO_IDX = {n: i for i, n in enumerate(V.G1_VIZ)}
G1_BONE_IDX = [(_G1_NAME_TO_IDX[p], _G1_NAME_TO_IDX[c]) for p, c in V.G1_BONES]


def draw_g1_skeleton(ax, joints, color, lw=2.5):
    """joints: (N,3) meters, plain [X,Y,Z] order. Direct X->ax.X, Y->ax.Y,
    Z->ax.Z mapping (NOT `V.draw_skeleton()`'s own baked-in axis swizzle,
    which expects g1_fk_batch()'s raw [lateral,height,forward] layout --
    see visualize_smpl_robot_pair.py's draw_g1_skeleton() docstring for the
    full story of that bug)."""
    for p, c in G1_BONE_IDX:
        p0, p1 = joints[p], joints[c]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, lw=lw)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=15, depthshade=False)


def process_one(name, pkl_path, out_path, frame_step, elev=10, azim=0):
    print(f'\n=== {name} ===')
    df_g1, robot_fps = VR.load_robot_pkl_as_df(pkl_path, name)
    robot_fps = float(robot_fps or 30.0)
    n = len(df_g1)
    sampled = list(range(0, n, frame_step))
    print(f'  robot frames: {n} @ {robot_fps:.1f}fps  sampled: {len(sampled)}')

    pos_raw = V.g1_fk_batch(V.G1_VIZ, df_g1, sampled) / 1000.0  # mm -> m
    # g1_fk_batch's raw layout is [lateral(Y), height(Z), forward(X)] ->
    # reorder to plain [X=forward, Y=lateral, Z=height] to match
    # draw_g1_skeleton()'s direct-mapping convention.
    pos = np.stack([pos_raw[:, :, 2], pos_raw[:, :, 0], pos_raw[:, :, 1]], axis=-1)
    foot_idx = [V.G1_VIZ.index('left_ankle_roll_link'), V.G1_VIZ.index('right_ankle_roll_link')]
    pos = pos.copy()
    pos[:, :, 2] -= pos[:, foot_idx, 2].min()

    all_xy = pos[:, :, :2].ravel()
    all_z = pos[:, :, 2].ravel()
    lim_xy = float(np.percentile(np.abs(all_xy), 95)) * 1.3
    lim_z = float(np.percentile(all_z, 95)) * 1.2

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    canvas = fig.canvas
    suptitle = fig.suptitle('', fontsize=11, fontweight='bold')
    layout_done = [False]

    def style():
        # BUG FIX: `set_zlim(0, 2 * lim_z)` doubled the Z-axis padding
        # relative to X/Y's 1.3x-only padding, and combined with the CUBIC
        # `set_box_aspect((1,1,1))` below, this squashed the actual figure
        # into a small, visually "tangled"-looking blob occupying only the
        # bottom half of the plot (confirmed: a G1 figure is only
        # ~1.3-1.5m tall, but the axis went up to ~2.7m). Use the SAME
        # padding factor as X/Y (no extra doubling).
        ax.set_xlim(-lim_xy, lim_xy); ax.set_ylim(-lim_xy, lim_xy); ax.set_zlim(0, lim_z)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X', fontsize=6); ax.set_ylabel('Y', fontsize=6); ax.set_zlabel('Z', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.2)
        ax.set_title('Retargeted G1 (motion_lib)', fontsize=10, fontweight='bold', pad=5)
        ax.view_init(elev=elev, azim=azim)

    def render(fi):
        ax.clear()
        draw_g1_skeleton(ax, pos[fi], C_G1)
        style()
        frame_num = sampled[fi]
        suptitle.set_text(f'{name}  ·  frame {frame_num}/{n}  ·  t={frame_num / robot_fps:.2f}s')
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
            print(f'    [render] {fi + 1}/{N} frames  ({rate:.1f} fps, ETA {eta:.0f}s)', flush=True)
    plt.close(fig)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:],
                        duration=50, loop=0, optimize=False)
    print(f'  saved -> {out_path}  ({N} frames, {time.time() - t0:.0f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', help='single motion_lib robot .pkl path')
    ap.add_argument('--out', help='output .gif path (used with --pkl)')
    ap.add_argument('--pkl_dir', help='directory of motion_lib robot .pkl files')
    ap.add_argument('--out_dir', help='output directory (used with --pkl_dir)')
    ap.add_argument('--sample_n', type=int, help='randomly sample N clips from --pkl_dir')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frame_step', type=int, default=4)
    ap.add_argument('--elev', type=float, default=10, help='camera elevation angle (degrees)')
    ap.add_argument('--azim', type=float, default=0, help='camera azimuth angle (degrees); '
                     '0=front view, 90=side view, 45=3/4 view, etc.')
    args = ap.parse_args()

    if args.pkl:
        if not args.out:
            ap.error('--out is required with --pkl')
        name = os.path.splitext(os.path.basename(args.pkl))[0]
        process_one(name, args.pkl, args.out, args.frame_step, args.elev, args.azim)
    elif args.pkl_dir:
        if not args.out_dir:
            ap.error('--out_dir is required with --pkl_dir')
        files = sorted(f for f in os.listdir(args.pkl_dir) if f.endswith('.pkl'))
        if args.sample_n:
            random.seed(args.seed)
            random.shuffle(files)
            files = files[:args.sample_n]
        print(f'{len(files)} clips selected')
        n_ok, n_fail = 0, 0
        for i, f in enumerate(files):
            name = os.path.splitext(f)[0]
            print(f'\n[{i + 1}/{len(files)}]', end='')
            out_path = os.path.join(args.out_dir, f'{name}.gif')
            try:
                process_one(name, os.path.join(args.pkl_dir, f), out_path, args.frame_step,
                            args.elev, args.azim)
                n_ok += 1
            except Exception as e:
                print(f'  [!] FAILED: {name}: {e}')
                n_fail += 1
        print(f'\n\nDone: {n_ok} succeeded, {n_fail} failed, out of {len(files)} total')
    else:
        ap.error('must specify either --pkl (single) or --pkl_dir (batch)')


if __name__ == '__main__':
    main()
