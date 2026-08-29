"""
Standalone visualization of a SOMA-skeleton BVH motion file (source side
ONLY -- no retargeted G1 CSV pairing required, unlike
`visualize_soma_retarget.py`).

Use this when you just want to eyeball a raw SOMA BVH (e.g. sanity-check a
new retargeter input, or a freshly-converted LAFAN1/AMASS BVH) without
needing a matching retargeted CSV on disk.

Reuses the BVH parser / FK / skeleton-drawing code from
`visualize_soma_retarget.py` (single source of truth for the BVH<->SMPL
convention), so any bugfixes there (e.g. the `End Site` block-skipping fix)
automatically apply here too.

Usage:
  .venv_sim/bin/python visualize_soma_bvh.py \
      --bvh /home/grease/soma-retargeter/assets/motions/bvh/Neutral_walk_forward_002__A057.bvh \
      --out data_visualization/soma_bvh_check/Neutral_walk_forward_002__A057.gif \
      --fps 120 --frame_step 8

  # Or point at a whole directory to render every .bvh file in it:
  .venv_sim/bin/python visualize_soma_bvh.py \
      --bvh_dir /home/grease/ego_dataset/lafan1_all/soma_bvh \
      --out_dir data_visualization/soma_bvh_check \
      --fps 30 --frame_step 8
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visualize_soma_retarget as V  # reuse BVH parser / FK / drawing code

C_SOMA = '#4fc3f7'


def render_bvh(bvh_path, out_path, fps, frame_step):
    name = os.path.splitext(os.path.basename(bvh_path))[0]
    print(f'\n=== {name} ===')

    joints, offsets, channels, parents, frames = V.parse_bvh(bvh_path)
    n = len(frames)
    sampled = list(range(0, n, frame_step))
    print(f'  BVH frames: {n}  sampled: {len(sampled)}')

    pos = V.fk_batch(V.VIZ_JOINTS, joints, offsets, channels, parents, frames, sampled)
    foot_idx = [V.VIZ_JOINTS.index(n_) for n_ in ('LeftFoot', 'RightFoot')]
    pos[:, :, 1] -= pos[:, foot_idx, 1].min()  # floor-align (feet at z=0)

    lim = V.axis_limits([pos])

    # Single reused Figure/Axes (see visualize_soma_retarget.py's
    # process_one() for why this + grabbing the raw RGBA buffer directly is
    # much faster than creating a new Figure and going through a
    # savefig(PNG)->PIL.open() round-trip every frame).
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    canvas = fig.canvas
    suptitle = fig.suptitle('', fontsize=11, fontweight='bold')
    layout_done = [False]

    def render(fi):
        ax.clear()
        V.draw_skeleton(ax, pos[fi], V.BONES, V.VIZ_JOINTS, C_SOMA)
        V.style_ax(ax, 'Source SOMA BVH', *lim)
        frame_num = sampled[fi]
        suptitle.set_text(f'{name}  ·  frame {frame_num}/{n}  ·  t={frame_num / fps:.2f}s')
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
    ap.add_argument('--bvh', help='path to a single .bvh file')
    ap.add_argument('--out', help='output .gif path (used with --bvh)')
    ap.add_argument('--bvh_dir', help='directory of .bvh files (renders ALL of them)')
    ap.add_argument('--out_dir', help='output directory (used with --bvh_dir)')
    ap.add_argument('--fps', type=float, default=30.0, help='source BVH frame rate')
    ap.add_argument('--frame_step', type=int, default=8,
                     help='render every Nth frame (higher = faster, coarser)')
    args = ap.parse_args()

    if args.bvh:
        if not args.out:
            ap.error('--out is required with --bvh')
        render_bvh(args.bvh, args.out, args.fps, args.frame_step)
    elif args.bvh_dir:
        if not args.out_dir:
            ap.error('--out_dir is required with --bvh_dir')
        bvh_files = sorted(glob.glob(os.path.join(args.bvh_dir, '*.bvh')))
        print(f'Found {len(bvh_files)} .bvh files in {args.bvh_dir}')
        for bvh_path in bvh_files:
            name = os.path.splitext(os.path.basename(bvh_path))[0]
            out_path = os.path.join(args.out_dir, f'{name}.gif')
            render_bvh(bvh_path, out_path, args.fps, args.frame_step)
    else:
        ap.error('must specify either --bvh (single file) or --bvh_dir (batch)')


if __name__ == '__main__':
    main()
