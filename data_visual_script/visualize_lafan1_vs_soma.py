"""Side-by-side visualization: RAW LAFAN1 BVH vs. our SOMA-converted BVH.

Reuses the generic BVH parser/FK from `visualize_soma_retarget.py` (name/
hierarchy-agnostic -- works on ANY BVH skeleton, not just SOMA), but defines
a separate joint/bone list matching LAFAN1's own naming convention
(LeftUpLeg/LeftLeg/LeftFoot/LeftToe, Spine/Spine1/Spine2/Neck/Head, ...)
for the left panel, while reusing the existing SOMA VIZ_JOINTS/BONES for
the right panel.

Usage:
    .venv_sim/bin/python visualize_lafan1_vs_soma.py \
        --raw_bvh /home/grease/egodata/downloads/lafan1_extracted/walk1_subject1.bvh \
        --soma_bvh /tmp/walk1_subject1_soma_direct.bvh \
        --out /tmp/walk1_subject1_raw_vs_soma.gif \
        --fps 30 --frame_step 8
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_soma_retarget import (  # noqa: E402
    parse_bvh, fk, fk_batch, axis_limits, draw_skeleton, style_ax,
    VIZ_JOINTS as SOMA_VIZ_JOINTS, BONES as SOMA_BONES,
)

# -- Raw LAFAN1 joint/bone list (its own native naming) --
LAFAN_VIZ_JOINTS = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe',
]
LAFAN_BONES = [
    ('Hips', 'Spine'), ('Spine', 'Spine1'), ('Spine1', 'Spine2'),
    ('Spine2', 'Neck'), ('Neck', 'Head'),
    ('Spine2', 'LeftShoulder'), ('LeftShoulder', 'LeftArm'), ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),
    ('Spine2', 'RightShoulder'), ('RightShoulder', 'RightArm'), ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand'),
    ('Hips', 'LeftUpLeg'), ('LeftUpLeg', 'LeftLeg'), ('LeftLeg', 'LeftFoot'), ('LeftFoot', 'LeftToe'),
    ('Hips', 'RightUpLeg'), ('RightUpLeg', 'RightLeg'), ('RightLeg', 'RightFoot'), ('RightFoot', 'RightToe'),
]

C_RAW = '#ef5350'
C_SOMA = '#4fc3f7'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_bvh', required=True)
    ap.add_argument('--soma_bvh', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--frame_step', type=int, default=8)
    args = ap.parse_args()

    print(f'Loading raw LAFAN1: {args.raw_bvh}')
    joints_r, offsets_r, channels_r, parents_r, frames_r = parse_bvh(args.raw_bvh)
    print(f'  {len(frames_r)} frames, {len(joints_r)} joints')

    print(f'Loading SOMA-converted: {args.soma_bvh}')
    joints_s, offsets_s, channels_s, parents_s, frames_s = parse_bvh(args.soma_bvh)
    print(f'  {len(frames_s)} frames, {len(joints_s)} joints')

    n = min(len(frames_r), len(frames_s))
    sampled = list(range(0, n, args.frame_step))
    print(f'Sampling {len(sampled)} frames (every {args.frame_step})')

    pos_raw = fk_batch(LAFAN_VIZ_JOINTS, joints_r, offsets_r, channels_r, parents_r, frames_r, sampled)
    pos_soma = fk_batch(SOMA_VIZ_JOINTS, joints_s, offsets_s, channels_s, parents_s, frames_s, sampled)

    raw_foot_idx = [LAFAN_VIZ_JOINTS.index(n_) for n_ in ('LeftFoot', 'RightFoot')]
    pos_raw[:, :, 1] -= pos_raw[:, raw_foot_idx, 1].min()
    soma_foot_idx = [SOMA_VIZ_JOINTS.index(n_) for n_ in ('LeftFoot', 'RightFoot')]
    pos_soma[:, :, 1] -= pos_soma[:, soma_foot_idx, 1].min()

    lim_raw = axis_limits([pos_raw])
    lim_soma = axis_limits([pos_soma])

    def render(fi):
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        draw_skeleton(ax1, pos_raw[fi], LAFAN_BONES, LAFAN_VIZ_JOINTS, C_RAW)
        draw_skeleton(ax2, pos_soma[fi], SOMA_BONES, SOMA_VIZ_JOINTS, C_SOMA)
        style_ax(ax1, 'Raw LAFAN1 BVH', *lim_raw)
        style_ax(ax2, 'SOMA-converted BVH', *lim_soma)
        frame_num = sampled[fi]
        plt.suptitle(f'frame {frame_num}/{n}  t={frame_num / args.fps:.2f}s',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        return fig

    gif_frames = []
    t0 = time.time()
    for fi in range(len(sampled)):
        fig = render(fi)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=85, bbox_inches='tight')
        buf.seek(0)
        gif_frames.append(Image.open(buf).copy())
        plt.close(fig)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    gif_frames[0].save(args.out, save_all=True, append_images=gif_frames[1:],
                        duration=50, loop=0, optimize=False)
    print(f'saved -> {args.out}  ({len(sampled)} frames, {time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
