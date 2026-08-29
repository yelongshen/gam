"""
Side-by-side visualization of a `smpl/<name>.pkl` (human) <-> `robot/<name>.pkl`
(motion_lib retargeted G1) pair, for datasets laid out like `eval_subset/`,
`amass_evalset/`, `lafan1_evalset/`, `amass_trainset/` (i.e. flat
`smpl/`+`robot/` subdirectories, NOT the BVH-paired layout that
`visualize_soma_retarget.py`/`visualize_soma_retarget_robot_pkl.py` expect).

LEFT  panel : source human `smpl_joints` (T,24,3) stored directly inside the
              `smpl/<name>.pkl` (smpl_filtered format) -- no FK needed, these
              are already Z-up, root-local (see SMPL_FILTERED_DATA_FORMAT.md).
RIGHT panel : retargeted G1, computed via the REAL G1-MJCF forward kinematics
              (reusing `visualize_soma_retarget.py`'s `G1_CHAIN_MJCF`/`G1_VIZ`/
              `G1_BONES`) driven by the `robot/<name>.pkl`'s `dof` (T,29 rad,
              MuJoCo order) + `root_trans_offset` (T,3 m) + `root_rot` (T,4 xyzw).

Usage:
  .venv_sim/bin/python visualize_smpl_robot_pair.py \
      --smpl_dir /home/grease/ego_dataset/amass_trainset/smpl \
      --robot_dir /home/grease/ego_dataset/amass_trainset/robot \
      --name "CMU__CMU__143__143_30_stageii" \
      --out data_visualization/amass_trainset_check/CMU__CMU__143__143_30_stageii.gif \
      --frame_step 4

  # Batch: sample N random pairs
  .venv_sim/bin/python visualize_smpl_robot_pair.py \
      --smpl_dir /home/grease/ego_dataset/amass_trainset/smpl \
      --robot_dir /home/grease/ego_dataset/amass_trainset/robot \
      --out_dir data_visualization/amass_trainset_check \
      --sample_n 30 --seed 0 --frame_step 4
"""
import argparse
import os
import random
import sys
import time

import joblib
import numpy as np
import scipy.spatial.transform as sT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_process'))
import visualize_soma_retarget as V  # reuse VIZ_JOINTS/BONES, G1 FK chain, drawing
import visualize_soma_retarget_robot_pkl as VR  # reuse load_robot_pkl_as_df() + verified g1_fk_batch path
import stream_clip_mode2 as S  # reuse the VALIDATED per-frame root-derotation (official_encoder_joints)

C_SMPL = '#4fc3f7'
C_G1 = '#66bb6a'

# Same SMPL kinematic tree as stream_clip_mode2.SMPL_LINKS (verified
# identical parent structure -- both describe the standard 24-joint SMPL
# skeleton), reused here as (parent, child) pairs for draw_smpl_skeleton().
SMPL_BONES = S.SMPL_LINKS


def draw_smpl_skeleton(ax, joints, color, lw=2.5):
    """joints: (24,3) meters, Z-up. Plot: X->ax.X, Y->ax.Y(depth), Z->ax.Z(height)."""
    for p, c in SMPL_BONES:
        p0, p1 = joints[p], joints[c]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, lw=lw)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=15, depthshade=False)


def draw_g1_skeleton(ax, joints, bones, color, lw=2.5):
    """joints: (N,3) meters, plain [X,Y,Z] order (NOT `V.g1_fk_batch`'s own
    raw [lateral,height,forward] output layout).

    BUG FIX: `V.draw_skeleton()` (from visualize_soma_retarget.py) has its
    OWN built-in axis swizzle baked in (`ax.plot([p0[0],p1[0]], [p0[2],p1[2]],
    [p0[1],p1[1]], ...)`), designed to consume `g1_fk_batch()`'s raw
    [lateral,height,forward] output directly. Since this script ALREADY
    reorders `pos_g1` into plain [X=forward,Y=lateral,Z=height] (for its own
    floor-alignment / axis_limits_m() code, shared with the SMPL panel),
    passing that into `V.draw_skeleton()` double-swizzled the axes -- e.g.
    ax.Z (should be height) was actually getting the LATERAL coordinate,
    completely scrambling the G1 panel. This drawer uses the SAME direct
    X->ax.X, Y->ax.Y, Z->ax.Z mapping as draw_smpl_skeleton() above, matching
    the plain [X,Y,Z] convention `pos_g1` is actually stored in here."""
    for p, c in bones:
        p0, p1 = joints[p], joints[c]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, lw=lw)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=15, depthshade=False)


def process_one(name, smpl_dir, robot_dir, out_path, frame_step):
    print(f'\n=== {name} ===')
    smpl_data = joblib.load(os.path.join(smpl_dir, f'{name}.pkl'))
    smpl_joints = smpl_data['smpl_joints']  # (T,24,3) meters, Z-up, root-local
    smpl_fps = float(smpl_data.get('fps', 50.0))

    # Reuse the ALREADY-VERIFIED motion_lib -> retargeter-CSV-schema -> g1_fk
    # path from visualize_soma_retarget_robot_pkl.py (battle-tested against
    # the same G1_CHAIN_MJCF FK used everywhere else in this pipeline),
    # instead of a fresh from-scratch FK -- avoids re-introducing the
    # heading/orientation bugs that plagued earlier ad-hoc attempts.
    robot_pkl_path = os.path.join(robot_dir, f'{name}.pkl')
    df_g1, robot_fps = VR.load_robot_pkl_as_df(robot_pkl_path, name)
    robot_fps = float(robot_fps or 30.0)

    # BUG FIX: smpl_joints (smpl_filtered, ~50fps) and df_g1 (motion_lib,
    # ~30fps) are recorded at DIFFERENT frame rates -- confirmed empirically
    # (e.g. DFaust__.../50002_shake_arms_stageii: 96 smpl frames @ 50fps vs
    # 58 robot frames @ 30fps for the SAME ~1.92s clip). Treating raw index
    # `i` as equivalent between the two arrays (the previous
    # `n = min(len(smpl_joints), len(df_g1)); sampled = range(0, n, frame_step)`
    # approach) silently desyncs the two panels in TIME as the clip
    # progresses, since index i means t=i/50s on one side but t=i/30s on the
    # other. Fix: sample both sides on a SHARED time axis (seconds), so
    # frame `fi` in the output GIF always shows the same real timestamp on
    # both panels.
    duration = min((len(smpl_joints) - 1) / smpl_fps, (len(df_g1) - 1) / robot_fps)
    dt = frame_step / max(smpl_fps, robot_fps)  # finer of the two rates sets the step size
    sample_times = np.arange(0.0, duration, dt)
    smpl_idx = np.clip(np.round(sample_times * smpl_fps).astype(int), 0, len(smpl_joints) - 1)
    robot_idx = np.clip(np.round(sample_times * robot_fps).astype(int), 0, len(df_g1) - 1)
    n_out = len(sample_times)
    print(f'  smpl frames: {len(smpl_joints)} @ {smpl_fps:.1f}fps  '
          f'robot frames: {len(df_g1)} @ {robot_fps:.1f}fps  '
          f'time-synced output frames: {n_out}')

    pos_smpl_raw = smpl_joints[smpl_idx].astype(np.float64)

    # BUG FIX #2: `V.g1_fk_batch()` internally derotates the G1 skeleton to a
    # FIXED HEADING anchored to `df_g1`'s own frame-0 `root_rotateZ`. The
    # SMPL side was previously derotated ONLY at frame 0 (a single fixed
    # correction applied to every subsequent frame), which is NOT what the
    # already-validated live streaming pipeline does: `stream_clip_mode2.py`'s
    # `LiveSkeleton` visualizes joints AFTER `official_encoder_joints()`,
    # which removes EACH FRAME'S OWN root rotation independently (per-frame,
    # not just frame 0) -- see `smpl_joints_multi_future_local()`'s real
    # semantics. Frame-0-only derotation leaves the body still rotating with
    # its true world heading for every later frame, which is what made the
    # SMPL panel look wrong as soon as the person turned. Fix: reuse the
    # EXACT SAME `official_encoder_joints()` transform (not a re-derived
    # approximation), driven by this clip's own `pose_aa[:, :3]` root
    # axis-angle, resampled onto the same `smpl_idx` time-sync indices.
    pose_aa_root = smpl_data['pose_aa'][smpl_idx, :3].astype(np.float64)
    pos_smpl, _root_quat_w = S.official_encoder_joints(pos_smpl_raw, pose_aa_root)

    foot_idx_smpl = [10, 11]  # SMPL LeftToeBase/RightToeBase
    pos_smpl = pos_smpl.copy()
    pos_smpl[:, :, 2] -= pos_smpl[:, foot_idx_smpl, 2].min()

    # V.g1_fk_batch returns MILLIMETERS (matches G1_CHAIN_MJCF's own mm-scale
    # convention, see visualize_soma_retarget.py's parse_g1_mjcf docstring)
    # and floor axis order [lateral(Y), height(Z), forward(X)] -- convert to
    # meters + [X,Y,Z] here to match this script's own (X,Y,Z) convention.
    pos_g1_raw = V.g1_fk_batch(V.G1_VIZ, df_g1, robot_idx) / 1000.0  # mm -> m
    pos_g1 = np.stack([pos_g1_raw[:, :, 2], pos_g1_raw[:, :, 0], pos_g1_raw[:, :, 1]], axis=-1)
    foot_g1_idx = [V.G1_VIZ.index('left_ankle_roll_link'), V.G1_VIZ.index('right_ankle_roll_link')]
    pos_g1 = pos_g1.copy()
    pos_g1[:, :, 2] -= pos_g1[:, foot_g1_idx, 2].min()

    # V.G1_BONES is a list of (parent_name, child_name) pairs; convert to
    # index pairs into V.G1_VIZ once here, for draw_g1_skeleton() (which
    # expects index pairs, matching SMPL_BONES' convention).
    g1_name_to_idx = {n: i for i, n in enumerate(V.G1_VIZ)}
    G1_BONE_IDX = [(g1_name_to_idx[p], g1_name_to_idx[c]) for p, c in V.G1_BONES]

    def axis_limits_m(pos):
        all_xy = pos[:, :, :2].ravel()
        all_z = pos[:, :, 2].ravel()
        lim_xy = float(np.percentile(np.abs(all_xy), 95)) * 1.3
        lim_z = float(np.percentile(all_z, 95)) * 1.2
        return lim_xy, lim_z

    lim_smpl = axis_limits_m(pos_smpl)
    lim_g1 = axis_limits_m(pos_g1)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    canvas = fig.canvas
    suptitle = fig.suptitle('', fontsize=11, fontweight='bold')
    layout_done = [False]

    def style(ax, title, lim):
        lim_xy, lim_z = lim
        # BUG FIX: same "2 * lim_z" over-padding bug as visualize_robot_pkl.py
        # -- see that file's style()'s comment for the full explanation of
        # how this squashed the figure into a "tangled blob" appearance.
        ax.set_xlim(-lim_xy, lim_xy); ax.set_ylim(-lim_xy, lim_xy); ax.set_zlim(0, lim_z)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X', fontsize=6); ax.set_ylabel('Y', fontsize=6); ax.set_zlabel('Z', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.2)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
        # BUG FIX: elev=15/azim=-60 foreshortened the LATERAL (left/right)
        # axis enough that left/right limbs visually overlapped into what
        # looked like a single folded chain -- CONFIRMED the skeleton is NOT
        # actually collapsed (left/right joints are properly separated in
        # 3D). Use a true front-facing view instead.
        ax.view_init(elev=10, azim=0)

    def render(fi):
        ax1.clear(); ax2.clear()
        draw_smpl_skeleton(ax1, pos_smpl[fi], C_SMPL)
        draw_g1_skeleton(ax2, pos_g1[fi], G1_BONE_IDX, C_G1)  # meters, plain [X,Y,Z] (see draw_g1_skeleton docstring)
        style(ax1, 'Source SMPL (smpl_filtered)', lim_smpl)
        style(ax2, 'Retargeted G1 (motion_lib)', lim_g1)
        frame_num = smpl_idx[fi]
        suptitle.set_text(f'{name}  ·  t={sample_times[fi]:.2f}s  ·  '
                           f'smpl_frame={smpl_idx[fi]}  robot_frame={robot_idx[fi]}')
        if not layout_done[0]:
            plt.tight_layout()
            layout_done[0] = True
        canvas.draw()
        w, h = canvas.get_width_height()
        renderer = canvas.get_renderer()
        rgba = np.asarray(renderer.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4).copy()
        return Image.fromarray(rgba, mode='RGBA').convert('RGB')

    N = n_out
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
    ap.add_argument('--smpl_dir', required=True)
    ap.add_argument('--robot_dir', required=True)
    ap.add_argument('--name', help='single clip name (no extension)')
    ap.add_argument('--out', help='output .gif path (used with --name)')
    ap.add_argument('--out_dir', help='output directory (used with --sample_n)')
    ap.add_argument('--sample_n', type=int, help='randomly sample N clips from smpl_dir/robot_dir')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frame_step', type=int, default=4)
    args = ap.parse_args()

    if args.name:
        if not args.out:
            ap.error('--out is required with --name')
        process_one(args.name, args.smpl_dir, args.robot_dir, args.out, args.frame_step)
    elif args.sample_n:
        if not args.out_dir:
            ap.error('--out_dir is required with --sample_n')
        smpl_names = {os.path.splitext(f)[0] for f in os.listdir(args.smpl_dir) if f.endswith('.pkl')}
        robot_names = {os.path.splitext(f)[0] for f in os.listdir(args.robot_dir) if f.endswith('.pkl')}
        paired = sorted(smpl_names & robot_names)
        print(f'{len(paired)} paired clips available')
        random.seed(args.seed)
        random.shuffle(paired)
        selected = paired[:args.sample_n]
        n_ok, n_fail = 0, 0
        for i, name in enumerate(selected):
            print(f'\n[{i + 1}/{len(selected)}]', end='')
            out_path = os.path.join(args.out_dir, f'{name}.gif')
            try:
                process_one(name, args.smpl_dir, args.robot_dir, out_path, args.frame_step)
                n_ok += 1
            except Exception as e:
                print(f'  [!] FAILED: {name}: {e}')
                n_fail += 1
        print(f'\n\nDone: {n_ok} succeeded, {n_fail} failed, out of {len(selected)} total')
    else:
        ap.error('must specify either --name (single) or --sample_n (batch random sample)')


if __name__ == '__main__':
    main()
