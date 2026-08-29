"""
Side-by-side visualization of a SOMA Retargeter conversion:
   LEFT  : source SOMA-skeleton BVH motion (input to the retargeter)
   RIGHT : retargeted Unitree G1 (29 DOF) motion (output CSV of the retargeter)

Both panels are floor-aligned (feet at z=0) and heading-fixed to frame 0, so
body turning/rotation is visible as body motion rather than being cancelled by
a rotating camera.

Expects a `soma-retargeter` checkout (https://github.com/NVIDIA/soma-retargeter)
with a BVH source folder and a mirrored CSV export folder, e.g.:
    <root>/assets/motions/bvh/<name>.bvh
    <root>/assets/motions/test-export/<name>.csv

Usage:
  .venv_sim/bin/python visualize_soma_retarget.py \
      --bvh_dir ~/soma-retargeter/assets/motions/bvh \
      --csv_dir ~/soma-retargeter/assets/motions/test-export \
      --name Neutral_walk_forward_002__A057 \
      --out data_visualization/soma_retarget_test/Neutral_walk_forward_002__A057_soma_vs_g1.gif \
      --fps 120 --frame_step 8
"""
import argparse
import io
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image


# ── BVH parser (same convention used elsewhere in this pipeline, e.g.
#    data_process/classify_motions.py) ──────────────────────────────────────
def parse_bvh(path):
    with open(path) as f:
        lines = f.read().split('\n')
    joints, offsets, channels, parents = [], {}, {}, {}
    stack, ch_idx, i = [], 0, 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('ROOT ') or line.startswith('JOINT '):
            name = line.split()[1]
            joints.append(name)
            parents[name] = stack[-1] if stack else None
            stack.append(name)
        elif line.startswith('OFFSET') and stack:
            p = line.split()
            offsets[stack[-1]] = np.array([float(p[1]), float(p[2]), float(p[3])])
        elif line.startswith('CHANNELS') and stack:
            p = line.split(); n = int(p[1])
            channels[stack[-1]] = {'start': ch_idx, 'types': p[2:2 + n]}
            ch_idx += n
        elif line.startswith('End Site'):
            # Consume the WHOLE End Site { ... } block (including its own
            # closing brace) WITHOUT popping the enclosing joint off the
            # stack. Skipping this causes every subsequent sibling/child
            # joint to attach to the wrong parent.
            i += 1
            while i < len(lines):
                if lines[i].strip() == '}':
                    break
                i += 1
        elif line == '}' and stack:
            stack.pop()
        elif line.strip() == 'MOTION':
            break
        i += 1
    mo = next(k for k, l in enumerate(lines) if l.strip() == 'MOTION')
    nf = int(lines[mo + 1].split()[1])
    frame_data = []
    for k in range(mo + 3, mo + 3 + nf):
        vals = lines[k].split()
        if vals:
            frame_data.append([float(v) for v in vals])
    return joints, offsets, channels, parents, np.array(frame_data)


def _rot(angles, types):
    def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = np.eye(3)
    for ang, t in zip(angles, types):
        a = np.radians(ang)
        if t == 'Xrotation': R = R @ Rx(a)
        elif t == 'Yrotation': R = R @ Ry(a)
        elif t == 'Zrotation': R = R @ Rz(a)
    return R


def fk(joints, offsets, channels, parents, frame_data):
    wp, wr = {}, {}
    for j in joints:
        if j not in channels:
            wp[j] = wp.get(parents.get(j), np.zeros(3)).copy()
            wr[j] = wr.get(parents.get(j), np.eye(3)).copy()
            continue
        ch = channels[j]; start, types = ch['start'], ch['types']
        pos_v = [None] * 3; rot_a, rot_t = [], []
        for k, t in enumerate(types):
            v = frame_data[start + k]
            if 'position' in t.lower():
                pos_v['XYZ'.index(t[0].upper())] = v
            else:
                rot_a.append(v); rot_t.append(t)
        local_rot = _rot(rot_a, rot_t)
        parent = parents[j]
        if parent is None:
            wp[j] = np.array([v if v else 0.0 for v in pos_v])
            wr[j] = local_rot
        else:
            local_pos = np.array([v if v else 0.0 for v in pos_v])
            wp[j] = wp.get(parent, np.zeros(3)) + wr.get(parent, np.eye(3)) @ offsets.get(j, np.zeros(3)) + local_pos
            wr[j] = wr.get(parent, np.eye(3)) @ local_rot
    return wp


# ── Body-only joints used for visualization (no fingers/face) ──────────────
VIZ_JOINTS = [
    'Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftLeg', 'LeftShin', 'LeftFoot',
    'RightLeg', 'RightShin', 'RightFoot',
]
BONES = [
    ('Hips', 'Spine1'), ('Spine1', 'Spine2'), ('Spine2', 'Chest'),
    ('Chest', 'Neck1'), ('Neck1', 'Head'),
    ('Chest', 'LeftShoulder'), ('LeftShoulder', 'LeftArm'), ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),
    ('Chest', 'RightShoulder'), ('RightShoulder', 'RightArm'), ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand'),
    ('Hips', 'LeftLeg'), ('LeftLeg', 'LeftShin'), ('LeftShin', 'LeftFoot'),
    ('Hips', 'RightLeg'), ('RightLeg', 'RightShin'), ('RightShin', 'RightFoot'),
]


def fk_batch(viz, joints, offsets, channels, parents, all_frames, sampled):
    out = np.zeros((len(sampled), len(viz), 3))
    n = len(sampled)
    t0 = time.time()
    for fi, f in enumerate(sampled):
        wp = fk(joints, offsets, channels, parents, all_frames[f])
        for ji, name in enumerate(viz):
            out[fi, ji] = wp.get(name, np.zeros(3))
        if (fi + 1) % 50 == 0 or fi + 1 == n:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed if elapsed > 0 else 0
            eta = (n - fi - 1) / rate if rate > 0 else 0
            print(f'    [BVH FK] {fi + 1}/{n} frames  ({rate:.0f} fps, ETA {eta:.0f}s)',
                  flush=True)
    return out


# ── G1 (29 DOF) forward kinematics: REAL kinematic chain, parsed from the
#    actual robot description (gear_sonic_deploy/g1/g1_29dof.xml MJCF) ────
#
# An earlier version of this used a hand-typed, APPROXIMATE `G1_CHAIN` that
# collapsed each 3-DOF group (shoulder pitch/roll/yaw, hip pitch/roll/yaw,
# wrist roll/pitch/yaw) into ONE combined joint at a single guessed offset,
# e.g. `left_shoulder` at offset [0,215,0] from chest with all 3 rotations
# applied there. The REAL robot has THREE SEPARATE links
# (`left_shoulder_pitch_link` -> `left_shoulder_roll_link` ->
# `left_shoulder_yaw_link`), each with its OWN offset AND, critically, its
# own non-trivial REST ORIENTATION quaternion (e.g.
# `left_shoulder_pitch_link` has `quat="0.990264 0.139201 ..."`, roughly an
# 8 deg tilt) -- meaning the joint's rotation AXIS (e.g. axis="0 1 0" for
# pitch) is expressed in an ALREADY-ROTATED local frame, not world/parent Y
# directly. Ignoring these rest quaternions while approximating the whole
# 3-DOF group as one joint produced a visually wrong arm shape even though
# bone lengths/angles looked numerically "plausible" in isolation. Verified
# by comparing against the retargeter's OWN bundled reference BVH/CSV pair
# (`Neutral_throw_ball_001__A057`) -- a known-good retargeting -- which
# still looked wrong with the approximate chain, isolating the bug to this
# FK model rather than the retargeting data itself.
G1_MJCF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gear_sonic_deploy", "g1", "g1_29dof.xml")


def _quat_to_matrix(q):
    """MJCF quat convention: (w, x, y, z)."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


def _axis_angle_to_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def parse_g1_mjcf(path=G1_MJCF_PATH):
    """Parse the real G1 MJCF `<worldbody>` into a joint-chain list:
    [(body_name, parent_body_name, rest_pos_mm, rest_quat (wxyz),
      joint_name_or_None, joint_axis_or_None), ...] in document (=
      topological) order. `pos` is converted from the XML's native METERS
      to MILLIMETERS here, to match this module's existing mm-scale
      convention (root translation from the CSV is likewise `* 10`,
      cm -> mm). `quat`/`pos` are the body's REST transform relative to its
      parent BODY frame (not yet including this body's own joint rotation,
      which is applied about `joint_axis` at the joint's own `pos` -- here
      always [0,0,0] i.e. at the body origin, true for every joint in this
      file)."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    assert worldbody is not None, f"no <worldbody> found in {path}"

    chain = []

    def walk(body_el, parent_name):
        name = body_el.get("name")
        pos_m = np.array([float(v) for v in body_el.get("pos", "0 0 0").split()])
        quat = np.array([float(v) for v in body_el.get("quat", "1 0 0 0").split()])
        joint_el = body_el.find("joint")
        jname, jaxis = None, None
        if joint_el is not None and joint_el.get("type") != "free":
            jname = joint_el.get("name")
            jaxis = np.array([float(v) for v in joint_el.get("axis", "0 0 1").split()])
        chain.append((name, parent_name, pos_m * 1000.0, quat, jname, jaxis))
        for child in body_el.findall("body"):
            walk(child, name)

    top_body = worldbody.find("body")  # pelvis
    assert top_body is not None
    chain.append((top_body.get("name"), None, np.zeros(3),
                  np.array([1., 0., 0., 0.]), None, None))
    for child in top_body.findall("body"):
        walk(child, top_body.get("name"))

    return chain


G1_CHAIN_MJCF = parse_g1_mjcf()
G1_VIZ = [name for name, *_ in G1_CHAIN_MJCF]
_G1_PARENT_OF = {name: parent for name, parent, *_ in G1_CHAIN_MJCF}
G1_BONES = [(parent, name) for name, parent, *_ in G1_CHAIN_MJCF if parent is not None]

# G1's head is a FIXED, non-articulated mesh glued onto `torso_link`
# (`<geom pos="0.0039635 0 -0.054" ... mesh="head_link"/>` -- a geom, not a
# separate `<body>`/`<joint>`), so it never appears in the parsed kinematic
# chain above and the source-BVH panel's `Head` marker would otherwise have
# no G1 counterpart at all. Add a SYNTHETIC "head" point for visual
# completeness only: a fixed approximate offset above `torso_link` along
# its own local up axis. This is NOT derived from real kinematics (there is
# no rotatable head joint on this robot) and is for display only.
G1_VIZ = G1_VIZ + ['head']
G1_BONES = G1_BONES + [('torso_link', 'head')]
_G1_HEAD_OFFSET_MM = np.array([0.0, 0.0, 220.0])  # approximate, torso_link local frame



def g1_fk(row, fixed_yaw_rad=None):
    """Compute G1 world joint positions from one row of the retargeter CSV,
    using the REAL kinematic chain parsed from `g1_29dof.xml` (see
    `G1_CHAIN_MJCF` above for why the previous hand-approximated chain was
    wrong).

    For each body: world_rot(body) = world_rot(parent) @ quat(rest_quat) @
    axis_angle(joint_axis, joint_angle) -- the body's rest quaternion is
    applied FIRST (establishing the local frame the joint axis is defined
    in), THEN the joint's own live rotation about that axis. This matches
    MJCF semantics directly (a joint's `axis` is expressed in the frame
    established by its body's `pos`/`quat`, not world/parent axes).

    fixed_yaw_rad: constant yaw (radians) to derotate by, so the body's
    actual turning is visible as motion rather than being cancelled every
    frame by a per-frame-relative camera.
    """
    def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # Root translation/rotation come from the retargeter's own CSV export
    # of the MJCF floating-base joint; translate values are in cm (matches
    # MJCF's native meter convention * 100, verified: root_translateZ ~=
    # 78-79 cm for a standing pose, matching the MJCF pelvis rest height
    # `pos="0 0 0.793"` i.e. 79.3 cm). Scale to mm to match the rest of this
    # FK (MJCF offsets converted to mm at parse time).
    root_pos = np.array([row['root_translateX'], row['root_translateY'], row['root_translateZ']]) * 10.0
    R_root = (Rz(np.radians(row['root_rotateZ']))
              @ Ry(np.radians(row['root_rotateY']))
              @ Rx(np.radians(row['root_rotateX'])))

    wp, wr = {}, {}
    for name, parent, rest_pos_mm, rest_quat, jname, jaxis in G1_CHAIN_MJCF:
        R_rest = _quat_to_matrix(rest_quat)
        if jname is not None:
            angle = np.radians(float(row.get(f"{jname}_dof", 0.0)))
            R_joint = _axis_angle_to_matrix(jaxis, angle)
        else:
            R_joint = np.eye(3)
        R_local = R_rest @ R_joint
        if parent is None:
            wp[name] = root_pos.copy()
            wr[name] = R_root.copy()
        else:
            wp[name] = wp[parent] + wr[parent] @ rest_pos_mm
            wr[name] = wr[parent] @ R_local

    # Synthetic head marker (torso_link's head mesh is a fixed geom, not a
    # rotatable joint -- see G1_VIZ/G1_BONES comment above).
    if 'torso_link' in wp:
        wp['head'] = wp['torso_link'] + wr['torso_link'] @ _G1_HEAD_OFFSET_MM

    yaw = fixed_yaw_rad if fixed_yaw_rad is not None else np.radians(row['root_rotateZ'])
    c, s = np.cos(-yaw), np.sin(-yaw)
    R_yaw_inv = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    # MJCF world convention here is (X=forward, Y=left, Z=up). Remap to the
    # [lateral, height, forward] tuple order `draw_skeleton()` expects
    # (plotted as ax.X=lateral, ax.Y=forward/depth, ax.Z=height).
    result = {}
    for k, v in wp.items():
        vd = R_yaw_inv @ v
        result[k] = np.array([vd[1], vd[2], vd[0]])  # [lateral(Y), height(Z), forward(X)]
    return result


def g1_fk_batch(viz_joints, df, sampled):
    fixed_yaw = np.radians(df.iloc[sampled[0]]['root_rotateZ'])
    out = np.zeros((len(sampled), len(viz_joints), 3))
    n = len(sampled)
    t0 = time.time()
    for fi, frame_idx in enumerate(sampled):
        wp = g1_fk(df.iloc[frame_idx], fixed_yaw_rad=fixed_yaw)
        for ji, jname in enumerate(viz_joints):
            out[fi, ji] = wp.get(jname, np.zeros(3))
        if (fi + 1) % 50 == 0 or fi + 1 == n:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed if elapsed > 0 else 0
            eta = (n - fi - 1) / rate if rate > 0 else 0
            print(f'    [G1 FK] {fi + 1}/{n} frames  ({rate:.0f} fps, ETA {eta:.0f}s)',
                  flush=True)
    return out


def axis_limits(arrays, pct=95):
    """Percentile-based axis limits so a few outlier frames (e.g. a jump
    apex) don't blow out the scale for the rest of the clip."""
    all_x = np.concatenate([a[:, :, 0].ravel() for a in arrays])
    all_y = np.concatenate([a[:, :, 1].ravel() for a in arrays])
    all_z = np.concatenate([a[:, :, 2].ravel() for a in arrays])
    xmin, xmax = np.percentile(all_x, 100 - pct), np.percentile(all_x, pct)
    ymax = np.percentile(all_y, pct)  # floor already at 0
    zmin, zmax = np.percentile(all_z, 100 - pct), np.percentile(all_z, pct)
    cx = (xmin + xmax) / 2; cz = (zmin + zmax) / 2
    span = max(xmax - xmin, ymax, zmax - zmin) * 0.6
    return (cx - span, cx + span), (0, ymax * 1.05), (cz - span, cz + span)


def draw_skeleton(ax, pos_frame, bones, viz, color, alpha=1.0, lw=2.5):
    """pos_frame: (N_joints, 3) meters. Plot: X->ax.X, Z->ax.Y(depth), Y->ax.Z(height)."""
    for b0, b1 in bones:
        if b0 in viz and b1 in viz:
            i0, i1 = viz.index(b0), viz.index(b1)
            p0, p1 = pos_frame[i0], pos_frame[i1]
            ax.plot([p0[0], p1[0]], [p0[2], p1[2]], [p0[1], p1[1]], color=color, lw=lw, alpha=alpha)
    for p in pos_frame:
        ax.scatter([p[0]], [p[2]], [p[1]], color=color, s=15, alpha=alpha, depthshade=False)


def style_ax(ax, title, xlim, ylim, zlim):
    ax.set_xlim(xlim); ax.set_ylim(zlim); ax.set_zlim(ylim)
    # CRITICAL: without this, matplotlib's default 3D axes do NOT preserve
    # equal units-per-axis scaling. Since `axis_limits()` deliberately gives
    # the height (Y) axis a DIFFERENT-sized range than the symmetric X/Z
    # span, straight/rigid bones (verified numerically correct: constant
    # 260mm shoulder-elbow, 250mm elbow-wrist every frame) visually appear
    # bent/kinked purely from this unequal axis scaling -- not a data bug.
    # `set_box_aspect` must be given in the SAME (ax.X, ax.Y, ax.Z) order as
    # set_xlim/set_ylim/set_zlim above, i.e. (xlim, zlim, ylim) per this
    # function's own X->ax.X, Z->ax.Y(depth), Y->ax.Z(height) convention.
    ax.set_box_aspect((xlim[1] - xlim[0], zlim[1] - zlim[0], ylim[1] - ylim[0]))
    ax.set_xlabel('X', fontsize=6, labelpad=1)
    ax.set_ylabel('Z', fontsize=6, labelpad=1)
    ax.set_zlabel('Height', fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5)
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
    ax.view_init(elev=12, azim=45)


C_SOMA = '#4fc3f7'
C_G1 = '#66bb6a'


def process_one(name, bvh_dir, csv_dir, out_path, fps, frame_step):
    bvh_path = os.path.join(bvh_dir, f'{name}.bvh')
    csv_path = os.path.join(csv_dir, f'{name}.csv')
    print(f'\n=== {name} ===')

    joints, offsets, channels, parents, frames = parse_bvh(bvh_path)
    df_g1 = pd.read_csv(csv_path)
    n = min(len(frames), len(df_g1))
    sampled = list(range(0, n, frame_step))
    print(f'  BVH frames: {len(frames)}  CSV rows: {len(df_g1)}  sampled: {len(sampled)}')

    pos_soma = fk_batch(VIZ_JOINTS, joints, offsets, channels, parents, frames, sampled)
    foot_idx = [VIZ_JOINTS.index(n_) for n_ in ('LeftFoot', 'RightFoot')]
    pos_soma[:, :, 1] -= pos_soma[:, foot_idx, 1].min()

    pos_g1 = g1_fk_batch(G1_VIZ, df_g1, sampled)
    foot_g1_idx = [G1_VIZ.index('left_ankle_roll_link'), G1_VIZ.index('right_ankle_roll_link')]
    pos_g1[:, :, 1] -= pos_g1[:, foot_g1_idx, 1].min()

    lim_soma = axis_limits([pos_soma])
    lim_g1 = axis_limits([pos_g1])

    # PERF: create the Figure/Axes ONCE and reuse them for every frame,
    # rather than building a brand-new Figure (+ tight_layout + PNG
    # encode/decode via BytesIO+PIL) per frame. `ax.clear()` + `style_ax()`
    # is much cheaper than re-creating the whole figure, and grabbing the
    # rendered RGBA buffer directly (`canvas.buffer_rgba()`) skips the PNG
    # compress/decompress round-trip entirely. `tight_layout()` is only run
    # ONCE (axis limits/labels don't change frame-to-frame, so the layout
    # is identical every time). Measured ~5-8x speedup over the old
    # per-frame Figure + savefig(bbox_inches='tight') + PIL.open() path.
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    canvas = fig.canvas
    suptitle = fig.suptitle('', fontsize=11, fontweight='bold')
    layout_done = [False]

    def render(fi):
        ax1.clear()
        ax2.clear()
        draw_skeleton(ax1, pos_soma[fi], BONES, VIZ_JOINTS, C_SOMA)
        draw_skeleton(ax2, pos_g1[fi], G1_BONES, G1_VIZ, C_G1)
        style_ax(ax1, 'Source SOMA BVH', *lim_soma)
        style_ax(ax2, 'Retargeted G1 (SOMA Retargeter)', *lim_g1)
        frame_num = sampled[fi]
        suptitle.set_text(f'{name}  ·  frame {frame_num}/{n}  ·  t={frame_num / fps:.2f}s')
        if not layout_done[0]:
            plt.tight_layout()
            layout_done[0] = True
        canvas.draw()
        # buffer_rgba() returns a live memoryview into the canvas's own
        # buffer, so it must be copied (via np.asarray(...).copy()) before
        # the NEXT canvas.draw() overwrites it in place. Accessed via
        # `canvas.renderer` (the Agg renderer) rather than `canvas` itself,
        # since `buffer_rgba()` is specific to FigureCanvasAgg and isn't on
        # the generic FigureCanvasBase type.
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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:],
                        duration=50, loop=0, optimize=False)
    print(f'  saved -> {out_path}  ({N} frames, {time.time() - t0:.0f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bvh_dir', required=True,
                     help="soma-retargeter's source BVH folder, e.g. assets/motions/bvh")
    ap.add_argument('--csv_dir', required=True,
                     help="soma-retargeter's retargeted CSV export folder")
    ap.add_argument('--name', required=True,
                     help='motion basename (without extension), must exist in both dirs')
    ap.add_argument('--out', required=True, help='output .gif path')
    ap.add_argument('--fps', type=float, default=120.0, help='source BVH frame rate')
    ap.add_argument('--frame_step', type=int, default=8,
                     help='render every Nth frame (higher = faster, coarser)')
    args = ap.parse_args()

    process_one(args.name, args.bvh_dir, args.csv_dir, args.out, args.fps, args.frame_step)


if __name__ == '__main__':
    main()
