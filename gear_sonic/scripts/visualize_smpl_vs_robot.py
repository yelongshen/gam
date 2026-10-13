"""Side-by-side SMPL vs retargeted-G1 visualization, with a floor grid.

LEFT  : SMPL source  -- `smpl_filtered` pkl (pelvis-pinned `smpl_joints` + real
        `transl`), i.e. exactly what was fed into the retargeter.
RIGHT : retargeted G1 -- GMR pkl (`root_pos`, `root_rot`, `local_body_pos`).

Both panels share a common floor at z = 0 and a synchronized frame index, so a
retargeting failure (feet floating, body sinking, limbs diverging from the
source) is visible directly.

Usage:
    # one clip
    python gear_sonic/scripts/visualize_smpl_vs_robot.py --clip pico0928_clip_002

    # every clip in the dataset
    python gear_sonic/scripts/visualize_smpl_vs_robot.py --all
"""

from __future__ import annotations

import argparse
import glob
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SMPL_DIR = os.environ.get("VIS_SMPL_DIR", "/home/grease/gam/logs_pkl/picoset_20260928")
ROBOT_DIR = os.environ.get("VIS_ROBOT_DIR", "/home/grease/GMR/picoset_20260928_retargeted_g1")

# SMPL 24-joint kinematic tree.
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]
SMPL_LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}
SMPL_RIGHT = {2, 5, 8, 11, 14, 17, 19, 21, 23}

# G1 link tree, by name (parent -> child), resolved against link_body_list.
G1_TREE = [
    ("pelvis", "left_hip_pitch_link"), ("left_hip_pitch_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"), ("left_ankle_roll_link", "left_toe_link"),
    ("pelvis", "right_hip_pitch_link"), ("right_hip_pitch_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"), ("right_ankle_roll_link", "right_toe_link"),
    ("pelvis", "torso_link"), ("torso_link", "head_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_elbow_link"),
    ("left_elbow_link", "left_rubber_hand"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_elbow_link"),
    ("right_elbow_link", "right_rubber_hand"),
]
G1_LEFT = {"left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link",
           "left_toe_link", "left_shoulder_pitch_link", "left_elbow_link",
           "left_rubber_hand"}
G1_RIGHT = {"right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link",
            "right_toe_link", "right_shoulder_pitch_link", "right_elbow_link",
            "right_rubber_hand"}

PELVIS_OFFSET = np.array([0.003, -0.351, 0.012])


def quat_apply_xyzw(q, v):
    """Rotate v (..., J, 3) by quaternion q (..., 4) in (x, y, z, w) order."""
    xyz = q[..., 0:3][..., None, :]
    w = q[..., 3:4][..., None, :]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def load_smpl(name):
    """smpl_filtered pkl -> world joints (T, 24, 3)."""
    d = joblib.load(os.path.join(SMPL_DIR, name + ".pkl"))
    J = np.asarray(d["smpl_joints"], dtype=np.float64)
    t = np.asarray(d["transl"], dtype=np.float64)
    # joints are pelvis-pinned at PELVIS_OFFSET; transl carries world position
    world = J - PELVIS_OFFSET + t[:, None, :]
    return world, float(d["fps"])


def load_robot(name):
    """GMR retargeted pkl -> world link positions (T, 38, 3) + link names."""
    d = joblib.load(os.path.join(ROBOT_DIR, name + ".pkl"))
    root_pos = np.asarray(d["root_pos"], dtype=np.float64)
    root_rot = np.asarray(d["root_rot"], dtype=np.float64)   # (T,4) xyzw
    local = np.asarray(d["local_body_pos"], dtype=np.float64)
    world = quat_apply_xyzw(root_rot, local) + root_pos[:, None, :]
    names = [str(x) for x in d["link_body_list"]]
    return world, names, float(d["fps"])


def draw_floor(ax, xlim, ylim, step=0.5):
    corners = np.array([
        [xlim[0], ylim[0], 0.0], [xlim[1], ylim[0], 0.0],
        [xlim[1], ylim[1], 0.0], [xlim[0], ylim[1], 0.0],
    ])
    ax.add_collection3d(Poly3DCollection([corners], facecolor="0.85",
                                         alpha=0.45, zorder=0))
    for x in np.arange(np.floor(xlim[0]), xlim[1] + step, step):
        ax.plot([x, x], ylim, [0, 0], color="0.6", lw=0.5, zorder=1)
    for y in np.arange(np.floor(ylim[0]), ylim[1] + step, step):
        ax.plot(xlim, [y, y], [0, 0], color="0.6", lw=0.5, zorder=1)


def setup(ax, xlim, ylim, zmax, title):
    ax.clear()
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(0, zmax)
    try:
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zmax))
    except Exception:
        pass
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=18, azim=-60)
    ax.set_title(title, fontsize=11)
    draw_floor(ax, xlim, ylim)


def render(name, out_dir, stride=3, max_frames=400, fps=30):
    S, s_fps = load_smpl(name)
    Rb, names, r_fps = load_robot(name)
    n = min(len(S), len(Rb))
    S, Rb = S[:n], Rb[:n]

    # shared floor: drop each to its own robust floor (p5 of lowest point)
    S[..., 2] -= np.percentile(S[:, [10, 11], 2].min(axis=1), 5)
    toe = [names.index("left_toe_link"), names.index("right_toe_link")]
    Rb[..., 2] -= np.percentile(Rb[:, toe, 2].min(axis=1), 5)

    idx = np.arange(0, n, stride)
    if len(idx) > max_frames:
        idx = idx[:: int(np.ceil(len(idx) / max_frames))]

    nm = {k: i for i, k in enumerate(names)}
    pad = 0.6
    xs = np.concatenate([S[idx, :, 0].ravel(), Rb[idx, :, 0].ravel()])
    ys = np.concatenate([S[idx, :, 1].ravel(), Rb[idx, :, 1].ravel()])
    xlim = (xs.min() - pad, xs.max() + pad)
    ylim = (ys.min() - pad, ys.max() + pad)
    zmax = max(S[idx][..., 2].max(), Rb[idx][..., 2].max()) + 0.3

    fig = plt.figure(figsize=(16, 7))
    axl = fig.add_subplot(121, projection="3d")
    axr = fig.add_subplot(122, projection="3d")

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}_smpl_vs_robot.mp4")
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, out, dpi=90):
        for i in idx:
            setup(axl, xlim, ylim, zmax, f"SMPL source   frame {i+1}/{n}")
            j = S[i]
            for c, p in enumerate(SMPL_PARENTS):
                if p < 0:
                    continue
                col = ("tab:red" if c in SMPL_LEFT else
                       "tab:green" if c in SMPL_RIGHT else "0.2")
                axl.plot(*zip(j[p], j[c]), color=col, lw=2.5)
            axl.scatter(j[:, 0], j[:, 1], j[:, 2], s=12, c="k", depthshade=False)
            axl.plot(S[: i + 1, 0, 0], S[: i + 1, 0, 1],
                     np.zeros(i + 1), color="tab:blue", lw=1.0, alpha=0.35)

            setup(axr, xlim, ylim, zmax, f"retargeted G1   frame {i+1}/{n}")
            b = Rb[i]
            for p, c in G1_TREE:
                if p not in nm or c not in nm:
                    continue
                col = ("tab:red" if c in G1_LEFT else
                       "tab:green" if c in G1_RIGHT else "0.2")
                axr.plot(*zip(b[nm[p]], b[nm[c]]), color=col, lw=2.5)
            axr.scatter(b[:, 0], b[:, 1], b[:, 2], s=8, c="k", depthshade=False)
            axr.plot(Rb[: i + 1, nm["pelvis"], 0], Rb[: i + 1, nm["pelvis"], 1],
                     np.zeros(i + 1), color="tab:blue", lw=1.0, alpha=0.35)

            writer.grab_frame()
    plt.close(fig)
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="/home/grease/g1_robot_data/picoset_20260928_vis")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    if a.all:
        names = sorted(os.path.splitext(os.path.basename(f))[0]
                       for f in glob.glob(os.path.join(SMPL_DIR, "*.pkl")))
    else:
        names = [a.clip]

    for k, nm in enumerate(names, 1):
        try:
            out, n = render(nm, a.out, a.stride, a.max_frames, a.fps)
            print(f"[{k}/{len(names)}] {nm}: {n} frames -> {out}")
        except Exception as e:  # noqa: BLE001
            print(f"[{k}/{len(names)}] {nm}: ERROR {e}")


if __name__ == "__main__":
    main()
