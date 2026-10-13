"""Visualize a recorded PICO SMPL session (motion + floor).

Reconstructs WORLD-space SMPL joints from the recorded npz stream:

    joints_world = R(body_quat_w) * smpl_joints_local + transl_zup

where `smpl_joints_local` is root-local & z-up (the server already applied
`smpl_root_ytoz_up` + removed the base rotation), and `body_pos_w` is the raw
PICO root position in a Y-UP frame, so it is converted to Z-UP here.

Outputs:
  <out>/motion.mp4        animated skeleton over a floor grid
  <out>/trajectory.png    top-down path + height/か trajectory plots

Usage:
    python gear_sonic/scripts/visualize_pico_motion.py \
        /home/grease/g1_robot_data/pico_raw/20260928_162215 --stride 5
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# SMPL 24-joint kinematic tree (parent of each joint; -1 for the root).
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]
# Left/right colouring so the skeleton is readable.
LEFT_JOINTS = {1, 4, 7, 10, 13, 16, 18, 20, 22}
RIGHT_JOINTS = {2, 5, 8, 11, 14, 17, 19, 21, 23}


def quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v (..., J, 3) by quaternion q (..., 4) in (w, x, y, z) order."""
    w = q[..., 0:1][..., None, :]
    xyz = q[..., 1:4][..., None, :]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def yup_to_zup(p: np.ndarray) -> np.ndarray:
    """PICO Y-up (x, y, z) -> Z-up (x, -z, y), preserving handedness."""
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def load_world_joints(session_dir: str, stride: int = 1):
    """Load either a session directory of pose_*.npz, or a single clip npz."""
    if os.path.isfile(session_dir) and session_dir.endswith(".npz"):
        d = np.load(session_dir, allow_pickle=True)
        local = d["smpl_joints"][::stride].astype(np.float64)
        quat = d["body_quat_w"][::stride].astype(np.float64)
        root = d["body_pos_w"][::stride].astype(np.float64)
        ts = d["timestamp_monotonic"].reshape(-1)[::stride].astype(np.float64)
        files = [session_dir]
    else:
        files = sorted(glob.glob(os.path.join(session_dir, "pose_*.npz")))
        if not files:
            raise FileNotFoundError(f"no pose_*.npz in {session_dir}")
        files = files[::stride]

        local, quat, root, ts = [], [], [], []
        for f in files:
            d = np.load(f, allow_pickle=True)
            local.append(d["smpl_joints"][-1])
            quat.append(d["body_quat_w"][-1])
            root.append(d["body_pos_w"][-1] if "body_pos_w" in d.files else np.zeros(3))
            ts.append(float(d["timestamp_monotonic"][0]))
        local = np.asarray(local, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        root = np.asarray(root, dtype=np.float64)
        ts = np.asarray(ts)

    root = yup_to_zup(root)  # (T, 3) z-up

    # Rotate root-local joints back into world orientation, then translate.
    world = quat_apply_wxyz(quat, local) + root[:, None, :]

    # Drop the floor to z = 0 using the lowest joint across the whole take.
    world[..., 2] -= world[..., 2].min()
    root = world[:, 0, :]
    return world, root, ts - ts[0], files


def draw_floor(ax, xlim, ylim, step: float = 0.5):
    """Draw a shaded floor plane at z=0 with a grid."""
    corners = np.array([
        [xlim[0], ylim[0], 0.0],
        [xlim[1], ylim[0], 0.0],
        [xlim[1], ylim[1], 0.0],
        [xlim[0], ylim[1], 0.0],
    ])
    ax.add_collection3d(
        Poly3DCollection([corners], facecolor="0.85", alpha=0.45, zorder=0)
    )
    for x in np.arange(np.floor(xlim[0]), xlim[1] + step, step):
        ax.plot([x, x], ylim, [0, 0], color="0.6", lw=0.5, zorder=1)
    for y in np.arange(np.floor(ylim[0]), ylim[1] + step, step):
        ax.plot(xlim, [y, y], [0, 0], color="0.6", lw=0.5, zorder=1)


def animate(world, root, t, out_path: str, fps: int = 30, trail: int = 60):
    T = world.shape[0]
    pad = 0.6
    xlim = (world[..., 0].min() - pad, world[..., 0].max() + pad)
    ylim = (world[..., 1].min() - pad, world[..., 1].max() + pad)
    zmax = max(world[..., 2].max() + 0.3, 2.0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    writer_cls = FFMpegWriter
    if not out_path.endswith(".mp4"):
        writer_cls = PillowWriter
    writer = writer_cls(fps=fps)

    with writer.saving(fig, out_path, dpi=100):
        for i in range(T):
            ax.clear()
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(0, zmax)
            try:
                ax.set_box_aspect(
                    (xlim[1] - xlim[0], ylim[1] - ylim[0], zmax)
                )
            except Exception:
                pass
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            ax.view_init(elev=18, azim=-60)

            draw_floor(ax, xlim, ylim)

            # root path so far
            s = max(0, i - trail)
            ax.plot(root[: i + 1, 0], root[: i + 1, 1], np.zeros(i + 1),
                    color="tab:blue", lw=1.0, alpha=0.35)
            ax.plot(root[s : i + 1, 0], root[s : i + 1, 1], root[s : i + 1, 2],
                    color="tab:orange", lw=1.5, alpha=0.8)

            j = world[i]
            for c, p in enumerate(SMPL_PARENTS):
                if p < 0:
                    continue
                if c in LEFT_JOINTS:
                    col = "tab:red"
                elif c in RIGHT_JOINTS:
                    col = "tab:green"
                else:
                    col = "0.2"
                ax.plot(*zip(j[p], j[c]), color=col, lw=2.5)
            ax.scatter(j[:, 0], j[:, 1], j[:, 2], s=14, c="k", depthshade=False)
            # vertical drop line from pelvis to the floor
            ax.plot([j[0, 0]] * 2, [j[0, 1]] * 2, [0, j[0, 2]],
                    color="tab:blue", ls=":", lw=1.0)

            ax.set_title(f"frame {i + 1}/{T}   t = {t[i]:6.2f}s   "
                         f"root = ({j[0,0]:.2f}, {j[0,1]:.2f}, {j[0,2]:.2f}) m")
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_trajectory(world, root, t, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(root[:, 0], root[:, 1], lw=1.0)
    ax.scatter(root[0, 0], root[0, 1], c="g", s=60, label="start", zorder=5)
    ax.scatter(root[-1, 0], root[-1, 1], c="r", s=60, label="end", zorder=5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("top-down root path (floor plane)")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(t, root[:, 0], label="x")
    ax.plot(t, root[:, 1], label="y")
    ax.plot(t, root[:, 2], label="z (height)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("m")
    ax.set_title("root position vs time"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[2]
    lowest = world[..., 2].min(axis=1)
    ax.plot(t, lowest, lw=0.8, label="lowest joint")
    ax.plot(t, world[:, 15, 2], lw=0.8, label="head")
    ax.axhline(0, color="k", ls="--", lw=1, label="floor")
    ax.set_xlabel("t (s)"); ax.set_ylabel("z (m)")
    ax.set_title("floor contact check"); ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--out", default=None, help="output dir (default: <session>_vis)")
    ap.add_argument("--stride", type=int, default=5, help="use every Nth frame")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--gif", action="store_true", help="write .gif instead of .mp4")
    args = ap.parse_args()

    out_dir = args.out or (
        os.path.splitext(args.session_dir.rstrip("/"))[0] + "_vis"
    )
    os.makedirs(out_dir, exist_ok=True)

    world, root, t, files = load_world_joints(args.session_dir, args.stride)
    print(f"loaded {world.shape[0]} frames (stride {args.stride}) from {len(files)} files")
    print(f"world joint bounds  x:[{world[...,0].min():.2f},{world[...,0].max():.2f}] "
          f"y:[{world[...,1].min():.2f},{world[...,1].max():.2f}] "
          f"z:[{world[...,2].min():.2f},{world[...,2].max():.2f}]")

    plot_trajectory(world, root, t, os.path.join(out_dir, "trajectory.png"))

    if world.shape[0] > args.max_frames:
        k = int(np.ceil(world.shape[0] / args.max_frames))
        world, root, t = world[::k], root[::k], t[::k]
        print(f"animating {world.shape[0]} frames (extra stride {k})")

    name = "motion.gif" if args.gif else "motion.mp4"
    animate(world, root, t, os.path.join(out_dir, name), fps=args.fps)


if __name__ == "__main__":
    main()
