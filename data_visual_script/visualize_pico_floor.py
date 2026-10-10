#!/usr/bin/env python3
"""visualize_pico_floor.py — PICO capture viewer WITH a ground plane, showing the
true world posture (standing / sitting / crawling).

Why this exists
---------------
`visualize_pico.py` plots `smpl_joints` as stored, which is **root-relative AND
de-rotated** (`pico_manager_thread_server.process_smpl_joints()` applies
`quat_apply(quat_inv(body_quat_w), joints)`). That throws away the body's world
orientation, so a crawl and a stand look almost identical:

    clip [12635:12900)   as-stored     torso tilt 13 deg   <- looks upright
                         world-oriented torso tilt 64 deg  <- actually crawling

This script re-applies `body_quat_w` to recover world orientation, then places the
skeleton on a floor by lifting it so the lowest joint rests at z=0. That makes the
posture legible: standing -> feet on floor; sitting -> pelvis low, feet forward;
crawling -> hands AND knees both on the floor, torso horizontal.

Floor height caveat
-------------------
The capture has no absolute pelvis height (`transl` is all-zero), so the floor is
*inferred* per frame as "lowest joint touches the ground". That is exact whenever
some joint really is in contact (which is almost always) but it cannot represent a
genuine airborne phase -- during a jump the whole figure stays glued to the floor.
Use the printed `lift` trace to see when this assumption is doing work.

Usage
-----
    .venv_sim/bin/python data_visual_script/visualize_pico_floor.py \
        --dir /home/grease/g1_robot_data/pico_raw/20260927_174628 \
        --start 12635 --end 12900 --out /tmp/crawl.mp4
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation as R

SMPL_LINKS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
    (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
    (20, 22), (21, 23),
]
LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}
FOOT, HAND, KNEE = [7, 8, 10, 11], [20, 21, 22, 23], [4, 5]

# Joint CENTRES sit inside the body; the contact SURFACE is lower. Without these
# offsets a hand resting on the floor reads ~3 cm up and a kneeling knee ~6 cm up,
# so nothing ever looks like it is touching. Values are rough anthropometrics.
SURFACE_OFFSET = {7: 0.07, 8: 0.07,      # ankle centre -> sole
                  10: 0.03, 11: 0.03,    # toe
                  4: 0.06, 5: 0.06,      # knee centre -> kneecap front
                  20: 0.03, 21: 0.03,    # wrist centre -> palm
                  22: 0.02, 23: 0.02}    # hand
CONTACT = [4, 5, 7, 8, 10, 11, 20, 21, 22, 23]


def load(d, start, end, stride):
    fs = sorted(glob.glob(os.path.join(d, "pose_*.npz")))[start:end:stride]
    if not fs:
        raise SystemExit(f"no frames in {d}[{start}:{end}]")
    J, Q = [], []
    for f in fs:
        z = np.load(f)
        J.append(z["smpl_joints"][0])
        Q.append(z["body_quat_w"][0])          # wxyz
    return np.asarray(J), np.asarray(Q)


def to_world_on_floor(J, Q, per_frame=False):
    """Re-apply world rotation, then place the figure on a floor at z=0.

    The floor is a SINGLE height for the whole clip (10th percentile of the
    per-frame lowest contact-joint surface), not a per-frame argmin. Per-frame
    argmin forces whichever joint happens to be lowest to exactly 0 and lets
    everything else float, which makes a 4-contact crawl look like a 1-contact
    balance act and hides genuine reconstruction errors. A fixed floor lets
    joints sit slightly above or below it, which is what you want to see.
    """
    rot = R.from_quat(np.asarray(Q)[:, [1, 2, 3, 0]])      # wxyz -> xyzw
    W = np.stack([rot[t].apply(J[t]) for t in range(len(J))])

    surf = W[:, :, 2].copy()                               # contact-surface heights
    for j, off in SURFACE_OFFSET.items():
        surf[:, j] -= off
    per_frame_low = surf[:, CONTACT].min(axis=1)
    floor = per_frame_low.min() if per_frame else np.percentile(per_frame_low, 10)
    W[:, :, 2] -= floor
    surf -= floor
    return W, surf, per_frame_low - floor


def contact_count(surf, tol=0.05):
    """How many contact joints are within `tol` of the floor each frame."""
    return (np.abs(surf[:, CONTACT]) < tol).sum(axis=1)


def posture(W):
    """Per-frame posture label for the on-screen readout.

    The crawl test is deliberately strict on HAND HEIGHT. A deep forward bend and
    a crawl both reach 80-90 deg of torso tilt, so tilt alone over-fires badly
    (it labelled 161/265 frames of a clip whose real crawl is ~60 frames). The
    discriminator is whether the hands are *load-bearing*: in a crawl they are
    among the lowest joints, so after the floor-lift `hand_z ~ 0`, whereas a
    bent-over stand keeps them ~0.2 m up with the pelvis still high:

        bent-over stand   tilt 84 deg   hand_z 0.19   pelvis 0.68
        true crawl        tilt 80 deg   hand_z 0.00   pelvis 0.44
    """
    torso = W[:, 12] - W[:, 0]
    tilt = np.degrees(np.arccos(np.clip(torso[:, 2] / np.linalg.norm(torso, axis=1), -1, 1)))
    pelvis = W[:, 0, 2]
    hands_bearing = W[:, HAND, 2].min(axis=1) < 0.10     # hands ON the floor
    out = []
    for t in range(len(W)):
        if tilt[t] > 50 and hands_bearing[t] and pelvis[t] < 0.60:
            out.append("CRAWL")
        elif tilt[t] > 50 and pelvis[t] > 0.60:
            out.append("BEND")
        elif pelvis[t] < 0.55:
            out.append("SIT/CROUCH")
        else:
            out.append("STAND")
    return out, tilt, pelvis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=float, default=50.0, help="chunk rate (recorder --target_fps)")
    ap.add_argument("--max_frames", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    J, Q = load(args.dir, args.start, args.end, args.stride)
    J, Q = J[:args.max_frames], Q[:args.max_frames]
    W, surf, low = to_world_on_floor(J, Q)
    nc = contact_count(surf)
    lab, tilt, pelvis = posture(W)
    print(f"[floor-vis] {len(W)} frames  pelvis {pelvis.min():.2f}-{pelvis.max():.2f} m  "
          f"tilt {tilt.min():.0f}-{tilt.max():.0f} deg")
    print(f"[floor-vis] contacts/frame (within 5cm of floor): mean {nc.mean():.1f}  "
          f"min {nc.min()}  max {nc.max()}   lowest-joint offset {low.min():+.2f}..{low.max():+.2f} m")
    from collections import Counter
    print(f"[floor-vis] posture: {dict(Counter(lab))}")

    fig = plt.figure(figsize=(11, 5.5))
    ax3 = fig.add_subplot(121, projection="3d")
    axs = fig.add_subplot(122, projection="3d")

    rad = 1.0
    for ax, (elev, azim), ttl in ((ax3, (18, -70), "perspective"), (axs, (2, 0), "side view")):
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(ttl)

    def draw(i):
        for ax, (elev, azim) in ((ax3, (18, -70)), (axs, (2, 0))):
            ax.cla()
            ax.view_init(elev=elev, azim=azim)
            cx, cy = W[i, 0, 0], W[i, 0, 1]
            # floor grid
            g = np.linspace(-rad, rad, 9)
            for v in g:
                ax.plot([cx - rad, cx + rad], [cy + v, cy + v], [0, 0], c="0.85", lw=.5, zorder=0)
                ax.plot([cx + v, cx + v], [cy - rad, cy + rad], [0, 0], c="0.85", lw=.5, zorder=0)
            for a, b in SMPL_LINKS:
                c = "tab:blue" if (a in LEFT or b in LEFT) else "tab:red"
                ax.plot(*[[W[i, a, k], W[i, b, k]] for k in range(3)], c=c, lw=2)
            # contact joints: filled green when within 5 cm of the floor, hollow otherwise
            touch = [j for j in CONTACT if abs(surf[i, j]) < 0.05]
            free = [j for j in CONTACT if abs(surf[i, j]) >= 0.05]
            if touch:
                ax.scatter(*W[i, touch].T, c="lime", s=55, edgecolors="k", zorder=5)
            if free:
                ax.scatter(*W[i, free].T, facecolors="none", edgecolors="0.4", s=30)
            ax.scatter(*W[i, 0].T, c="k", s=28)
            ax.set_xlim(cx - rad, cx + rad); ax.set_ylim(cy - rad, cy + rad); ax.set_zlim(0, 2 * rad)
            ax.set_box_aspect((1, 1, 1)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_zticks([0, .5, 1.0, 1.5])
        t = (args.start + i * args.stride) / args.fps
        ax3.set_title(f"{lab[i]}   t={t:.1f}s   chunk={args.start + i * args.stride}")
        axs.set_title(f"pelvis {pelvis[i]:.2f} m   tilt {tilt[i]:.0f}$\\degree$   "
                      f"contacts {nc[i]}")

    ani = animation.FuncAnimation(fig, draw, frames=len(W), interval=40)
    ani.save(args.out, writer="ffmpeg", fps=max(5, int(args.fps / args.stride)), dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
