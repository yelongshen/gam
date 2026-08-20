"""
Visualize a recorded PICO teleop dataset (logs/yelong_cliptest_*).

Each frame is a `pose_*.npz` written by the PICO manager and contains a
4-frame lookahead chunk. We take frame 0 of each chunk to rebuild the
trajectory and animate:

  LEFT  : SMPL body skeleton (24 joints, root-relative, Z-up)
  RIGHT : VR 3-point tracking (head + both wrists) and trigger/grip state

Usage:
  .venv_sim/bin/python visualize_pico.py --dir logs/yelong_cliptest_0 \
      --out data/evaluation_visualization_set/pico_clip0.mp4
"""
import os
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SMPL_LINKS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
    (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
    (20, 22), (21, 23),
]


def load_sequence(d):
    files = sorted(glob.glob(os.path.join(d, "pose_*.npz")))
    if not files:
        raise SystemExit(f"no pose_*.npz in {d}")
    joints, vr, trig, fidx, ts = [], [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        joints.append(z['smpl_joints'][0])          # (24,3) first of the chunk
        vr.append(z['vr_position'])                 # (9,) L|R|H
        trig.append([float(z['left_trigger'][0]), float(z['right_trigger'][0]),
                     float(z['left_grip'][0]), float(z['right_grip'][0])])
        fidx.append(int(z['frame_index'][0]))
        ts.append(float(z['timestamp_monotonic'][0]))
    fps = float(np.load(files[0])['pico_fps'][0])
    return (np.asarray(joints), np.asarray(vr).reshape(-1, 3, 3),
            np.asarray(trig), np.asarray(fidx), np.asarray(ts), fps, len(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max_frames', type=int, default=900)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    J, VR, TRIG, FI, TS, fps, n = load_sequence(args.dir)
    dur = (TS[-1] - TS[0])
    print(f"[pico] {n} frames @ {fps:.1f} fps  ({dur:.1f}s)  joints={J.shape}")

    frames = list(range(0, min(len(J), args.max_frames), args.stride))
    fig = plt.figure(figsize=(12, 6))
    axs = fig.add_subplot(121, projection='3d')
    axv = fig.add_subplot(122, projection='3d')

    lo = J.reshape(-1, 3).min(0)
    hi = J.reshape(-1, 3).max(0)
    vlo = VR.reshape(-1, 3).min(0)
    vhi = VR.reshape(-1, 3).max(0)
    labels = ['L wrist', 'R wrist', 'head']
    colors = ['tab:red', 'tab:blue', 'tab:green']

    def update(k):
        i = frames[k]
        axs.clear(); axv.clear()

        p = J[i]
        axs.scatter(p[:, 0], p[:, 1], p[:, 2], c='deepskyblue', s=16)
        for a, b in SMPL_LINKS:
            axs.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]],
                     [p[a, 2], p[b, 2]], c='navy', lw=2.0)
        axs.set_xlim(lo[0], hi[0]); axs.set_ylim(lo[1], hi[1]); axs.set_zlim(lo[2], hi[2])
        axs.set_box_aspect((hi - lo))
        axs.set_axis_off()
        axs.set_title(f"SMPL body (24 joints)\nframe {i}/{len(J)}  idx={FI[i]}", fontsize=10)

        v = VR[i]
        for j in range(3):
            axv.scatter(*v[j], c=colors[j], s=90, label=labels[j])
        # trail of the head for context
        s = max(0, i - 60)
        axv.plot(VR[s:i+1, 2, 0], VR[s:i+1, 2, 1], VR[s:i+1, 2, 2],
                 c='tab:green', alpha=0.4, lw=1)
        axv.set_xlim(vlo[0], vhi[0]); axv.set_ylim(vlo[1], vhi[1]); axv.set_zlim(vlo[2], vhi[2])
        axv.legend(loc='upper left', fontsize=8)
        lt, rt, lg, rg = TRIG[i]
        axv.set_title(f"VR 3-point tracking\n"
                      f"trig L={lt:.2f} R={rt:.2f}   grip L={lg:.2f} R={rg:.2f}",
                      fontsize=10)
        fig.suptitle(os.path.basename(args.dir.rstrip('/')), fontsize=12)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=40)
    ani.save(args.out, writer='ffmpeg', fps=max(5, int(fps / args.stride)), dpi=110)
    plt.close()
    print("saved", args.out)


if __name__ == "__main__":
    main()
