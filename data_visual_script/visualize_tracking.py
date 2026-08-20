"""
Side-by-side motion-tracking visualization:
   LEFT  : human SMPL target skeleton (what we streamed over ZMQ)
   RIGHT : G1 robot as actually simulated (rendered from the logged q.csv)

Only the tracked window (encoder_mode == 2) is used, so both panels correspond
to the portion where the policy was really following the streamed motion.

Usage:
  .venv_sim/bin/python visualize_tracking.py \
      --clip ../egodata/downloads/amass/extracted/.../0015_sitting1_poses.npz \
      --log_dir gear_sonic_deploy/logs/fixtest3 \
      --out data/evaluation_visualization_set/tracking_demo.mp4
"""
import os
import sys
import argparse
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_process"))
import classify_motions as C

SMPL_LINKS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
    (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]
SCENE = "gear_sonic_deploy/g1/scene_29dof.xml"


def load_csv(path):
    """Deploy CSVs have 5 leading timing columns; drop them."""
    return np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)[:, 5:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', required=True)
    ap.add_argument('--log_dir', required=True)
    ap.add_argument('--out',
                    default='data/evaluation_visualization_set/tracking_demo.mp4')
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--max_frames', type=int, default=400)
    ap.add_argument('--full', action='store_true',
                    help='render the WHOLE episode (spawn/settle + tracking) '
                         'instead of only the tracked window, so you can see '
                         'whether a fall happens before tracking even starts')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ── robot side ──────────────────────────────────────────────────────────
    q = load_csv(os.path.join(args.log_dir, 'q.csv'))             # (T,29)
    bq = load_csv(os.path.join(args.log_dir, 'base_quat.csv'))    # (T,4) w,x,y,z
    enc = load_csv(os.path.join(args.log_dir, 'encoder_mode.csv'))[:, 0]
    tracked = np.where(enc == 2)[0]
    if len(tracked) == 0:
        raise SystemExit("No tracked (encoder_mode==2) frames in this log!")
    if args.full:
        lo, hi = 0, len(q) - 1
        track_start = int(tracked.min()) - lo
    else:
        lo, hi = int(tracked.min()), int(tracked.max())
        track_start = 0
    q, bq = q[lo:hi + 1], bq[lo:hi + 1]
    Tr = q.shape[0]
    print(f"[robot] window {lo}..{hi}  ({Tr} frames), tracking starts at {track_start}")

    # ── human side (same FK the publisher streamed) ─────────────────────────
    hj = C.load_joints(args.clip)          # (Th,24,3) meters, Z-up
    Th = hj.shape[0]
    print(f"[human] {Th} FK frames")

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=480)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135, -15
    cam.lookat[:] = [0, 0, 0.8]

    # Geoms used to find the robot's lowest point (exclude the floor plane).
    _not_plane = np.array(
        [model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE
         for i in range(model.ngeom)], dtype=bool)

    def set_robot_pose(fi):
        """Set base orientation + 29 joints, then drop the root so the lowest
        geom rests on the floor.

        The deploy logs contain base_quat but NOT base position, so a fixed
        root height makes the robot appear to float (or sink) whenever it
        crouches/kneels. We instead solve the root height from forward
        kinematics so the lowest body always touches z=0.
        """
        data.qpos[:] = 0
        data.qpos[0:3] = [0, 0, 0]
        data.qpos[3:7] = bq[fi]                 # w,x,y,z
        data.qpos[7:7 + 29] = q[fi]
        mujoco.mj_forward(model, data)
        zs = data.geom_xpos[_not_plane, 2] - model.geom_rbound[_not_plane]
        data.qpos[2] = -float(zs.min())          # ground the robot
        mujoco.mj_forward(model, data)

    frames = list(range(0, min(Tr, args.max_frames), args.stride))
    print(f"[render] {len(frames)} frames -> {args.out}")

    # per-frame diagnostics for the metrics strip
    tau = load_csv(os.path.join(args.log_dir, 'motor_torque.csv'))[lo:hi + 1]
    tau_series = np.abs(tau).mean(axis=1)
    w = bq[:, 1] ** 2 + bq[:, 2] ** 2
    tilt_series = np.degrees(np.arccos(np.clip(1 - 2 * w, -1, 1)))

    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], hspace=0.28)
    axh = fig.add_subplot(gs[0, 0], projection='3d')
    axr = fig.add_subplot(gs[0, 1])
    axm = fig.add_subplot(gs[1, :])
    floor = float(np.percentile(hj[:, C.FOOT_JOINTS, 2], 5))

    def update(k):
        fi = frames[k]
        axh.clear()
        axr.clear()
        axm.clear()

        # human target (map robot frame index -> human frame index)
        if fi < track_start:
            hidx = 0
            phase = "SETTLE (not tracking yet)"
        else:
            span = max(Tr - 1 - track_start, 1)
            hidx = min(Th - 1, int((fi - track_start) / span * (Th - 1)))
            phase = "TRACKING"
        p = hj[hidx]
        axh.scatter(p[:, 0], p[:, 1], p[:, 2], c='deepskyblue', s=14)
        for a, b in SMPL_LINKS:
            axh.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]],
                     [p[a, 2], p[b, 2]], c='navy', lw=1.8)
        c0 = p[0]
        r = 1.0
        axh.set_xlim(c0[0] - r, c0[0] + r)
        axh.set_ylim(c0[1] - r, c0[1] + r)
        axh.set_zlim(floor - 0.05, floor + 1.9)
        axh.set_axis_off()
        axh.set_title(f"TARGET — human SMPL (streamed)\nframe {hidx}/{Th}",
                      fontsize=10)

        # robot: base orientation + 29 logged joints, grounded on the floor
        set_robot_pose(fi)
        renderer.update_scene(data, camera=cam)
        axr.imshow(renderer.render())
        axr.set_axis_off()
        tilt_now = tilt_series[fi]
        fallen = tilt_now > 45
        axr.set_title(f"ACHIEVED — G1 in MuJoCo  [{phase}]\n"
                      f"frame {fi}/{Tr}   tilt={tilt_now:.0f}°"
                      + ("   ⚠ FALLEN" if fallen else ""),
                      fontsize=10, color=('red' if fallen else 'black'))

        # metrics strip with a moving cursor
        t = np.arange(Tr) / 50.0
        axm.plot(t, tilt_series, c='crimson', lw=1.2, label='base tilt (deg)')
        axm.plot(t, tau_series, c='seagreen', lw=1.2, label='mean |torque| (N·m)')
        axm.axhline(45, color='crimson', ls=':', lw=1.0, alpha=0.7)
        if track_start > 0:
            axm.axvline(track_start / 50.0, color='blue', lw=1.5, ls='-',
                        alpha=0.7, label='tracking starts')
        axm.axvline(fi / 50.0, color='k', lw=1.4, ls='--')
        axm.set_xlim(0, t[-1])
        axm.set_xlabel("time (s)", fontsize=9)
        axm.legend(loc='upper right', fontsize=8, ncol=3)
        axm.grid(alpha=0.3)
        axm.set_title(f"tilt={tilt_now:.1f}°   torque={tau_series[fi]:.1f} N·m",
                      fontsize=9)

        fig.suptitle(os.path.basename(args.clip), fontsize=11)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=40)
    ani.save(args.out, writer='ffmpeg', fps=max(5, 50 // args.stride), dpi=110)
    plt.close()
    print("saved", args.out)


if __name__ == "__main__":
    main()
