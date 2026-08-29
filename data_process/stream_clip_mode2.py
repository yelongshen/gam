"""
One-shot Mode-2 ZMQ publisher: stream a SINGLE motion clip (smpl_filtered
.pkl, raw AMASS .npz, or LAFAN .bvh) as proper SMPL joints into
g1_deploy_onnx_ref, then EXIT.

Reproduces the OFFICIAL eval-time transform from
gear_sonic/envs/manager_env/mdp/observations.py::smpl_joints_multi_future_local
(gear_sonic/isaac_utils/rotations.py for the quaternion helpers). This is
shared utility code used both at training/eval time and by the live PICO
pipeline - NOT PICO-specific.

The stored `smpl_joints` in smpl_filtered are root-local (pelvis pinned) but
STILL carry the raw root rotation baked in. The official observation does NOT
stream them verbatim - it further rotates each frame's joints by the INVERSE
of that frame's own root quaternion:

    root_quat = smpl_root_ytoz_up(quat(pose_aa[:, :3]))   # Y-up -> Z-up
    root_quat = remove_smpl_base_rot(root_quat)           # SMPL rest-pose offset
    joints_local = quat_apply(quat_inv(root_quat), stored_smpl_joints)

That `root_quat` is also what's streamed as body_quat_w (smpl_anchor_orientation).

Used by run_sim_eval.py to drive per-clip simulation evaluation.

CLI arguments:
  --path         (required) clip path: smpl_filtered .pkl, raw AMASS .npz, or LAFAN .bvh
  --port         int, default 5556      ZMQ publish port
  --fps          float, default 50.0    playback frame rate
  --loops        int, default 1         number of times to loop the clip
  --settle       float, default 1.0     seconds to hold the first frame before playing
                                        (lets the policy/robot settle into the pose)
  --visualize    flag                   enable the live 3D LiveSkeleton matplotlib viewer
  --vis_every    int, default 5         render every Nth frame in the viewer
  --no_official  flag                   skip the official per-frame root-rotation removal
                                        transform described above (debugging only)

Usage (basic):
  .venv_teleop/bin/python stream_clip_mode2.py --path <clip> --fps 50 [--loops 1]

Usage examples (from data_process/, real clips used in this repo):
  # LAFAN1, with live 3D visualization and a 2s settle before playback
  .venv_teleop/bin/python ./data_process/stream_clip_mode2.py \\
      --path ../ego_dataset/lafan1_smpl_filtered_FIXED/walk1_subject1.pkl \\
      --fps 50 --settle 2.0 --visualize

  # AMASS (smpl_filtered v2)
  .venv_teleop/bin/python ./data_process/stream_clip_mode2.py \\
      --path ../ego_dataset/amass_smpl_filtered_v2/ACCAD__Male2Walking_c3d__B10_-__Walk_turn_left_45_poses.pkl \\
      --fps 50 --settle 2.0 --visualize

  # AMASS (smpl_filtered FPS30 source)
  .venv_teleop/bin/python ./data_process/stream_clip_mode2.py \\
      --path ../ego_dataset/amass_smpl_filtered_FPS30/CMU__CMU__29__29_24_stageii.pkl \\
      --fps 50 --settle 2.0 --visualize

  .venv_teleop/bin/python ./data_process/stream_clip_mode2.py \\
      --path ../ego_dataset/amass_smpl_filtered_FPS30/EyesJapanDataset__Eyes_Japan_Dataset__kudo__walk-23-shuffle_oneleg-kudo_stageii.pkl \\
      --fps 50 --settle 2.0 --visualize

Prerequisites for any of the above (full sim2sim pipeline):
  1. MuJoCo sim already running:
       env -u CYCLONEDDS_HOME .venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py
  2. g1_deploy_onnx_ref running with --input-type zmq, with ']' then Enter sent
     via stdin to enter CONTROL state and enable ZMQ streaming mode.
  3. THEN run this script to publish the clip over ZMQ (port 5556 by default);
     it streams the clip once and exits.

Orchestrated (called as a subprocess) by:
  - model_eval/run_sim_eval.py        (per-clip batch simulation evaluation)
  - model_eval/visualize_tracker_sim.py (paired with sim tracker recording)

Reused as a library (not directly invoked) by:
  - data_process/normalize_split_test.py
  - data_process/convert_lafan_to_smpl_filtered.py
  both `import stream_clip_mode2 as S` to reuse `official_root_quat_w()`,
  `_YTOZ`, `_BASE_CONJ` for consistent Y-up->Z-up root-rotation math shared
  between the offline conversion pipeline and this live streaming path.
"""
import argparse
import os
import sys
import time
import numpy as np
import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling imports

import classify_motions as C
import normalize_split_test as N
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


# ── Official eval_agent_trl.py / observations.py transform ─────────────────
_YTOZ = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])   # aa[pi/2,0,0]
_BASE_CONJ = np.array([0.5, -0.5, -0.5, -0.5])                        # conj([.5,.5,.5,.5])


def _aa_to_quat(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    ax = np.where(th < 1e-8, 0.0, aa / np.maximum(th, 1e-12))
    return np.concatenate([np.cos(th / 2), ax * np.sin(th / 2)], -1)


def _qmul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], -1)


def _qconj(q):
    c = q.copy()
    c[..., 1:] *= -1
    return c


def _qapply(q, v):
    w = q[..., 0:1]
    u = q[..., 1:]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def official_root_quat_w(pose_aa_root):
    """pose_aa_root: (T,3) root axis-angle -> (T,4) Z-up root quat [w,x,y,z]."""
    T = len(pose_aa_root)
    q = _aa_to_quat(pose_aa_root)
    q = _qmul(np.broadcast_to(_YTOZ, (T, 4)), q)       # smpl_root_ytoz_up
    q = _qmul(q, np.broadcast_to(_BASE_CONJ, (T, 4)))  # remove_smpl_base_rot
    return q


def official_encoder_joints(stored_smpl_joints, pose_aa_root):
    """Reproduce smpl_joints_multi_future_local(): rotate the STORED (pinned)
    smpl_joints by the inverse of each frame's own root quaternion.
    Returns (joints_local (T,24,3), root_quat_w (T,4)).

    root_quat_w is now derived via fk_anchor_quat() (see its docstring) rather
    than from the raw pose_aa root rotation, so it is always geometrically
    consistent with the joints being streamed."""
    root_q = official_root_quat_w(pose_aa_root)
    qi = np.repeat(_qconj(root_q)[:, None, :], stored_smpl_joints.shape[1], axis=1)
    joints_local = _qapply(qi, stored_smpl_joints)
    return joints_local, root_q



SMPL_LINKS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
    (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
    (20, 22), (21, 23),
]


class LiveSkeleton:
    """Lightweight real-time 3D view of the SMPL joints being streamed.

    Drawing is far slower than the 50 Hz send loop, so we redraw from the
    GUI/main thread while the send loop runs in a worker thread: the ZMQ
    timing stays authoritative and is never blocked by matplotlib.
    """

    def __init__(self, joints, every=5, title=""):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.every = max(1, every)

        lo = joints.reshape(-1, 3).min(0)
        hi = joints.reshape(-1, 3).max(0)
        pad = 0.1

        plt.ion()
        self.fig = plt.figure(figsize=(5.5, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(lo[0] - pad, hi[0] + pad)
        self.ax.set_ylim(lo[1] - pad, hi[1] + pad)
        self.ax.set_zlim(lo[2] - pad, hi[2] + pad)
        self.ax.set_box_aspect((hi - lo) + pad)
        self.ax.set_title(title, fontsize=9)

        # Color-code joints by body side so turning is visible directly on
        # the skeleton itself (no separate arrow needed): LEFT-side joints
        # (1,4,7,10,13,16,18,20,22 - hip/knee/ankle/toe/shoulder/arm/
        # forearm/hand chain) are drawn orange, RIGHT-side joints are drawn
        # teal, and midline joints (pelvis/spine/neck/head) stay navy. When
        # the body turns/rotates, the orange (left) and teal (right) sides
        # visibly swap which one faces the camera/viewer, making left-right
        # turning unambiguous even in a static-camera 3D view.
        LEFT_JOINTS = {1, 4, 7, 10, 13, 16, 18, 20, 22}
        RIGHT_JOINTS = {2, 5, 8, 11, 14, 17, 19, 21, 23}
        colors = []
        for j in range(24):
            if j in LEFT_JOINTS:
                colors.append('darkorange')
            elif j in RIGHT_JOINTS:
                colors.append('teal')
            else:
                colors.append('navy')
        self._joint_colors = colors

        p = joints[0]
        self.scat = self.ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                                    c=colors, s=28, depthshade=False)
        self.lines = []
        for a, b in SMPL_LINKS:
            if a in LEFT_JOINTS or b in LEFT_JOINTS:
                c = 'darkorange'
            elif a in RIGHT_JOINTS or b in RIGHT_JOINTS:
                c = 'teal'
            else:
                c = 'navy'
            self.lines.append(self.ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]],
                                           [p[a, 2], p[b, 2]], c=c, lw=2.5)[0])

        # Fixed camera angle (not auto-rotating top-down) so rotation of the
        # BODY - not the view - is what you see frame to frame.
        self.ax.view_init(elev=15, azim=-60)

        self.fig.canvas.draw()
        self.fig.show()

    def update_now(self, i, joints):
        """Redraw immediately (called from the GUI/main thread)."""
        i = min(i, len(joints) - 1)
        p = joints[i]
        self.scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        for ln, (a, b) in zip(self.lines, SMPL_LINKS):
            ln.set_data_3d([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]],
                           [p[a, 2], p[b, 2]])
        self.ax.set_title(f"streaming frame {i}/{len(joints)}  "
                          f"(orange=LEFT side, teal=RIGHT side)", fontsize=9)

    def update(self, i, joints):
        if i % self.every:
            return
        self.update_now(i, joints)
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def close(self):
        try:
            self.plt.close(self.fig)
        except Exception:
            pass


def load_stream_data(path, fps, official=True):
    """Return (joints (T,24,3), pose_aa (T,72), transl (T,3), root_quat (T,4)).

    Accepted inputs:
      *.pkl  smpl_filtered / normalize_split_test output -> already normalized
      *.npz  raw AMASS                                   -> normalized here
      *.bvh  LAFAN                                       -> normalized here
    """
    if path.endswith('.pkl'):
        try:
            import pickle
            with open(path, 'rb') as fh:
                d = pickle.load(fh)
        except Exception:
            import joblib
            d = joblib.load(path)
        joints = np.asarray(d['smpl_joints'], dtype=np.float64)
        pose_aa = np.asarray(d['pose_aa'], dtype=np.float64)
        trans = np.asarray(d['transl'], dtype=np.float64)
        src = float(d.get('fps', 50.0))
        if abs(src - fps) > 1e-6:
            step = max(1, int(round(src / fps)))
            joints, pose_aa, trans = joints[::step], pose_aa[::step], trans[::step]
        kind = "pre-normalized (smpl_filtered)"
    elif path.endswith('.npz'):
        pose_aa, trans = N.load_clip(path)          # resampled to 50 fps
        joints = N.to_local_zup(pose_aa)
        kind = "AMASS -> root-local Z-up"
    else:                                            # LAFAN .bvh
        j = C.load_joints(path)
        if j is None:
            return None, None, None, None, None
        joints = j - j[:, 0:1, :] + N.PELVIS_OFFSET
        pose_aa = np.zeros((len(joints), 72))
        trans = j[:, 0, :].copy()
        kind = "LAFAN -> root-local"

    # `joints_world` is kept UN-de-rotated (root yaw/heading intact) purely
    # for visualization: `official_encoder_joints()` below removes each
    # frame's own root rotation (that's what the policy needs - a
    # root-relative local frame), which erases exactly the left/right
    # turning signal a viewer wants to see. Streaming still uses the
    # de-rotated `joints`.
    joints_world = joints.copy()

    if official:
        joints, root_quat = official_encoder_joints(joints, pose_aa[:, :3])
        kind += " + official per-frame root de-rotation"
    else:
        root_quat = np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]),
                                    (len(joints), 4)).copy()
        joints_world = joints.copy()

    print(f"[clip] {os.path.basename(path)}  {joints.shape[0]} frames  ({kind})"
          f"  pelvis={joints[:, 0, :].mean(0).round(3)}")
    return joints, pose_aa, trans, root_quat, joints_world


def build_frames(joints, pose_aa, trans, root_quat):
    """joints:(T,24,3) meters. Return list of packed messages, 4-frame lookahead.

    NOTE: protocol v3 requires EVERY per-frame field to carry the same frame
    count (4). Sending joint_vel as (1,29) or frame_index as (1,) makes the C++
    decoder abort with "Version 3 frame count mismatch" and silently ignore the
    whole stream.
    """
    T = joints.shape[0]
    NF = 4  # lookahead chunk size
    cache = []
    for i in range(T):
        idxs = [min(i + off, T - 1) for off in range(NF)]
        d = {
            "smpl_joints": np.zeros((NF, 24, 3), dtype=np.float32),
            "smpl_pose":  np.zeros((NF, 21, 3), dtype=np.float32),
            "body_quat_w": np.zeros((NF, 4), dtype=np.float32),
            "joint_pos": np.zeros((NF, 29), dtype=np.float32),
            "joint_vel": np.zeros((NF, 29), dtype=np.float32),
            "vr_position": trans[i].astype(np.float32),
            "vr_orientation": np.array([1, 0, 0, 0], dtype=np.float32),
            "frame_index": np.array(idxs, dtype=np.int64),
            "left_trigger": np.zeros(1, dtype=np.float32),
            "right_trigger": np.zeros(1, dtype=np.float32),
            "left_grip": np.zeros(1, dtype=np.float32),
            "right_grip": np.zeros(1, dtype=np.float32),
            "pico_dt": np.array([1.0 / 50.0], dtype=np.float64),
            "pico_fps": np.array([50.0], dtype=np.float64),
            "timestamp_realtime": np.array([time.time()], dtype=np.float64),
            "timestamp_monotonic": np.array([time.monotonic()], dtype=np.float64),
            "left_hand_joints": np.zeros((7,), dtype=np.float32),
            "right_hand_joints": np.zeros((7,), dtype=np.float32),
            "toggle_data_collection": np.zeros(1, dtype=bool),
            "toggle_data_abort": np.zeros(1, dtype=bool),
            "heading_increment": np.zeros(1, dtype=np.float32),
        }
        for off, idx in enumerate(idxs):
            d["smpl_joints"][off] = joints[idx].astype(np.float32)
            d["body_quat_w"][off] = root_quat[idx].astype(np.float32)
            if pose_aa is not None:
                d["smpl_pose"][off] = pose_aa[idx, 3:3 + 63].reshape(21, 3)
        cache.append(pack_pose_message(d, topic="pose"))
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True)
    ap.add_argument('--port', type=int, default=5556)
    ap.add_argument('--fps', type=float, default=50.0)
    ap.add_argument('--loops', type=int, default=1)
    ap.add_argument('--settle', type=float, default=1.0,
                    help='seconds to hold first frame so the robot can settle')
    ap.add_argument('--visualize', action='store_true',
                    help='show the streamed SMPL skeleton live in 3D')
    ap.add_argument('--vis_every', type=int, default=5,
                    help='redraw every N streamed frames (default 5 = ~10 Hz)')
    ap.add_argument('--no_official', action='store_true',
                    help='disable the official per-frame root de-rotation '
                         '(sends stored/pinned joints verbatim; body_quat_w=identity)')
    args = ap.parse_args()

    joints, pose_aa, trans, root_quat, joints_world = load_stream_data(
        args.path, args.fps, official=not args.no_official)
    if joints is None:
        print("FAILED to load clip:", args.path)
        return 1

    viz = None
    if args.visualize:
        try:
            viz = LiveSkeleton(joints_world, every=args.vis_every,
                               title=os.path.basename(args.path))
        except Exception as e:
            print(f"[viz] disabled ({e})")

    cache = build_frames(joints, pose_aa, trans, root_quat)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)
    try:
        sock.bind(f"tcp://127.0.0.1:{args.port}")
    except Exception:
        sock.bind(f"tcp://*:{args.port}")
    time.sleep(0.3)  # allow subscriber to connect

    dt = 1.0 / args.fps

    def stream_loop(state):
        """Publish at a strict `fps`. Never touches the GUI."""
        t_end = time.time() + args.settle
        while time.time() < t_end:                    # settle on frame 0
            sock.send(cache[0])
            time.sleep(dt)
        for _ in range(args.loops):
            for i, msg in enumerate(cache):
                t0 = time.time()
                sock.send(msg)
                state['i'] = i
                d = dt - (time.time() - t0)
                if d > 0:
                    time.sleep(d)
        state['done'] = True

    state = {'i': 0, 'done': False}

    if viz is None:
        stream_loop(state)
    else:
        import threading
        th = threading.Thread(target=stream_loop, args=(state,), daemon=True)
        th.start()
        while not state['done']:
            viz.update_now(state['i'], joints_world)
            viz.plt.pause(0.05)
        th.join(timeout=5)

    print(f"Streamed {len(cache)} frames x{args.loops} for {args.path}")
    if viz is not None:
        viz.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
