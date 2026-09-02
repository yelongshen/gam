"""
Visualize the low-latency motion tracker in the MuJoCo simulation.

Runs the full pipeline on ONE clip and renders a video of the result using the
TRUE simulated state (base position + orientation + joints) captured over DDS,
so world translation is shown correctly:

    sim (sim_loop_eval.py)  ->  policy (g1_deploy_onnx_ref, Mode 2)
                            ->  stream_clip_mode2.py (human SMPL over ZMQ)
                            ->  record_sim_state.py (DDS odometry + joints)

Usage:
  .venv_sim/bin/python visualize_tracker_sim.py \
      --clip ../egodata/downloads/amass/extracted/.../walk.npz \
      --out data/evaluation_visualization_set/tracker_walk.mp4
"""
import os
import sys
import time
import signal
import argparse
import subprocess

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import model_eval.backup_run_sim_eval as R
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_process"))
import classify_motions as C

SCENE = "gear_sonic_deploy/g1/scene_29dof.xml"
SMPL_LINKS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
    (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]


def load_target_joints(clip, fps=50.0):
    """Return (T,24,3) SMPL joints for either an smpl_filtered .pkl (which
    already stores smpl_joints) or a raw AMASS/LAFAN file (needs FK)."""
    if clip.endswith('.pkl'):
        import joblib
        d = joblib.load(clip)
        j = np.asarray(d['smpl_joints'], dtype=np.float64)
        src = float(d.get('fps', 50.0))
        if abs(src - fps) > 1e-6:
            j = j[::max(1, int(round(src / fps)))]
        return j
    return C.load_joints(clip)


def run_and_record(clip, tag, state_out, settle=1.0, fps=50.0):
    """Start policy, activate tracking, stream the clip, record DDS state."""
    logs_dir = f"logs/{tag}"
    os.system(f"rm -rf {os.path.join(R.DEPLOY, logs_dir)}")
    plog = f"/tmp/policy_{tag}.log"
    env = dict(os.environ); env.pop("CYCLONEDDS_HOME", None)
    lf = open(plog, "wb")
    pol = subprocess.Popen([R.BIN] + R.POLICY_ARGS + ["--logs-dir", logs_dir],
                           cwd=R.DEPLOY, stdin=subprocess.PIPE, stdout=lf,
                           stderr=subprocess.STDOUT, env=env)
    try:
        if not R.wait_for(plog, "Init Done", timeout=90):
            raise SystemExit("policy failed to initialize")
        pol.stdin.write(b']'); pol.stdin.flush(); time.sleep(1.5)
        pol.stdin.write(b'\n'); pol.stdin.flush(); time.sleep(1.5)
        R.wait_for(plog, "ZMQ STREAMING MODE: ENABLED", timeout=10)

        # how long is the clip?
        j = C.load_joints(clip)
        dur = j.shape[0] / fps + settle + 1.5

        env2 = dict(os.environ); env2.pop("CYCLONEDDS_HOME", None)
        rec = subprocess.Popen([R.SIM_PY, "model_eval/record_sim_state.py",
                                "--out", state_out, "--duration", str(dur)],
                               cwd=R.REPO, env=env2)
        time.sleep(1.0)  # let the recorder latch on
        subprocess.run([R.TELEOP_PY, "data_process/stream_clip_mode2.py", "--path", clip,
                        "--fps", str(fps), "--settle", str(settle)],
                       cwd=R.REPO, env=env2, timeout=dur + 60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rec.wait(timeout=90)
    finally:
        pol.send_signal(signal.SIGINT)
        try: pol.wait(timeout=15)
        except subprocess.TimeoutExpired: pol.kill()
        lf.close()
    return os.path.join(R.DEPLOY, logs_dir)


def render(clip, state_npz, out, stride=2):
    d = np.load(state_npz)
    pos, quat, q = d['base_pos'], d['base_quat'], d['q']
    T = len(pos)
    hj = C.load_joints(clip)
    Th = hj.shape[0]

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=480)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.5, 130, -12

    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (quat[:, 1]**2 + quat[:, 2]**2), -1, 1)))
    # tracking is assumed to start once the base starts moving appreciably
    frames = list(range(0, T, stride))

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3)
    axh = fig.add_subplot(gs[0, 0], projection='3d')
    axr = fig.add_subplot(gs[0, 1])
    axm = fig.add_subplot(gs[1, :])
    floor = float(np.percentile(hj[:, C.FOOT_JOINTS, 2], 5))

    def update(k):
        fi = frames[k]
        axh.clear(); axr.clear(); axm.clear()

        hidx = min(Th - 1, int(fi / max(T - 1, 1) * (Th - 1)))
        p = hj[hidx]
        axh.scatter(p[:, 0], p[:, 1], p[:, 2], c='deepskyblue', s=14)
        for a, b in SMPL_LINKS:
            axh.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]],
                     [p[a, 2], p[b, 2]], c='navy', lw=1.8)
        c0 = p[0]; r = 1.0
        axh.set_xlim(c0[0]-r, c0[0]+r); axh.set_ylim(c0[1]-r, c0[1]+r)
        axh.set_zlim(floor-0.05, floor+1.9); axh.set_axis_off()
        axh.set_title(f"TARGET — human SMPL\nframe {hidx}/{Th}", fontsize=10)

        # TRUE simulated state: real base position + orientation + joints
        data.qpos[:] = 0
        data.qpos[0:3] = pos[fi]
        data.qpos[3:7] = quat[fi]
        data.qpos[7:7+29] = q[fi]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [pos[fi][0], pos[fi][1], 0.7]   # follow the robot
        renderer.update_scene(data, camera=cam)
        axr.imshow(renderer.render()); axr.set_axis_off()
        fallen = tilt[fi] > 45
        axr.set_title(f"ACHIEVED — G1 in MuJoCo (true sim state)\n"
                      f"h={pos[fi][2]:.2f}m tilt={tilt[fi]:.0f}°"
                      + ("  ⚠ FALLEN" if fallen else ""),
                      fontsize=10, color=('red' if fallen else 'black'))

        t = np.arange(T) / 50.0
        axm.plot(t, tilt, c='crimson', lw=1.2, label='base tilt (deg)')
        axm.plot(t, pos[:, 2] * 100, c='navy', lw=1.2, label='base height (cm)')
        axm.axhline(45, color='crimson', ls=':', lw=1.0, alpha=0.6)
        axm.axvline(fi / 50.0, color='k', lw=1.4, ls='--')
        axm.set_xlim(0, t[-1]); axm.set_xlabel("time (s)", fontsize=9)
        axm.legend(loc='upper right', fontsize=8, ncol=2); axm.grid(alpha=0.3)
        fig.suptitle(os.path.basename(clip), fontsize=11)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=40)
    ani.save(out, writer='ffmpeg', fps=max(5, int(50 / stride)), dpi=110)
    plt.close()
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='tracker')
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--no_reset', action='store_true')
    args = ap.parse_args()

    clip = R.abspath(args.clip)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not args.no_reset:
        print("[sim] resetting simulator for a clean standing start...")
        R.restart_sim()
    state = f"/tmp/state_{args.tag}.npz"
    run_and_record(clip, args.tag, state)
    render(clip, state, args.out, stride=args.stride)


if __name__ == "__main__":
    main()
