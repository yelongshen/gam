#!/usr/bin/env python3
"""Render reference videos: SMPL stick figure (left) | G1 robot reference replay (right).

The robot panel is a kinematic MuJoCo replay of the retargeted `robot/` pkl
(root_trans_offset, root_rot [xyzw], dof in HW joint order), i.e. the reference the
policy is asked to track, NOT a simulation. The SMPL panel draws `smpl_joints`
(Z-up) of the `smpl/` pkl. Both panels follow the pelvis.

Usage (needs MUJOCO_GL=egl, run with .venv_sim):
    MUJOCO_GL=egl .venv_sim/bin/python sim2real/render_refs.py \
        --smpl-dir ~/ego_dataset/amass_evalset/smpl --robot-dir ~/ego_dataset/amass_evalset/robot \
        --names-file sim2real/icl_hard_set/amass11.txt --out sim2real/icl_hard_set/ref_videos/amass11
"""
import argparse
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = f"{REPO}/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
HW = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch",
      "left_ankle_roll", "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
      "right_ankle_pitch", "right_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
      "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
      "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch",
      "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
      "right_wrist_pitch", "right_wrist_yaw"]
LINKS = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10),
         (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19),
         (18, 20), (19, 21), (20, 22), (21, 23)]
SZ = 480


def first(d):
    return d[next(iter(d))] if isinstance(d, dict) and "dof" not in d else d


def build_model_with_floor():
    """G1 model + an infinite checkerboard floor at z=0 (the reference data has its lowest body at z=0)
    + a light and a sky-ish background, so jumps / crouches / foot contact are readable."""
    spec = mujoco.MjSpec.from_file(XML)
    spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.22, 0.30, 0.40], rgb2=[0.14, 0.21, 0.30],
                     markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    spec.add_material(name="floor_mat", texrepeat=[6, 6], texuniform=True, reflectance=0.15)
    mat = spec.material("floor_mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    spec.worldbody.add_geom(name="ref_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                            material="floor_mat", contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8], castshadow=False)
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]
    return spec.compile()


def smpl_panel(fig, ax, joints, root):
    ax.cla()
    c = root
    ax.set_xlim(c[0] - 1.2, c[0] + 1.2)
    ax.set_ylim(c[1] - 1.2, c[1] + 1.2)
    ax.set_zlim(0, 2.0)
    ax.set_box_aspect((1, 1, 0.83))
    ax.view_init(elev=15, azim=-60)
    for a, b in LINKS:
        ax.plot([joints[a, 0], joints[b, 0]], [joints[a, 1], joints[b, 1]],
                [joints[a, 2], joints[b, 2]], "-o", ms=2, lw=2, c="tab:blue")
    g = np.array([[c[0] - 1.2, c[1] - 1.2], [c[0] + 1.2, c[1] + 1.2]])
    ax.plot([g[0, 0], g[1, 0]], [g[0, 1], g[0, 1]], [0, 0], c="gray", lw=0.5)
    ax.set_title("SMPL (streamed)", fontsize=9)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def render(name, smpl_dir, robot_dir, out_dir, model, rend, data, jadr, fps_out=30, win_s=0.0):
    out = f"{out_dir}/{name}.mp4"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return "skip"
    s = joblib.load(f"{smpl_dir}/{name}.pkl")
    r = first(joblib.load(f"{robot_dir}/{name}.pkl"))
    sj = np.asarray(s["smpl_joints"], dtype=np.float64)
    sfps = float(s.get("fps", 50))
    dof = np.asarray(r["dof"], dtype=np.float64)
    root = np.asarray(r["root_trans_offset"], dtype=np.float64)
    rot = np.asarray(r["root_rot"], dtype=np.float64)  # xyzw
    rfps = float(r.get("fps", 30))
    T = len(dof)
    tr = np.asarray(s["transl"], dtype=np.float64)
    fig = plt.figure(figsize=(SZ / 100, SZ / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{2 * SZ}x{SZ}", "-r", str(fps_out), "-i", "-", "-pix_fmt", "yuv420p", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.0, 150, -10
    step = max(1, int(round(rfps / fps_out)))
    t0, t1 = 0, T
    if win_s and T / rfps > win_s + 4:
        # hardest `win_s` window by pelvis-local SMPL joint speed (skip first 2 s)
        loc = sj - sj[:, :1]
        v = np.linalg.norm(np.diff(loc, axis=0), axis=2).mean(1)
        W = int(win_s * sfps)
        cs = np.concatenate([[0], np.cumsum(v)])
        sc = cs[W:] - cs[:-W]
        lo = int(2 * sfps)
        a = int(np.argmax(sc[lo:])) + lo
        t0 = int(a / sfps * rfps)
        t1 = min(T, t0 + int(win_s * rfps))
    for t in range(t0, t1, step):
        data.qpos[:] = 0
        data.qpos[0:3] = root[t]
        q = rot[t]
        data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        for i, a in enumerate(jadr):
            data.qpos[a] = dof[t, i]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = root[t]
        rend.update_scene(data, camera=cam)
        right = rend.render()
        ts = min(int(round(t / rfps * sfps)), len(sj) - 1)
        left = smpl_panel(fig, ax, sj[ts], sj[ts, 0])
        left = left[:SZ, :SZ]
        p.stdin.write(np.concatenate([left, right], axis=1).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    plt.close(fig)
    return f"{(t1 - t0) / rfps:.1f}s window @{t0 / rfps:.0f}s of {T / rfps:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl-dir", required=True)
    ap.add_argument("--robot-dir", required=True)
    ap.add_argument("--names-file", help="one clip name per line (default: all in robot-dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="render only the hardest N-second window (0 = whole clip)")
    args = ap.parse_args()
    smpl_dir, robot_dir = (os.path.expanduser(args.smpl_dir), os.path.expanduser(args.robot_dir))
    os.makedirs(args.out, exist_ok=True)
    if args.names_file:
        names = [l.strip() for l in open(args.names_file) if l.strip()]
    else:
        names = sorted(f[:-4] for f in os.listdir(robot_dir) if f.endswith(".pkl"))
    model = build_model_with_floor()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, SZ)
    model.vis.global_.offheight = max(model.vis.global_.offheight, SZ)
    data = mujoco.MjData(model)
    jadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint")]
            for n in HW]
    rend = mujoco.Renderer(model, SZ, SZ)
    for i, n in enumerate(names):
        try:
            print(f"[{i + 1}/{len(names)}] {n}: {render(n, smpl_dir, robot_dir, args.out, model, rend, data, jadr, win_s=args.seconds)}",
                  flush=True)
        except Exception as exc:
            print(f"[{i + 1}/{len(names)}] {n}: FAILED {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
