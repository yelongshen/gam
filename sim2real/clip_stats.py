#!/usr/bin/env python3
"""Compute the `sim2real/eval_clip_manifest.md` §3 clip-property row for a clip.

Usage:
    python sim2real/clip_stats.py reach_jump_R_001__A072_M high_jump_R_103__A389_M
"""
import os
import sys

import joblib
import numpy as np

ROOT = os.path.expanduser("~/ego_dataset/eval_subset")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HW_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ARM = [i for i, n in enumerate(HW_JOINTS) if "shoulder" in n or "elbow" in n or "wrist" in n]
LEG = [i for i, n in enumerate(HW_JOINTS) if "hip" in n or "knee" in n or "ankle" in n]


def stats(name):
    smpl = joblib.load(f"{ROOT}/smpl/{name}.pkl")
    rob = joblib.load(f"{ROOT}/robot/{name}.pkl")[name]

    n_smpl = len(smpl["transl"])
    fps_s = float(smpl.get("fps", 50))
    dof = np.asarray(rob["dof"], dtype=np.float64)
    fps_r = float(rob.get("fps", 30))
    T = len(dof)

    deg = np.rad2deg(dof)
    root = np.asarray(rob["root_trans_offset"], dtype=np.float64)
    d = np.diff(root, axis=0)
    path = float(np.linalg.norm(d, axis=1).sum())
    speed = np.linalg.norm(d, axis=1) * fps_r
    jspeed = np.abs(np.diff(deg, axis=0)) * fps_r
    rom = deg.max(0) - deg.min(0)

    print(f"### {name}")
    print(f"  SMPL frames      : {n_smpl} ({n_smpl / fps_s:.2f} s @ {fps_s:g} fps)")
    print(f"  robot dof        : {dof.shape} ({T / fps_r:.2f} s @ {fps_r:g} fps)")
    print(f"  dof range        : {deg.min():.1f} … +{deg.max():.1f} deg")
    disp = root[-1] - root[0]
    print(f"  root disp        : ({disp[0]:+.2f}, {disp[1]:+.2f}, {disp[2]:+.2f}) m")
    print(f"  path (3D)        : {path:.2f} m")
    print(f"  height range     : {root[:, 2].max() - root[:, 2].min():.3f} m")
    print(f"  root speed m/pk  : {speed.mean():.2f} / {speed.max():.2f} m/s")
    print(f"  joint speed m/pk : {jspeed.mean():.1f} / {jspeed.max():.0f} deg/s")
    print(f"  ROM arms / legs  : {rom[ARM].mean():.1f} / {rom[LEG].mean():.1f} deg")
    top = np.argsort(rom)[::-1][:6]
    print("  top ROM joints   : "
          + ", ".join(f"{HW_JOINTS[i]} {rom[i]:.0f}" for i in top))

    for n in ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
              "left_wrist_roll_joint", "right_wrist_roll_joint"):
        i = HW_JOINTS.index(n)
        print(f"  ROM {n:26s}: {rom[i]:.1f} deg")
    print(f"  dof min joint    : {HW_JOINTS[deg.min(0).argmin()]}")
    print(f"  dof max joint    : {HW_JOINTS[deg.max(0).argmax()]}")

    # load_stream_data() prints the de-rotated pelvis mean itself.
    try:
        from data_process.stream_clip_mode2 import load_stream_data
        load_stream_data(f"{ROOT}/smpl/{name}.pkl", 50)
    except Exception as exc:  # pragma: no cover - diagnostic helper
        print(f"  pelvis mean (derot): unavailable ({exc})")
    print()


if __name__ == "__main__":
    for n in sys.argv[1:]:
        stats(n)
