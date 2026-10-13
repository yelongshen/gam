#!/usr/bin/env python3
"""Normalize + validate regenerated LAFAN robot pkls in <set>/robot_fixed.

 * strips the `pkl__` prefix the batch tools add to file names and dict keys
 * validates: real motion (dof speed), joint limits, root height / floor, duration vs SMPL
 * writes the normalized files in place; prints a summary and CHECK lines

Usage: .venv_sim/bin/python data_process/normalize_lafan_robot_fixed.py lafan1_evalset [lafan1_trainset]
"""
import glob
import os
import sys

import joblib
import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = f"{REPO}/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
HW = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch",
      "left_ankle_roll", "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
      "right_ankle_pitch", "right_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
      "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
      "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch",
      "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
      "right_wrist_pitch", "right_wrist_yaw"]


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    lim = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint")] for n in HW])
    for ds in sys.argv[1:]:
        root = os.path.expanduser(f"~/ego_dataset/{ds}")
        rows = []
        for f in sorted(glob.glob(f"{root}/robot_fixed/*.pkl")):
            base = os.path.basename(f)[:-4]
            n = base[5:] if base.startswith("pkl__") else base
            d = joblib.load(f)
            e = d[next(iter(d))] if "dof" not in d else d
            new = f"{root}/robot_fixed/{n}.pkl"
            joblib.dump({n: e}, new)
            if new != f:
                os.remove(f)
            dof = np.asarray(e["dof"])
            deg = np.rad2deg(dof)
            fps = float(e.get("fps", 30))
            sp = np.abs(np.diff(deg, axis=0)).mean() * fps
            over = float(((dof < lim[:, 0] - 1e-3) | (dof > lim[:, 1] + 1e-3)).mean())
            z = np.asarray(e["root_trans_offset"])[:, 2]
            s = joblib.load(f"{root}/smpl/{n}.pkl")
            rows.append((n, sp, float((deg.max(0) - deg.min(0)).mean()), over, z.min(), z.max(),
                         len(dof) / fps, len(s["transl"]) / float(s["fps"])))
        r = np.array([x[1:] for x in rows], dtype=float)
        print(f"== {ds}: {len(rows)} clips")
        print("  dof speed deg/s min/med/max: %.1f / %.1f / %.1f" % (r[:, 0].min(), np.median(r[:, 0]), r[:, 0].max()))
        print("  mean ROM deg min/med: %.1f / %.1f" % (r[:, 1].min(), np.median(r[:, 1])))
        print("  frames over joint limit (max over clips): %.4f" % r[:, 2].max())
        print("  root z min/max: %.2f / %.2f" % (r[:, 3].min(), r[:, 4].max()))
        print("  max duration mismatch vs SMPL: %.2f s" % np.abs(r[:, 5] - r[:, 6]).max())
        for x in rows:
            if x[1] < 10 or x[3] > 0.001 or abs(x[6] - x[7]) > 1 or x[4] < 0.2:
                print("  CHECK %s speed %.1f over %.4f z %.2f-%.2f dur %.1f vs %.1f" %
                      (x[0], x[1], x[3], x[4], x[5], x[6], x[7]))


if __name__ == "__main__":
    main()
