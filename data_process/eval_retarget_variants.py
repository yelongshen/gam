#!/usr/bin/env python3
"""Evaluate retarget variants on a cropped LAFAN window: CSV -> motion_lib -> arm/torso metrics.
Used by variant_test.sh."""
import os
import subprocess
import sys

import joblib
import numpy as np

REPO = os.path.expanduser("~/gam")
sys.path.insert(0, f"{REPO}/sim2real")
from detect_arm_retarget_errors import ARM_IDX, G1, HW, qc_clip  # noqa: E402

WORK = "/tmp/lafan_var"
NAME = "dance2_subject3"


def first(d):
    return d[next(iter(d))] if isinstance(d, dict) and "dof" not in d else d


def tilt(q):
    x, y, z, w = q
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


def main():
    g = G1()
    for v in sys.argv[1:]:
        csvdir, mldir = f"{WORK}/csv_{v}", f"{WORK}/ml_{v}"
        subprocess.run([os.path.expanduser("~/miniforge3/envs/env_isaaclab/bin/python"),
                        os.path.expanduser("~/GR00T-WholeBodyControl/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py"),
                        "--input", csvdir, "--output", mldir, "--individual", "--fps", "30", "--fps_source", "30"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pk = [os.path.join(dp, f) for dp, _, fs in os.walk(mldir) for f in fs if f.endswith(".pkl")][0]
        os.makedirs(f"{WORK}/robot_{v}", exist_ok=True)
        r = first(joblib.load(pk))
        joblib.dump({NAME: r}, f"{WORK}/robot_{v}/{NAME}.pkl")
        m = qc_clip(g, f"{WORK}/smpl/{NAME}.pkl", f"{WORK}/robot_{v}/{NAME}.pkl", 3)
        dof = np.asarray(r["dof"], dtype=np.float64)
        lo, hi = g.lim[:, 0], g.lim[:, 1]
        at = ((dof < lo + 0.02 * (hi - lo)) | (dof > hi - 0.02 * (hi - lo))).mean(0)
        rot = np.asarray(r["root_rot"])
        tl = np.array([tilt(q) for q in rot[::10]])
        d = np.degrees(dof)
        step = np.abs(np.diff(d[:, ARM_IDX], axis=0)).max(1)
        pinned = {HW[i]: round(float(at[i]), 2) for i in range(29) if at[i] > 0.05}
        print(f"\n[{v}] frames {len(dof)}  root tilt med/p90 {np.median(tl):.1f}/{np.percentile(tl, 90):.1f} deg"
              f"  arm speed {np.abs(np.diff(d[:, ARM_IDX], axis=0)).mean() * 30:.0f} deg/s"
              f"  frames w/ arm step>90deg: {(step > 90).sum()}")
        print(f"     elbow_err_p95 {m['elbow_ang_err_p95']:.1f}  reach_err_p95 {m['reach_err_p95']:.3f}  "
              f"hand_dist_err_p95 {m['hand_dist_err_p95']:.2f}  arm_limit {m['arm_limit_frac']:.2f}  spike {m['arm_spike_deg_s']:.0f}")
        print(f"     pinned joints (>5% frames at limit): {pinned}")
        print(f"     mean|shoulder_yaw| L/R {np.abs(d[:, HW.index('left_shoulder_yaw')]).mean():.0f}/"
              f"{np.abs(d[:, HW.index('right_shoulder_yaw')]).mean():.0f}  wrist rom mean "
              f"{(d[:, [19, 20, 21, 26, 27, 28]].max(0) - d[:, [19, 20, 21, 26, 27, 28]].min(0)).mean():.0f}  "
              f"waist_pitch mean {d[:, HW.index('waist_pitch')].mean():.0f}")


if __name__ == "__main__":
    main()
