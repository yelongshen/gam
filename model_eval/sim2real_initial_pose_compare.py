#!/usr/bin/env python3
"""Compare the initial pose (first N frames) of a run against DEFAULT_ANGLES
and against another reference run, to help distinguish "different policy
default" from "robot mis-calibration" (e.g. an encoder zero-offset error).

Usage:
    .venv_sim/bin/python model_eval/sim2real_initial_pose_compare.py RUN_DIR [--n 5] \
        [--ref RUN_DIR2]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import DEFAULT_ANGLES, JOINT_NAMES  # noqa: E402
from model_eval.sim2real_single_run_quicklook import load  # noqa: E402


def initial_q(run_dir, n):
    q, t = load(run_dir, "q")
    if len(q) == 0:
        return None
    return np.nanmean(q[:n], axis=0)


def report(q0, label, ref=None, ref_label="default_angles"):
    dev = q0 - ref
    order = np.argsort(-np.abs(dev))
    print(f"\n=== {label} vs {ref_label} ===")
    print(f"mean |dev| = {np.abs(dev).mean():.4f} rad ({np.degrees(np.abs(dev).mean()):.2f} deg)")
    print(f"{'joint':15s} {'q0(rad)':>9s} {'ref(rad)':>9s} {'dev(rad)':>9s} {'dev(deg)':>9s}")
    for j in order[:12]:
        print(f"{JOINT_NAMES[j]:15s} {q0[j]:9.4f} {ref[j]:9.4f} {dev[j]:9.4f} "
              f"{np.degrees(dev[j]):9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=5, help="average first N frames")
    ap.add_argument("--ref", help="another run dir to compare against, instead of DEFAULT_ANGLES")
    args = ap.parse_args()

    q0 = initial_q(args.run_dir, args.n)
    if q0 is None:
        print(f"{args.run_dir}: empty/no q data")
        return 1

    meta_path = os.path.join(args.run_dir, "metadata.json")
    title = os.path.basename(args.run_dir.rstrip("/"))
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        title += f" ({meta['robot_config']['model_path']})"

    report(q0, title, DEFAULT_ANGLES, "default_angles")

    if args.ref:
        qref = initial_q(args.ref, args.n)
        if qref is not None:
            report(q0, title, qref, os.path.basename(args.ref.rstrip("/")))


if __name__ == "__main__":
    raise SystemExit(main())
