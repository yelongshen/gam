#!/usr/bin/env python3
"""Time-windowed breakdown of a G1 run: is the wobble a single transient,
a steady limit cycle, or growing (instability)?
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES  # noqa: E402
from model_eval.sim2real_single_run_quicklook import load  # noqa: E402


def main():
    run_dir = sys.argv[1]
    win_s = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    dt = meta["logging"]["dt"]

    q, t = load(run_dir, "q")
    dq, _ = load(run_dir, "dq")
    n = min(len(q), len(dq))
    q, dq, t = q[:n], dq[:n], t[:n]

    watch = ["R_knee", "L_knee", "L_elbow", "R_sho_yaw", "waist_roll", "waist_pitch"]
    jidx = [JOINT_NAMES.index(nm) for nm in watch]

    win = int(win_s / dt)
    print(f"{'t_start(s)':>10s}  " + "  ".join(f"{nm:>18s}" for nm in watch))
    print(f"{'':>10s}  " + "  ".join(f"{'amp(rad) maxdq':>18s}" for _ in watch))
    for start in range(0, n - win, win):
        seg_q = q[start:start + win]
        seg_dq = dq[start:start + win]
        row = []
        for j in jidx:
            amp = seg_q[:, j].max() - seg_q[:, j].min()
            mdq = np.abs(seg_dq[:, j]).max()
            row.append(f"{amp:7.3f}/{mdq:7.2f}   ")
        print(f"{t[start] / 1000.0:10.1f}  " + " ".join(row))


if __name__ == "__main__":
    raise SystemExit(main())
