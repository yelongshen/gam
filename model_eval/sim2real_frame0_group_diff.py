#!/usr/bin/env python3
"""Mean frame-0 q across a list of "old" run dirs vs a list of "new" run dirs,
ranked by |difference|, across all 29 joints."""
import sys
sys.path.insert(0, "/home/grease/gam")
import numpy as np
from data_process.g1_params import JOINT_NAMES, DEFAULT_ANGLES
from model_eval.sim2real_single_run_quicklook import load

def frame0s(dirs, mode0mean=False):
    out = []
    for d in dirs:
        q, t = load(d, "q")
        if len(q) == 0:
            continue
        if mode0mean:
            mode, _ = load(d, "encoder_mode", ncols=1)
            n = min(len(q), len(mode))
            m0 = mode[:n, 0] == 0
            out.append(q[:n][m0].mean(0) if m0.sum() else q[0])
        else:
            out.append(q[0])
    return np.stack(out) if out else np.zeros((0, 29))

old_dirs = sys.argv[1].split(",")
new_dirs = sys.argv[2].split(",")
mode0mean = "--mode0mean" in sys.argv
old = frame0s(old_dirs, mode0mean)
new = frame0s(new_dirs, mode0mean)

old_mean, new_mean = old.mean(0), new.mean(0)
diff = new_mean - old_mean
order = np.argsort(-np.abs(diff))

print(f"old: {len(old_dirs)} runs, new: {len(new_dirs)} runs\n")
print(f"{'joint':15s} {'old_mean':>9s} {'new_mean':>9s} {'diff(rad)':>10s} {'diff(deg)':>10s} {'default':>9s}")
for j in order:
    print(f"{JOINT_NAMES[j]:15s} {old_mean[j]:9.4f} {new_mean[j]:9.4f} {diff[j]:10.4f} "
          f"{np.degrees(diff[j]):10.2f} {DEFAULT_ANGLES[j]:9.4f}")
