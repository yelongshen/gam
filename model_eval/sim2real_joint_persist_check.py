#!/usr/bin/env python3
"""Check whether a frame-0 pose offset persists into steady standing
(encoder_mode==0, i.e. before any active tracking engages), for one joint."""
import sys
sys.path.insert(0, "/home/grease/gam")
import numpy as np
from data_process.g1_params import JOINT_NAMES
from model_eval.sim2real_single_run_quicklook import load

joint = sys.argv[1]
dirs = sys.argv[2:]
j = JOINT_NAMES.index(joint)

print(f"joint={joint}")
print(f"{'run':45s} {'frame0':>9s} {'mode0_mean':>11s} {'mode0_std':>10s} {'n_mode0':>8s}")
for d in dirs:
    q, t = load(d, "q")
    mode, _ = load(d, "encoder_mode", ncols=1)
    n = min(len(q), len(mode))
    q, mode = q[:n], mode[:n, 0]
    if n == 0:
        print(f"{d:45s}  EMPTY")
        continue
    m0 = mode == 0
    if m0.sum() == 0:
        print(f"{d:45s} {q[0,j]:9.4f}  (no mode-0 samples)")
        continue
    print(f"{d:45s} {q[0,j]:9.4f} {q[m0,j].mean():11.4f} {q[m0,j].std():10.4f} {int(m0.sum()):8d}")
