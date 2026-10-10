#!/usr/bin/env python3
"""Print raw frame-0 q for every run, to compare cold-start pose consistency
across sub-runs / robots."""
import sys, os
sys.path.insert(0, "/home/grease/gam")
import numpy as np
from data_process.g1_params import JOINT_NAMES, DEFAULT_ANGLES
from model_eval.sim2real_single_run_quicklook import load

run_dirs = sys.argv[1:]
rows = []
for rd in run_dirs:
    q, t = load(rd, "q")
    if len(q) == 0:
        print(f"{os.path.basename(rd):40s}  EMPTY")
        continue
    rows.append((os.path.basename(rd), q[0]))

watch = ["L_elbow","R_elbow","L_ankle_pitch","R_ankle_pitch","L_sho_yaw","R_sho_yaw",
         "L_hip_pitch","R_hip_pitch","waist_roll","waist_pitch"]
jidx = [JOINT_NAMES.index(n) for n in watch]
print(f"{'run':40s} " + " ".join(f"{n:>13s}" for n in watch))
for name, q0 in rows:
    print(f"{name:40s} " + " ".join(f"{q0[j]:13.4f}" for j in jidx))
print(f"{'DEFAULT_ANGLES':40s} " + " ".join(f"{DEFAULT_ANGLES[j]:13.4f}" for j in jidx))

if len(rows) > 1:
    stacked = np.stack([q0 for _, q0 in rows])
    std = stacked.std(axis=0)
    print(f"\nstd across these {len(rows)} runs, top-5 most variable joints:")
    for j in np.argsort(-std)[:5]:
        print(f"  {JOINT_NAMES[j]:15s} std={std[j]:.4f} rad ({np.degrees(std[j]):.2f} deg) "
              f"range=[{stacked[:,j].min():.3f},{stacked[:,j].max():.3f}]")
