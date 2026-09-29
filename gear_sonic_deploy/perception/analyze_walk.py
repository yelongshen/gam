#!/usr/bin/env python3
"""Summarise a walking-test recording (see WALKING_TEST.md): the terrain topic log and tegrastats.

    python3 gear_sonic_deploy/perception/analyze_walk.py ~/g1_perception_logs/walk_<T>
      (reads walk_<T>_terrain.npz and walk_<T>_tegrastats.log; numpy only, runs on the robot)

DLIO accuracy comes from eval_motion.py on the bag (walk_<T>/), see WALKING_TEST.md.
"""
import re
import sys

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # else numpy's OpenBLAS spins a busy thread per core
import numpy as np  # noqa: E402

prefix = sys.argv[1].rstrip("/")

# ---- terrain topic ----
try:
    d = np.load(prefix + "_terrain.npz")
except FileNotFoundError:
    d = None
    print(f"(no {prefix}_terrain.npz)")
if d is not None:
    t = d["recv_time"] - d["recv_time"][0]
    dt = np.diff(d["recv_time"])
    stale = (d["timestamp"] - d["map_stamp"])[:, 0]
    pos = d["torso_pos"]
    # path length on a 1 Hz subsample: summing DLIO's mm-level jitter at 50 Hz inflates it
    sec = np.searchsorted(t, np.arange(0, t[-1] + 1e-9, 1.0))
    step = np.zeros(len(t))
    step[sec[1:]] = np.linalg.norm(np.diff(pos[sec, :2], axis=0), axis=1)
    win_seen = d["terrain_height_valid"].reshape(len(t), -1).mean(1)
    grid_seen = d["height_grid_valid"].reshape(len(t), -1).mean(1)
    print(f"terrain topic: {len(t)} msgs over {t[-1]:.0f} s -> {len(t)/max(t[-1],1e-9):.1f} Hz; "
          f"gaps > 100 ms: {(dt > 0.1).sum()} (longest {dt.max()*1000:.0f} ms)")
    print(f"  map staleness: median {np.median(stale)*1000:.0f} ms, p99 {np.percentile(stale, 99)*1000:.0f} ms, "
          f"max {stale.max()*1000:.0f} ms")
    print(f"  torso path (DLIO, robot/odom): {step.sum():.2f} m walked; start {np.round(pos[0], 3)}, "
          f"end {np.round(pos[-1], 3)} -> {np.linalg.norm(pos[-1, :2] - pos[0, :2])*100:.1f} cm apart")
    speed = np.interp(t, t[sec[1:]], step[sec[1:]]) if len(sec) > 1 else np.zeros(len(t))  # m per s
    moving = speed > 0.05
    print(f"  VideoMimic window seen: {win_seen.mean()*100:.0f}% overall, "
          f"{win_seen[moving].mean()*100 if moving.any() else float('nan'):.0f}% while moving, "
          f"max {win_seen.max()*100:.0f}%;  2x2 m grid seen {grid_seen.mean()*100:.0f}%")
    # per 10 s: seen fraction + window median (torso_z - terrain_z) over seen cells
    print("  per 10 s:  t      moved   window seen   window median (seen)")
    w, wv = d["terrain_height"], d["terrain_height_valid"]
    for a in np.arange(0, t[-1], 10.0):
        k = (t >= a) & (t < a + 10)
        if not k.any():
            continue
        vals = w[k][wv[k]]
        moved = step[k].sum()
        print(f"           {a:4.0f}s  {moved:5.2f} m   {win_seen[k].mean()*100:5.0f}%       "
              f"{np.median(vals) if vals.size else float('nan'):.3f}")

# ---- tegrastats ----
try:
    lines = open(prefix + "_tegrastats.log").read().splitlines()
except FileNotFoundError:
    lines = []
    print(f"(no {prefix}_tegrastats.log)")
if lines:
    ram = [int(m.group(1)) for m in (re.search(r"RAM (\d+)/", l) for l in lines) if m]
    gpu = [int(m.group(1)) for m in (re.search(r"GR3D_FREQ (\d+)%", l) for l in lines) if m]
    cpu = [[int(x) for x in re.findall(r"(\d+)%@", m.group(1))] for m in (re.search(r"CPU \[([^\]]*)\]", l) for l in lines) if m]
    tj = [float(m.group(1)) for m in (re.search(r"tj@([\d.]+)C", l) for l in lines) if m]
    cpu = np.array(cpu)
    print(f"tegrastats: {len(lines)} samples")
    if cpu.size:
        print("  CPU busy % per core, median / p90 (c0 = gear_sonic control thread):  "
              + "  ".join(f"c{i} {np.median(cpu[:, i]):.0f}/{np.percentile(cpu[:, i], 90):.0f}" for i in range(cpu.shape[1])))
    if ram:
        print(f"  RAM used: max {max(ram)} MB;  GPU busy: median {np.median(gpu):.0f}%, p90 {np.percentile(gpu, 90):.0f}%;  "
              f"junction temp max {max(tj):.1f} C" if gpu and tj else f"  RAM used: max {max(ram)} MB")
