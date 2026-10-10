#!/usr/bin/env python
"""Locate a clip inside the ROBOT-side deploy logs (`g1_deploy_run_*/q.csv`).

`locate_clip_in_session.py` finds a clip in the `streamed_*` logs -- that is the
SMPL reference that went out on the wire, and proves only that the clip was
streamed. To say anything about how the robot *responded* you need the window
inside the policy run's own logs, which is what this does.

Method: take the retargeted robot reference (`eval_subset/robot/<clip>.pkl`,
`dof` at 30 fps), resample it to the 50 Hz control rate, and slide it over
`q.csv`. Unlike the SMPL-side match this is a *tracking* comparison, so the
residual is the robot's tracking error (degrees), not a reconstruction
mismatch (millimetres) -- expect ~2-10 deg on a good match versus ~20+ deg
for a wrong clip.

`motion_name.csv` cannot be used for this: with ZMQ input every frame is
labelled `"streamed"`, with no clip identity.

Usage:
  .venv_sim/bin/python sim2real/locate_clip_in_robot_logs.py \
      --clip jog_ff_start_180_R_002__A192_M \
      --runs /home/grease/g1_robot_data/g1_run_0909
"""
import argparse
import csv
import glob
import os

import joblib
import numpy as np

ROBOT_REF_DIR = "/home/grease/ego_dataset/eval_subset/robot"


def load_reference_dof(clip, target_fps=50.0):
    d = joblib.load(os.path.join(ROBOT_REF_DIR, clip + ".pkl"))
    vals = list(d.values())[0] if "dof" not in d else d
    dof = np.asarray(vals["dof"], dtype=np.float64)       # (N, 29) rad
    fps = float(vals.get("fps", 30.0))
    n_out = int(round(len(dof) * target_fps / fps))
    src = np.arange(len(dof)) / fps
    dst = np.arange(n_out) / target_fps
    out = np.empty((n_out, dof.shape[1]))
    for j in range(dof.shape[1]):
        out[:, j] = np.interp(dst, src, dof[:, j])
    return out, fps


def load_q(run_dir):
    q, t = [], []
    with open(os.path.join(run_dir, "q.csv")) as fh:
        r = csv.reader(fh)
        hdr = next(r)
        qi = [i for i, h in enumerate(hdr) if h.startswith("q_")]
        ti = hdr.index("time_realtime_ms")
        for row in r:
            if len(row) <= qi[-1]:
                continue
            q.append([float(row[i]) for i in qi])
            t.append(float(row[ti]))
    return np.asarray(q), np.asarray(t)


def best_window(ref, q, stride=1):
    T, N = len(ref), len(q)
    if N < T:
        return -1, np.inf
    best_off, best_err = -1, np.inf
    for off in range(0, N - T + 1, stride):
        err = np.abs(q[off:off + T] - ref).mean()
        if err < best_err:
            best_err, best_off = err, off
    return best_off, best_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--runs", required=True,
                    help="directory containing g1_deploy_run_* subdirectories")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    ref, src_fps = load_reference_dof(a.clip, a.fps)
    print(f"[ref] {a.clip}: {len(ref)} frames @ {a.fps:g} fps "
          f"(resampled from {src_fps:g} fps), {ref.shape[1]} dof\n")

    rows = []
    for run in sorted(glob.glob(os.path.join(a.runs, "*"))):
        if not os.path.isfile(os.path.join(run, "q.csv")):
            continue
        q, t = load_q(run)
        off, err = best_window(ref, q, a.stride)
        rows.append((os.path.basename(run), len(q), off, np.degrees(err), t))

    rows.sort(key=lambda r: r[3])
    print(f"{'run':<34}{'frames':>8}{'offset':>8}{'t_start':>9}{'t_end':>8}"
          f"{'mean_err[deg]':>15}")
    for name, n, off, errd, t in rows:
        t0 = off / a.fps
        t1 = t0 + len(ref) / a.fps
        print(f"{name:<34}{n:>8}{off:>8}{t0:>9.2f}{t1:>8.2f}{errd:>15.2f}")

    if rows:
        name, n, off, errd, t = rows[0]
        print(f"\nBEST: {name}  frames [{off}, {off + len(ref)})  "
              f"t = {off / a.fps:.2f}-{(off + len(ref)) / a.fps:.2f} s  "
              f"({errd:.2f} deg mean tracking error)")


if __name__ == "__main__":
    main()
