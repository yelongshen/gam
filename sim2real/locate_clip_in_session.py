#!/usr/bin/env python
"""Locate a streamed SMPL clip inside the real-robot `streamed_*` session logs.

The `20260905/streamed_*` sessions log `smpl_joint.csv` (24x3 root-de-rotated SMPL
joints, exactly what `data_process/stream_clip_mode2.py` puts on the wire). This
script reproduces the streamed joint trajectory for a clip and finds where (and
whether) it appears inside each session, so the session -> clip mapping needed by
`sim2real/online_deployment_eval_plan.md` can be established from data alone.

Usage:
  .venv_teleop/bin/python sim2real/locate_clip_in_session.py \
      --clip ../ego_dataset/eval_subset/smpl/walk_sideway_045_stop_005__A042_M.pkl \
      --sessions /home/grease/g1_robot_data/20260905
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.stream_clip_mode2 import load_stream_data  # noqa: E402


def load_session_joints(session_dir):
    """Read smpl_joint.csv, tolerating a truncated final row (logger killed mid-write)."""
    path = os.path.join(session_dir, "smpl_joint.csv")
    rows = []
    with open(path) as fh:
        next(fh)  # header
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 72:
                continue  # partially flushed last row
            rows.append([float(x) for x in parts])
    arr = np.asarray(rows, dtype=np.float64)
    return arr.reshape(len(arr), 24, 3)


def best_alignment(clip, sess, stride=1):
    """Slide `clip` (T,24,3) over `sess` (N,24,3); return (offset, mean per-joint err[m])."""
    T, N = len(clip), len(sess)
    if N < T:
        return None, np.inf
    best_off, best_err = -1, np.inf
    for off in range(0, N - T + 1, stride):
        err = np.linalg.norm(sess[off : off + T] - clip, axis=-1).mean()
        if err < best_err:
            best_err, best_off = err, off
    return best_off, best_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--sessions", default="/home/grease/g1_robot_data/20260905")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    joints, _, _, _, _ = load_stream_data(args.clip, args.fps, official=True)
    clip = np.asarray(joints, dtype=np.float64)
    print(f"[clip] {os.path.basename(args.clip)}: {len(clip)} frames "
          f"({len(clip) / args.fps:.2f} s @ {args.fps:g} fps)\n")

    rows = []
    for sdir in sorted(glob.glob(os.path.join(args.sessions, "*"))):
        if not os.path.isdir(sdir):
            continue
        try:
            sess = load_session_joints(sdir)
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"  {os.path.basename(sdir)}: unreadable ({exc})")
            continue
        off, err = best_alignment(clip, sess, args.stride)
        rows.append((os.path.basename(sdir), len(sess), off, err))

    rows.sort(key=lambda r: r[3])
    print(f"{'session':<20}{'frames':>8}{'offset':>9}{'t_start':>10}"
          f"{'t_end':>9}{'mean_err[mm]':>14}")
    for name, n, off, err in rows:
        t0 = off / args.fps if off is not None and off >= 0 else float("nan")
        t1 = t0 + len(clip) / args.fps
        print(f"{name:<20}{n:>8}{off if off is not None else -1:>9}"
              f"{t0:>10.2f}{t1:>9.2f}{err * 1000:>14.2f}")

    if rows:
        name, n, off, err = rows[0]
        print(f"\nBEST MATCH: {name}  frames [{off}, {off + len(clip)})  "
              f"t = {off / args.fps:.2f}-{(off + len(clip)) / args.fps:.2f} s  "
              f"(mean per-joint error {err * 1000:.2f} mm)")


if __name__ == "__main__":
    main()
