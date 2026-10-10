"""Chunk a recorded PICO session into clips.

This session has no manual markers (`toggle_data_collection` is all-False and the
triggers never cross 0.5), so boundaries are derived from MOTION ACTIVITY:

  1. activity = smoothed (root speed + mean joint speed)
  2. cut at "rest gaps" -- runs below `--rest-pct` percentile lasting >= `--min-rest`
  3. any clip still longer than `--max-len` is recursively split at its
     lowest-activity interior point
  4. clips shorter than `--min-len` are dropped

Each clip is written as ONE consolidated npz with every recorded key stacked
over time (the newest frame of each source npz buffer), so downstream
retargeting can load a clip in a single call.

Usage:
    python gear_sonic/scripts/chunk_pico_session.py \
        /home/grease/g1_robot_data/pico_raw/20260928_162215
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

# Keys that are per-frame buffers (T_buf, ...) -> take the newest entry.
BUFFERED_KEYS = {"smpl_pose", "smpl_joints", "body_quat_w", "joint_pos",
                 "joint_vel", "frame_index", "body_pos_w"}


def load_session(session_dir: str):
    files = sorted(glob.glob(os.path.join(session_dir, "pose_*.npz")))
    if not files:
        raise FileNotFoundError(f"no pose_*.npz in {session_dir}")

    keys = list(np.load(files[0], allow_pickle=True).files)
    data: dict[str, list] = {k: [] for k in keys}
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in keys:
            a = d[k]
            data[k].append(a[-1] if (k in BUFFERED_KEYS and a.ndim >= 1 and a.shape[0] > 1) else
                           (a[-1] if k in BUFFERED_KEYS and a.ndim > 1 else a))
    out = {k: np.asarray(v) for k, v in data.items()}
    t = out["timestamp_monotonic"].reshape(-1).astype(np.float64)
    out["_t"] = t - t[0]
    return out, files


def activity_signal(d, smooth_win: int = 25):
    """Smoothed per-frame motion energy: root speed + mean joint speed."""
    t = d["_t"]
    dt = np.maximum(np.diff(t), 1e-3)
    J = d["smpl_joints"].astype(np.float64)
    P = d["body_pos_w"].astype(np.float64) if "body_pos_w" in d else np.zeros_like(J[:, 0])

    v_root = np.linalg.norm(np.diff(P, axis=0), axis=1) / dt
    v_joint = np.linalg.norm(np.diff(J, axis=0), axis=2).mean(axis=1) / dt
    v = v_root + v_joint
    v = np.concatenate([v[:1], v])  # pad back to T

    k = np.ones(smooth_win) / smooth_win
    return np.convolve(v, k, mode="same")


def find_rest_cuts(act, t, rest_pct: float, min_rest: float):
    """Cut indices at the centre of each sustained low-activity run."""
    thr = np.percentile(act, rest_pct)
    idle = act < thr
    cuts, start = [], None
    for i, v in enumerate(idle):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if t[i - 1] - t[start] >= min_rest:
                cuts.append((start + i - 1) // 2)
            start = None
    if start is not None and t[-1] - t[start] >= min_rest:
        cuts.append((start + len(idle) - 1) // 2)
    return cuts, float(thr)


def split_long(seg, t, act, max_len: float, min_len: float):
    """Recursively split a (lo, hi) segment at its lowest-activity interior point."""
    lo, hi = seg
    if t[hi - 1] - t[lo] <= max_len:
        return [seg]
    # only consider interior points that leave both halves >= min_len
    guard = np.searchsorted(t[lo:hi] - t[lo], min_len)
    a, b = lo + guard, hi - guard
    if b <= a:
        return [seg]
    cut = a + int(np.argmin(act[a:b]))
    return (split_long((lo, cut), t, act, max_len, min_len)
            + split_long((cut, hi), t, act, max_len, min_len))


def chunk(session_dir, out_dir, rest_pct, min_rest, min_len, max_len):
    d, files = load_session(session_dir)
    t = d["_t"]
    T = len(t)
    act = activity_signal(d)

    cuts, thr = find_rest_cuts(act, t, rest_pct, min_rest)
    bounds = [0] + cuts + [T]
    segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    segs = [s for s in segs if s[1] > s[0]]

    final = []
    for s in segs:
        final.extend(split_long(s, t, act, max_len, min_len))
    final = [s for s in final if t[s[1] - 1] - t[s[0]] >= min_len]

    print(f"session      : {session_dir}")
    print(f"frames       : {T}  ({t[-1]:.1f}s)")
    print(f"rest thresh  : {thr:.3f} (p{rest_pct})  -> {len(cuts)} rest cut(s)")
    print(f"clips        : {len(final)}\n")

    os.makedirs(out_dir, exist_ok=True)
    save_keys = [k for k in d if not k.startswith("_")]
    rows = []
    for i, (lo, hi) in enumerate(final):
        name = f"clip_{i:03d}.npz"
        dur = t[hi - 1] - t[lo]
        clip = {k: d[k][lo:hi] for k in save_keys}
        clip["clip_start_index"] = np.int64(lo)
        clip["clip_duration_s"] = np.float32(dur)
        np.savez_compressed(os.path.join(out_dir, name), **clip)

        P = d["body_pos_w"][lo:hi].astype(np.float64)
        path = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        disp = float(np.linalg.norm(P[-1] - P[0]))
        rows.append(dict(clip=name, start_frame=lo, end_frame=hi, frames=hi - lo,
                         start_s=round(float(t[lo]), 2), dur_s=round(float(dur), 2),
                         path_m=round(path, 2), disp_m=round(disp, 2),
                         mean_act=round(float(act[lo:hi].mean()), 3)))
        print(f"  {name}  frames {lo:6d}-{hi:6d}  {dur:6.2f}s  "
              f"path {path:6.2f}m  disp {disp:5.2f}m  act {act[lo:hi].mean():.2f}")

    man = os.path.join(out_dir, "clips.csv")
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(final)} clips + manifest -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--out", default=None, help="default: <session>_clips")
    ap.add_argument("--rest-pct", type=float, default=15.0,
                    help="activity percentile treated as 'rest'")
    ap.add_argument("--min-rest", type=float, default=0.8,
                    help="min rest duration (s) to count as a boundary")
    ap.add_argument("--min-len", type=float, default=3.0, help="min clip length (s)")
    ap.add_argument("--max-len", type=float, default=20.0, help="max clip length (s)")
    a = ap.parse_args()

    out = a.out or (a.session_dir.rstrip("/") + "_clips")
    chunk(a.session_dir, out, a.rest_pct, a.min_rest, a.min_len, a.max_len)


if __name__ == "__main__":
    main()
