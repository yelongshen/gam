#!/usr/bin/env python3
"""Chunk a PICO "sitting" session into clips that each hold ONE complete episode:

        stand -> walk toward the chair -> sit -> stand up -> walk away (leave) -> stand

Unlike `chunk_pico_session.py` (cuts on low motion activity, which would cut a sit in half: the seated
person is nearly still), this segments on the PELVIS HEIGHT:

  1. pelvis height z (world, floor-corrected as in tag_clips.py), median-smoothed 0.5 s
  2. standing reference = 90th percentile of z; SEATED when z < stand_ref - enter_drop,
     leaving the seated state only when z > stand_ref - exit_drop (hysteresis, so sit-down / stand-up
     transitions are one episode)
  3. seated runs shorter than --min-sit are discarded (a dip, not a sit)
  4. between two consecutive sits the gap holds [leave chair i] + [walk toward chair i+1]; the cut is
     put in the longest standing standstill inside the gap, or at the slowest point if there is none
  5. each clip is trimmed to --lead s before the first walking step and --tail s after the last one,
     and kept only if it passes the completeness checks below
  6. clips are written as consolidated npz (every recorded key sliced), like chunk_pico_session.py,
     plus clips.csv (with the QC columns) and a timeline PNG

Completeness checks (a clip is "complete" only if all hold):
  * exactly one seated episode, not touching either clip edge
  * clip starts and ends STANDING (z >= stand_ref - 0.15)
  * walking (root speed > walk_thr) for >= --min-walk s before the sit AND after the stand-up
  * the sit lasts >= --min-sit s

Usage:
  .venv_teleop/bin/python gear_sonic/scripts/chunk_pico_sit_session.py \
      /home/grease/g1_robot_data/pico_raw/20261002_151150_session.npz     # or the session dir
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_pico_motion import quat_apply_wxyz, yup_to_zup  # noqa: E402

PELVIS, L_FOOT, R_FOOT = 0, 10, 11
BUFFERED = {"smpl_pose", "smpl_joints", "body_quat_w", "joint_pos", "joint_vel", "frame_index", "body_pos_w"}


def load(path):
    if path.endswith(".npz"):
        d = dict(np.load(path, allow_pickle=True))
    else:
        from chunk_pico_session import load_session
        d, _ = load_session(path)
    return d


def movmed(x, n):
    n = n | 1
    pad = n // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(xp, n), axis=1)


def movavg(x, n):
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def signals(d, fps):
    loc = d["smpl_joints"].astype(np.float64)
    q = d["body_quat_w"].astype(np.float64)
    r = yup_to_zup(d["body_pos_w"].astype(np.float64))
    W = quat_apply_wxyz(q, loc) + r[:, None, :]
    W[..., 2] -= np.percentile(W[:, [L_FOOT, R_FOOT], 2].min(axis=1), 5)
    z = W[:, PELVIS, 2]
    xy = np.stack([movavg(W[:, PELVIS, 0], int(fps)), movavg(W[:, PELVIS, 1], int(fps))], 1)   # 1 s average
    v = np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps])
    return z, v


def runs(mask):
    """(start, end) of True runs, end exclusive."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def seated_episodes(z, fps, stand_ref, enter_drop, exit_drop, min_sit, merge_gap=2.0):
    enter = z < stand_ref - enter_drop
    stay = z < stand_ref - exit_drop
    seated = np.zeros(len(z), bool)
    on = False
    for i in range(len(z)):
        if not on and enter[i]:
            on = True
        elif on and not stay[i]:
            on = False
        seated[i] = on
    eps = [(a, b) for a, b in runs(seated)]
    # pelvis-z glitches while sitting (sensor jumps to a standing height for ~1 s) break one long sit into
    # several: a real stand-up lasts > merge_gap, so seated runs separated by less than that are one episode
    merged = []
    for a, b in eps:
        if merged and (a - merged[-1][1]) / fps < merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if (b - a) / fps >= min_sit]


def standstills(v, z, a, b, fps, still_thr, stand_z, min_still):
    """standing standstill intervals inside [a, b)."""
    ok = (v[a:b] < still_thr) & (z[a:b] > stand_z)
    return [(a + s, a + e) for s, e in runs(ok) if (e - s) / fps >= min_still]


def pick_cut(v, z, a, b, fps, still_thr, stand_z):
    """frame index inside the gap [a,b) at which to split two episodes."""
    st = standstills(v, z, a, b, fps, still_thr, stand_z, 0.5)
    if st:
        s, e = max(st, key=lambda x: x[1] - x[0])
        return (s + e) // 2
    lo, hi = a + int(0.25 * (b - a)), a + int(0.75 * (b - a))
    if hi <= lo:
        return (a + b) // 2
    return lo + int(np.argmin(v[lo:hi]))


def trim(v, a, b, fps, walk_thr, lead, tail):
    """shrink [a,b) to [first walking step - lead, last walking step + tail]."""
    w = np.where(v[a:b] > walk_thr)[0]
    if len(w) == 0:
        return a, b
    return max(a, a + w[0] - int(lead * fps)), min(b, a + w[-1] + int(tail * fps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="session dir or the consolidated *_session.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--enter-drop", type=float, default=0.25, help="seated when z < stand_ref - this [m]")
    ap.add_argument("--exit-drop", type=float, default=0.12, help="stand-up complete when z > stand_ref - this [m]")
    ap.add_argument("--min-sit", type=float, default=2.0, help="min seated duration [s]")
    ap.add_argument("--walk-thr", type=float, default=0.25, help="walking if root speed (1 s avg) > this [m/s]")
    ap.add_argument("--still-thr", type=float, default=0.12)
    ap.add_argument("--lead", type=float, default=1.0, help="seconds kept before the first step")
    ap.add_argument("--tail", type=float, default=1.0, help="seconds kept after the last step")
    ap.add_argument("--min-walk", type=float, default=1.0, help="min walking time before sit / after stand-up [s]")
    ap.add_argument("--short-gap", type=float, default=6.0,
                    help="two sits separated by a standing gap shorter than this [s] share the gap frames (overlap)")
    a = ap.parse_args()

    d = load(a.session)
    t = d["_t"].astype(np.float64) if "_t" in d else d["timestamp_monotonic"].reshape(-1).astype(np.float64)
    T = len(t)
    fps = T / (t[-1] - t[0])
    z_raw, v = signals(d, fps)
    z = movmed(z_raw, int(0.5 * fps))
    stand_ref = float(np.percentile(z, 90))
    stand_z = stand_ref - 0.15
    eps = seated_episodes(z, fps, stand_ref, a.enter_drop, a.exit_drop, a.min_sit)
    print(f"frames {T}  {t[-1]:.1f}s  fps {fps:.1f}   stand_ref z = {stand_ref:.3f} m   seated if z < {stand_ref - a.enter_drop:.3f}")
    print(f"seated episodes found: {len(eps)}  (durations s: {[round((b - c) / fps, 1) for c, b in eps]})")

    # boundaries between consecutive episodes
    cuts = [pick_cut(v, z, eps[i][1], eps[i + 1][0], fps, a.still_thr, stand_z) for i in range(len(eps) - 1)]
    # two sits separated by a very short standing gap cannot each get a full "leave" and a full "walk toward"
    # from one cut.  For gaps < --short-gap the SAME gap frames are given to both clips (they overlap there):
    # clip i runs until the next sit starts, clip i+1 starts as soon as the stand-up is complete.
    lo_hi = []
    for i, (s, e) in enumerate(eps):
        lo = cuts[i - 1] if i > 0 else max(0, s - int(12 * fps))
        hi = cuts[i] if i < len(eps) - 1 else min(T, e + int(12 * fps))
        if i > 0 and (s - eps[i - 1][1]) / fps < a.short_gap:
            lo = eps[i - 1][1]
        if i < len(eps) - 1 and (eps[i + 1][0] - e) / fps < a.short_gap:
            hi = eps[i + 1][0] - int(0.3 * fps)
        lo_hi.append((lo, hi))
    rows, clips = [], []
    for i, (s, e) in enumerate(eps):
        lo, hi = lo_hi[i]
        lo, hi = trim(v, lo, hi, fps, a.walk_thr, a.lead, a.tail)
        # completeness checks
        edge_ok = lo < s - 1 and hi > e + 1
        z_start, z_end = float(np.median(z[lo:lo + int(0.5 * fps)])), float(np.median(z[max(lo, hi - int(0.5 * fps)):hi]))
        stand_ok = z_start >= stand_z and z_end >= stand_z
        walk_in = float((v[lo:s] > a.walk_thr).sum() / fps)
        walk_out = float((v[e:hi] > a.walk_thr).sum() / fps)
        n_sit = sum(1 for (c, b) in eps if c >= lo and b <= hi)
        complete = bool(edge_ok and stand_ok and walk_in >= a.min_walk and walk_out >= a.min_walk and n_sit == 1)
        why = []
        if not edge_ok: why.append("sit touches clip edge")
        if not stand_ok: why.append("does not start/end standing")
        if walk_in < a.min_walk: why.append(f"walk-in {walk_in:.1f}s")
        if walk_out < a.min_walk: why.append(f"walk-out {walk_out:.1f}s")
        if n_sit != 1: why.append(f"{n_sit} sits")
        rows.append(dict(clip=f"clip_{i:03d}.npz", start_frame=int(lo), end_frame=int(hi), frames=int(hi - lo),
                         start_s=round(float(t[lo] - t[0]), 2), dur_s=round((hi - lo) / fps, 2),
                         sit_start_s=round((s - lo) / fps, 2), sit_dur_s=round((e - s) / fps, 2),
                         walk_in_s=round(walk_in, 1), walk_out_s=round(walk_out, 1),
                         seat_z=round(float(np.median(z[s:e])), 3), z_start=round(z_start, 3), z_end=round(z_end, 3),
                         complete=complete, issues="; ".join(why)))
        clips.append((lo, hi))

    out = a.out or a.session.replace("_session.npz", "").rstrip("/") + "_sit_clips"
    os.makedirs(out, exist_ok=True)
    save_keys = [k for k in d if not k.startswith("_")]
    for r, (lo, hi) in zip(rows, clips):
        clip = {k: d[k][lo:hi] for k in save_keys}
        clip["clip_start_index"] = np.int64(lo)
        clip["clip_duration_s"] = np.float32((hi - lo) / fps)
        clip["sit_start_index"] = np.int64(eps[int(r["clip"][5:8])][0] - lo)
        clip["sit_end_index"] = np.int64(eps[int(r["clip"][5:8])][1] - lo)
        np.savez_compressed(os.path.join(out, r["clip"]), **clip)
    with open(os.path.join(out, "clips.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\n{'clip':13s}{'start_s':>8s}{'dur_s':>7s}{'sit@':>6s}{'sit_s':>7s}{'walk_in':>8s}{'walk_out':>9s}{'seat_z':>8s}{'z0':>6s}{'z1':>6s}  complete")
    for r in rows:
        print(f"{r['clip']:13s}{r['start_s']:8.1f}{r['dur_s']:7.1f}{r['sit_start_s']:6.1f}{r['sit_dur_s']:7.1f}{r['walk_in_s']:8.1f}"
              f"{r['walk_out_s']:9.1f}{r['seat_z']:8.3f}{r['z_start']:6.2f}{r['z_end']:6.2f}  {r['complete']}  {r['issues']}")
    nc = sum(r["complete"] for r in rows)
    print(f"\n{len(rows)} clips, {nc} complete, {len(rows) - nc} flagged; wrote -> {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(26, 7), sharex=True)
        tt = (t - t[0])
        ax[0].plot(tt, z, lw=0.6); ax[0].axhline(stand_ref - a.enter_drop, c="r", lw=0.5)
        ax[1].plot(tt, v, lw=0.6); ax[1].axhline(a.walk_thr, c="r", lw=0.5)
        for r, (lo, hi) in zip(rows, clips):
            c = "g" if r["complete"] else "orange"
            for x in ax:
                x.axvspan(tt[lo], tt[hi - 1], color=c, alpha=0.18)
        ax[0].set_ylabel("pelvis z [m]"); ax[1].set_ylabel("root speed [m/s]"); ax[1].set_xlabel("session time [s]")
        fig.tight_layout(); fig.savefig(os.path.join(out, "timeline.png"), dpi=70)
        print("timeline ->", os.path.join(out, "timeline.png"))
    except Exception as ex:  # noqa: BLE001
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
