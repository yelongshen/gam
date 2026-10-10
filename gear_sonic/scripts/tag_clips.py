#!/usr/bin/env python
"""Characterise and TAG every clip in a chunked PICO session.

QC (`qc_all_clips.py`) answers "is this clip broken?". This answers "what
motion IS it?", so a 163-clip session can be browsed, filtered and balanced
without watching 163 videos.

Tags are derived from measured kinematics, never from file names. Each is a
thresholded statement about one quantity, listed in `TAG_RULES` below so the
definition is auditable:

  locomotion   travel      net horizontal displacement of the pelvis
               in-place    path length large but displacement small
               static      pelvis barely moves at all
  posture      crawl       pelvis height stays very low
               crouch      pelvis dips well below its standing value
               upright     default
  dynamics     jump        both feet leave the floor together
               fast        high mean joint speed
               slow        low mean joint speed
  emphasis     arms        arm ROM dominates leg ROM
               legs        leg ROM dominates
  turning      turn        large accumulated yaw of the pelvis

THRESHOLDS ARE HEURISTIC. They were set by looking at the distribution of this
session and are printed with the summary so they can be judged. A clip sitting
just either side of a boundary is not meaningfully different from its neighbour
-- use the numeric columns, not the tag, when the distinction matters.

Usage:
  python gear_sonic/scripts/tag_clips.py <clips_dir>
  python gear_sonic/scripts/tag_clips.py <clips_dir> --sort dur
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

from visualize_pico_motion import SMPL_PARENTS, quat_apply_wxyz, yup_to_zup

L_FOOT, R_FOOT, HEAD, PELVIS = 10, 11, 15, 0
L_ARM = [16, 17, 18, 20, 21, 22]
R_ARM = [17, 19, 21, 23]
ARM_JOINTS = [16, 17, 18, 19, 20, 21, 22, 23]
LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]

TAG_RULES = [
    ("travel",   "net pelvis displacement > 1.5 m"),
    ("in-place", "path > 1.5 m but displacement < 0.8 m"),
    ("static",   "path < 0.8 m"),
    ("crawl",    "median pelvis height < 55% of its own max"),
    ("crouch",   "min pelvis height < 70% of max (but not crawl)"),
    ("jump",     "both feet > 12 cm above the floor simultaneously"),
    ("fast",     "mean joint speed > 1.6 m/s"),
    ("slow",     "mean joint speed < 0.6 m/s"),
    ("arms",     "arm ROM > 1.6x leg ROM"),
    ("legs",     "leg ROM > 1.6x arm ROM"),
    ("turn",     "net heading change > 60 deg, or yaw span > 100 deg"),
]


def world_joints(path):
    d = np.load(path, allow_pickle=True)
    loc = d["smpl_joints"].astype(np.float64)
    q = d["body_quat_w"].astype(np.float64)
    r = yup_to_zup(d["body_pos_w"].astype(np.float64))
    t = d["timestamp_monotonic"].reshape(-1).astype(np.float64)
    W = quat_apply_wxyz(q, loc) + r[:, None, :]
    W[..., 2] -= np.percentile(W[:, [L_FOOT, R_FOOT], 2].min(axis=1), 5)
    return W, q, t - t[0]


def yaw_of(q):
    """Root yaw statistics, in degrees.

    Returns (net, span). `net` is |yaw(end) - yaw(start)| on the unwrapped
    signal -- how far the subject ended up turned. `span` is max-min, which
    also catches a turn that came back.

    Deliberately NOT the accumulated sum of |dyaw|: that integrates sensor
    jitter, so it grows with clip length rather than with turning. Measured on
    this session it correlated 0.64 with duration and tagged 100/163 clips as
    turns, which is why it was replaced.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    deg = 180.0 / np.pi
    return float(abs(yaw[-1] - yaw[0]) * deg), float((yaw.max() - yaw.min()) * deg)


def features(path):
    W, q, t = world_joints(path)
    dur = float(t[-1])
    dt = np.maximum(np.diff(t), 1e-3)

    pel = W[:, PELVIS, :]
    path_len = float(np.linalg.norm(np.diff(pel[:, :2], axis=0), axis=1).sum())
    disp = float(np.linalg.norm(pel[-1, :2] - pel[0, :2]))
    pel_z = pel[:, 2]

    feet_z = W[:, [L_FOOT, R_FOOT], 2]
    airborne = float((feet_z.min(axis=1) > 0.12).mean())

    jspeed = np.linalg.norm(np.diff(W, axis=0), axis=2).mean(axis=1) / dt
    arm_rom = float(np.mean([W[:, j, :].max(0) - W[:, j, :].min(0)
                             for j in ARM_JOINTS]))
    leg_rom = float(np.mean([W[:, j, :].max(0) - W[:, j, :].min(0)
                             for j in LEG_JOINTS]))
    yaw_net, yaw_span = yaw_of(q)

    return dict(
        clip=os.path.basename(path)[:-4],
        n=len(W), dur=round(dur, 1),
        path=round(path_len, 2), disp=round(disp, 2),
        pelz_med=round(float(np.median(pel_z)), 3),
        pelz_min=round(float(pel_z.min()), 3),
        pelz_max=round(float(pel_z.max()), 3),
        air=round(100 * airborne, 1),
        jspd=round(float(jspeed.mean()), 2),
        arm_rom=round(arm_rom, 3), leg_rom=round(leg_rom, 3),
        yaw=round(yaw_net, 0), yaw_span=round(yaw_span, 0),
    )


def tag(f):
    tags = []
    # locomotion
    if f["disp"] > 1.5:
        tags.append("travel")
    elif f["path"] < 0.8:
        tags.append("static")
    elif f["path"] > 1.5:
        tags.append("in-place")
    # posture -- relative to the clip's own standing height, so subject
    # height does not matter
    ratio_med = f["pelz_med"] / max(f["pelz_max"], 1e-6)
    ratio_min = f["pelz_min"] / max(f["pelz_max"], 1e-6)
    if ratio_med < 0.55:
        tags.append("crawl")
    elif ratio_min < 0.70:
        tags.append("crouch")
    # dynamics
    if f["air"] > 5.0:
        tags.append("jump")
    if f["jspd"] > 1.6:
        tags.append("fast")
    elif f["jspd"] < 0.6:
        tags.append("slow")
    # emphasis
    if f["arm_rom"] > 1.6 * f["leg_rom"]:
        tags.append("arms")
    elif f["leg_rom"] > 1.6 * f["arm_rom"]:
        tags.append("legs")
    # turning: net heading change, or a there-and-back turn
    if f["yaw"] > 60 or f["yaw_span"] > 100:
        tags.append("turn")
    return tags or ["plain"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dir")
    ap.add_argument("--sort", default="clip",
                    choices=["clip", "dur", "path", "jspd", "pelz_med"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(a.clips_dir, "clip_*.npz"))):
        try:
            f = features(p)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {os.path.basename(p)}: {exc}")
            continue
        f["tags"] = ",".join(tag(f))
        rows.append(f)

    rows.sort(key=lambda r: r[a.sort] if a.sort != "clip" else r["clip"],
              reverse=a.sort != "clip")

    hdr = (f"{'clip':<11}{'dur':>6}{'path':>7}{'disp':>7}{'pelZ':>7}"
           f"{'pelZmin':>8}{'air%':>6}{'jspd':>6}{'armROM':>8}{'legROM':>8}"
           f"{'yaw':>6}{'yawSpan':>8}  tags")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['clip']:<11}{r['dur']:6.1f}{r['path']:7.2f}{r['disp']:7.2f}"
              f"{r['pelz_med']:7.3f}{r['pelz_min']:8.3f}{r['air']:6.1f}"
              f"{r['jspd']:6.2f}{r['arm_rom']:8.3f}{r['leg_rom']:8.3f}"
              f"{r['yaw']:6.0f}{r['yaw_span']:8.0f}  {r['tags']}")

    print(f"\n{len(rows)} clips, {sum(r['dur'] for r in rows)/60:.1f} min total")

    print("\n=== tag counts ===")
    cnt = {}
    for r in rows:
        for t in r["tags"].split(","):
            cnt[t] = cnt.get(t, 0) + 1
    for t, c in sorted(cnt.items(), key=lambda x: -x[1]):
        dur = sum(r["dur"] for r in rows if t in r["tags"].split(","))
        print(f"  {t:12s} {c:4d} clips  {dur/60:6.1f} min")

    print("\n=== tag rules (heuristic, tuned on this session) ===")
    for name, rule in TAG_RULES:
        if name in cnt:
            print(f"  {name:12s} {rule}")

    out = a.out or os.path.join(a.clips_dir, "clip_tags.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
