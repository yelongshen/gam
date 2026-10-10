"""Full QC sweep over all chunked clips.

Checks, per clip:

  boneSD       max per-bone length standard deviation. This is the ONLY honest
               "did body tracking fail" test: bone lengths are POSTURE-INVARIANT,
               so they stay constant whether the subject stands, squats, crawls
               or jumps. A collapsing skeleton makes them jitter.
  headPelv     |head - pelvis| 3D distance (also posture-invariant). Replaces the
               old vertical `stature` metric, which assumed an UPRIGHT subject and
               therefore mislabelled crawling as "stature collapse".
  statureZ     head_z - lowest_foot_z, kept for reference ONLY. Legitimately small
               when crawling / lying down -- never judge a clip by this alone.
  footDrift    p95-p5 spread of the raw lowest-foot height.
  airborne%    frames whose lowest foot sits >15 cm above the estimated floor.
               Legitimately HIGH for jumps and for stepping on/off a platform.
  maxJump      largest single-frame joint displacement.
  vxyMax       peak horizontal root speed -- separates a real leap (fast) from a
               tracking teleport (jump with no plausible approach speed).
  held%        frames byte-identical to the previous one (zero-order-hold).

Only `boneSD` / `headPelv` are treated as hard failures. Height/contact metrics
are reported but NOT used to reject, because crawling, jumping and platform work
all violate the flat-floor-upright assumptions they encode.

Usage:
    python gear_sonic/scripts/qc_all_clips.py <clips_dir>
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

from visualize_pico_motion import SMPL_PARENTS, quat_apply_wxyz, yup_to_zup

L_FOOT, R_FOOT, HEAD = 10, 11, 15


def qc_clip(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    loc = d["smpl_joints"].astype(np.float64)
    q = d["body_quat_w"].astype(np.float64)
    r = yup_to_zup(d["body_pos_w"].astype(np.float64))
    t = d["timestamp_monotonic"].reshape(-1).astype(np.float64)
    t = t - t[0]

    W = quat_apply_wxyz(q, loc) + r[:, None, :]  # world, no floor shift yet
    feet = W[:, [L_FOOT, R_FOOT], 2]
    low = feet.min(axis=1)

    # robust floor from the 5th percentile of the lowest foot
    floor = float(np.percentile(low, 5))
    fz = low - floor

    # --- posture-INVARIANT skeleton integrity ---
    bones = np.stack(
        [np.linalg.norm(W[:, i] - W[:, p], axis=1)
         for i, p in enumerate(SMPL_PARENTS) if p >= 0],
        axis=1,
    )
    bone_sd = float(bones.std(axis=0).max())
    head_pelv = np.linalg.norm(W[:, HEAD] - W[:, 0], axis=1)

    stature_z = W[:, HEAD, 2] - low
    foot_drift = float(np.percentile(low, 95) - np.percentile(low, 5))

    dt = np.maximum(np.diff(t), 1e-3)
    vxy = np.linalg.norm(np.diff(W[:, 0, :2], axis=0), axis=1) / dt

    jump = np.linalg.norm(np.diff(W, axis=0), axis=2).max(axis=1)
    flat = loc.reshape(len(loc), -1)
    held = float((np.abs(np.diff(flat, axis=0)).max(axis=1) == 0).mean())

    return dict(
        clip=os.path.basename(path)[:-4],
        n=len(W),
        dur=round(float(t[-1]), 1),
        boneSD=round(bone_sd, 5),
        headPelv=round(float(head_pelv.mean()), 3),
        headPelvSD=round(float(head_pelv.std()), 4),
        statureZ=round(float(stature_z.mean()), 2),
        footDrift=round(foot_drift, 3),
        airborne=round(100.0 * float((fz > 0.15).mean()), 1),
        vxyMax=round(float(vxy.max()), 2),
        maxJump=round(float(jump.max()), 3),
        held=round(100.0 * held, 1),
    )


def verdict(m: dict) -> str:
    """Only posture-invariant skeleton damage is a hard failure."""
    bad, note = [], []
    if m["boneSD"] > 0.010 or m["headPelvSD"] > 0.10:
        bad.append("SKELETON-BROKEN")
    if m["maxJump"] > 0.60 and m["vxyMax"] < 1.5:
        # big positional jump with no plausible approach speed -> tracking glitch
        bad.append("TELEPORT")

    # descriptive notes only -- these do NOT reject a clip
    if m["statureZ"] < 1.25:
        note.append("low-posture(crawl/ground)")
    if m["airborne"] > 30.0:
        note.append("airborne(jump/platform)")
    if m["maxJump"] > 0.60:
        note.append("fast-motion")
    if m["held"] > 30.0:
        note.append("many-held-frames")

    if bad:
        return "BAD   " + ",".join(bad)
    return "ok" + ("    " + ",".join(note) if note else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.clips_dir, "clip_*.npz")))
    rows = []
    hdr = f"{'clip':18s}{'dur':>6s}{'boneSD':>9s}{'headPelv':>10s}{'hpSD':>8s}" \
          f"{'statZ':>7s}{'footDrift':>10s}{'air%':>7s}{'vxyMax':>8s}{'jump':>7s}{'held%':>7s}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for p in paths:
        m = qc_clip(p)
        m["verdict"] = verdict(m)
        rows.append(m)
        print(f"{m['clip']:18s}{m['dur']:6.1f}{m['boneSD']:9.5f}{m['headPelv']:10.3f}"
              f"{m['headPelvSD']:8.4f}{m['statureZ']:7.2f}{m['footDrift']:10.3f}"
              f"{m['airborne']:7.1f}{m['vxyMax']:8.2f}{m['maxJump']:7.3f}{m['held']:7.1f}"
              f"  {m['verdict']}")

    out = a.out or os.path.join(a.clips_dir, "qc_report.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bad = [r["clip"] for r in rows if r["verdict"].startswith("BAD")]
    ok = [r["clip"] for r in rows if not r["verdict"].startswith("BAD")]
    print(f"\nok: {len(ok)}   BAD: {len(bad)}")
    if bad:
        print("BAD  :", " ".join(bad))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
