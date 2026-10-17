"""Sanity checks on ego_dataset/picoset_20261015: pairing/alignment, retarget start-up faults (early_contingency tests A and B), robot-vs-human
vertical agreement, and the tagged special clips (crawl / jump_A / jump_B).

  A  bowed IK: mean robot torso lean over the first 2 s minus the human lean (spine3-pelvis vs vertical) over the same 2 s  > 20 deg
  B  first-frame reference speed: max joint |d dof| / dt between frames 0 and 1 > 10 rad/s   (B' > 6 rad/s)
usage: python check_picoset_20261015.py
"""
import csv
import glob
import os

import joblib
import numpy as np

DS = "/home/grease/ego_dataset/picoset_20261015"
TAGS = "/home/grease/g1_robot_data/pico_raw/20261015_combined_clips/clip_tags.csv"


def unwrap(d):
    return d[next(iter(d))] if "dof" not in d else d


def lean_from_xyzw(q):
    x, y, z, w = q.T
    up_z = 1 - 2 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(up_z, -1, 1)))


tags = {r["clip"]: r for r in csv.DictReader(open(TAGS))}
rows = []
for rp in sorted(glob.glob(f"{DS}/robot/*.pkl")):
    name = os.path.basename(rp)[:-4]
    r = unwrap(joblib.load(rp))
    s = joblib.load(f"{DS}/smpl/{name}.pkl")
    s = s if "smpl_joints" in s else s[next(iter(s))]
    dof, rt, rq, fps = r["dof"], r["root_trans_offset"], r["root_rot"], float(r["fps"])
    J = np.asarray(s["smpl_joints"], float)
    v = J[:100, 9] - J[:100, 0]
    hl = float(np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))).mean())
    rl = float(lean_from_xyzw(rq[: int(2 * fps)]).mean())
    spd0 = float(np.abs(dof[1] - dof[0]).max() * fps)
    transl = np.asarray(s["transl"], float)
    key = name.replace("pico1015_", "")
    rows.append(dict(clip=key, T30=len(dof), T50=len(J), dur=len(dof) / fps, rl=rl, hl=hl, A=rl - hl, spd0=spd0,
                     rz_min=float(rt[:, 2].min()), rz_med=float(np.median(rt[:, 2])), rz_max=float(rt[:, 2].max()),
                     sz_range=float(np.ptp(transl[:, 2])), path=float(np.linalg.norm(np.diff(rt[:, :2], axis=0), axis=1).sum()),
                     event=tags.get(key, {}).get("event", ""), tg=tags.get(key, {}).get("tags", "")))
n = len(rows)
print(f"{n} clips, {sum(r['dur'] for r in rows) / 60:.1f} min   (smpl/robot pairs match: {sum(abs(r['T50'] - r['T30'] * 50 / 30) < 3 for r in rows)}/{n})")
fa = [r for r in rows if r["A"] > 20]
fb = [r for r in rows if r["spd0"] > 10]
fb2 = [r for r in rows if r["spd0"] > 6]
print(f"A  bowed IK (robot lean - human lean > 20 deg): {len(fa)}   | B first-frame speed > 10 rad/s: {len(fb)}   | B' > 6 rad/s: {len(fb2)}")
print(f"   lean diff (robot - human): median {np.median([r['A'] for r in rows]):+.1f} deg  p95 {np.percentile([r['A'] for r in rows], 95):+.1f}  max {max(r['A'] for r in rows):+.1f}")
print(f"   first-frame speed: median {np.median([r['spd0'] for r in rows]):.2f}  p95 {np.percentile([r['spd0'] for r in rows], 95):.2f}  max {max(r['spd0'] for r in rows):.2f} rad/s")
print(f"   robot root height (m): clip-median of medians {np.median([r['rz_med'] for r in rows]):.2f}; min over all {min(r['rz_min'] for r in rows):.2f}; max over all {max(r['rz_max'] for r in rows):.2f}")
for r in sorted(rows, key=lambda r: -r["A"])[:3]:
    print("   worst lean:", r["clip"], f"{r['A']:+.1f} deg (robot {r['rl']:.0f}, human {r['hl']:.0f})")
for r in sorted(rows, key=lambda r: -r["spd0"])[:3]:
    print("   worst first-frame speed:", r["clip"], f"{r['spd0']:.1f} rad/s")
print("\nspecial clips:")
for r in rows:
    if r["event"]:
        print(f"   {r['clip']:16s} event={r['event']:7s} {r['dur']:5.1f}s  robot root z {r['rz_min']:.2f}..{r['rz_med']:.2f}..{r['rz_max']:.2f} m   smpl transl z range {r['sz_range']:.2f} m   lean diff {r['A']:+.1f}  first-frame speed {r['spd0']:.1f}  tags={r['tg']}")

print("flagged (A>20 or spd0>10):", " ".join("pico1015_" + r["clip"] for r in rows if r["A"] > 20 or r["spd0"] > 10) or "none")
print("start/end standing check is done at cut time (cut_picoset_20261015_stand_clips.py); here: first-frame speed > 6 rad/s:", " ".join(f"{r['clip']}({r['spd0']:.1f})" for r in rows if r["spd0"] > 6) or "none")
