#!/usr/bin/env python3
"""Make the sit dataset's robot trajectories + chairs mutually consistent.

Run with the env_isaaclab python (needs torch + easydict for the G1 forward kinematics):

  /home/grease/miniforge3/envs/env_isaaclab/bin/python data_process/snap_robot_floor_check_chairs.py snap   <dataset>
  /home/grease/miniforge3/envs/env_isaaclab/bin/python data_process/snap_robot_floor_check_chairs.py check  <dataset>

snap  : per-clip robot floor correction.  The retargeted robot hovers while STANDING (lowest foot link a
        median ~0.08 m above the floor, vs ~0.02 m in the walking sets) because G1's legs are shorter
        than the scaled human's.  The root z is lowered by a smoothed foot-contact envelope, applied only
        where the robot is standing (pelvis high), so the seated frames -- and therefore the chair --
        are untouched.  Originals are kept in <dataset>/robot_orig_backup/.
check : (1) standing foot clearance, (2) chair contact gap at the seated frames, (3) collisions of the
        robot's links with the seat slab outside the seated segment (walking THROUGH the chair) and of
        non-contact links inside it, (4) frame-0 spawn: robot standing above the floor / chair overlap.
"""
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

sys.path.insert(0, "/home/grease/GR00T-WholeBodyControl/dev_notes/sit_subset_build")
import enforce_sit_contact as E  # noqa: E402  (provides Humanoid_Batch FK + conventions)

FOOT_KEYS = ("ankle_roll", "toe")


def foot_idx(names):
    return [i for i, n in enumerate(names) if any(k in n for k in FOOT_KEYS)]


STAND_FOOT_TARGET = 0.045  # lowest ankle-link height of a robot standing/walking on the floor (median of the 0929/0930 sets on this FK basis: 0.045-0.051).
                           # NOT applied to the 1002 sit set: after the SMPL floor correction its robot already sits at the same clearance.


def snap_one(fk, pkl, win_s=1.5, stand_lo=0.70, stand_hi=0.78):
    d = joblib.load(pkl)
    key = next(iter(d))
    rv = d[key]
    pos = E.fk_positions(fk, rv)
    names = list(fk.body_names)
    fidx = foot_idx(names)
    low = pos[:, fidx, 2].min(1)
    pel = pos[:, names.index("pelvis"), 2]
    fps = float(rv["fps"])
    n = max(3, int(round(win_s * fps)))
    env = uniform_filter1d(minimum_filter1d(low, size=n, mode="nearest"), size=n, mode="nearest")
    w = np.clip((uniform_filter1d(pel, size=n, mode="nearest") - stand_lo) / (stand_hi - stand_lo), 0.0, 1.0)
    shift = w * np.maximum(env - STAND_FOOT_TARGET, 0.0)   # absolute target, not a per-clip percentile
    rv["root_trans_offset"][:, 2] = rv["root_trans_offset"][:, 2] - shift.astype(np.float32)
    joblib.dump(d, pkl, compress=True)
    return float(shift.max()), float(np.median(low[pel > stand_hi])) if (pel > stand_hi).any() else np.nan


def main():
    mode, root = sys.argv[1], Path(sys.argv[2])
    fk = E.Humanoid_Batch(E._Cfg())
    clips = sorted(p.stem for p in (root / "robot").glob("*.pkl"))
    names = list(fk.body_names)
    if mode == "snap":
        bak = root / "robot_orig_backup"
        if bak.exists():
            raise SystemExit(f"{bak} already exists -- refusing to snap twice")
        shutil.copytree(root / "robot", bak)
        mx = []
        for c in clips:
            m, _ = snap_one(fk, root / "robot" / f"{c}.pkl")
            mx.append(m)
        mx = np.array(mx)
        print(f"snapped {len(clips)} clips; max root-z shift per clip: median {np.median(mx):.3f} m, range [{mx.min():.3f}, {mx.max():.3f}]")
        print(f"originals -> {bak}")
        return

    # ---------------- check ----------------
    fidx = foot_idx(names)
    contact = [names.index(b) for b in E.SEAT_CONTACT_BODIES if b in names]
    stand_clear, gaps, thru, inside_seated, f0_clear, f0_overlap = [], [], [], [], [], []
    bad_thru, bad_in = [], []
    for c in clips:
        rv = joblib.load(root / "robot" / f"{c}.pkl")
        rv = rv[next(iter(rv))]
        pos = E.fk_positions(fk, rv)
        env = json.loads((root / "envs" / f"{c}.json").read_text())
        cx, cy, top = env["seat_center"]
        yaw = env["seat_yaw"]
        hx, hy = env["seat_size"][0] / 2, env["seat_size"][1] / 2
        th = env["seat_size"][2]
        pel = pos[:, names.index("pelvis"), 2]
        seated = pel < pel.min() + E.SEATED_BAND
        standing = pel > 0.75
        low = pos[:, fidx, 2].min(1)
        stand_clear.append(float(np.median(low[standing])) if standing.any() else np.nan)
        gaps.append(float(pos[seated][:, contact, 2].min() - top))
        # slab-local coordinates of every body origin
        dx, dy = pos[..., 0] - cx, pos[..., 1] - cy
        lx = np.cos(yaw) * dx + np.sin(yaw) * dy
        ly = -np.sin(yaw) * dx + np.cos(yaw) * dy
        inxy = (np.abs(lx) < hx - 0.01) & (np.abs(ly) < hy - 0.01)
        inz = (pos[..., 2] < top - 0.01) & (pos[..., 2] > top - th + 0.005)      # inside the slab volume, 1 cm margin
        inside = inxy & inz                                                    # (T, B)
        # (a) outside the seated segment: any link inside the slab = walking through the chair
        n_thru = int(inside[~seated].any(axis=1).sum())
        # (b) inside the seated segment: only the designated contact links may be at/below the top surface
        nc = [i for i in range(len(names)) if i not in contact]
        n_in = int(inside[seated][:, nc].any(axis=1).sum())
        thru.append(n_thru); inside_seated.append(n_in)
        if n_thru > 3: bad_thru.append((c[-8:], n_thru))
        if n_in > 3: bad_in.append((c[-8:], n_in))
        f0_clear.append(float(low[0]))
        f0_overlap.append(bool(inside[0].any()))
    sc = np.array(stand_clear); g = np.array(gaps)
    print(f"clips {len(clips)}")
    print(f"(1) standing foot clearance (lowest foot link above the floor): median {np.nanmedian(sc):.3f} m  p90 {np.nanpercentile(sc, 90):.3f}  max {np.nanmax(sc):.3f}")
    print(f"(2) seat contact gap at seated frames: mean {g.mean():+.4f}  range [{g.min():+.4f}, {g.max():+.4f}]  |gap|>3mm on {int((np.abs(g) > 0.003).sum())}/{len(g)}")
    print(f"(3a) clips with robot links INSIDE the slab outside the seated segment (>3 frames): {len(bad_thru)}/{len(clips)}  {bad_thru[:8]}")
    print(f"(3b) clips with NON-contact links inside the slab while seated (>3 frames): {len(bad_in)}/{len(clips)}  {bad_in[:8]}")
    print(f"(4) spawn frame 0: lowest foot link height median {np.median(f0_clear):.3f} m (max {np.max(f0_clear):.3f}); any link inside the slab at frame 0: {int(sum(f0_overlap))}/{len(clips)}")


if __name__ == "__main__":
    main()
