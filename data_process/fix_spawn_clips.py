#!/usr/bin/env python3
"""Fix the clips of picoset_20261002_sit whose reference spawns floating / with a velocity spike.

Run with the env_isaaclab python:
  /home/grease/miniforge3/envs/env_isaaclab/bin/python data_process/fix_spawn_clips.py <dataset> clip_001 clip_008 ...

Two defects, found on 9 of 76 clips (they terminated in the first 0.3 s in sim):
  1. STANDING HOVER.  In these clips the robot's standing / walking frames never touch the floor: the lowest ankle link
     stays 0.06-0.13 m up (a healthy clip: 0.02-0.045), pelvis 0.86-0.89 m instead of 0.78 m.  Fixed with a smoothed
     foot-contact floor correction of the root z, applied ONLY while the robot is standing (pelvis high), so the seated
     frames -- and the chair, which was fitted to them -- are untouched.  Target = 0.045 m (the ankle-link clearance of
     a robot standing in the 0929 / 0930 sets, same FK basis).
  2. FIRST-FRAME VELOCITY SPIKE.  The IK jumps from the default pose in the first frames: 9-25 rad/s reference joint
     speed at the start (healthy clips ~2 rad/s), which the policy cannot track from the reset state.  The first
     frames up to the point where the reference is calm (< 3 rad/s for 5 consecutive frames; at most 15 frames) are cropped
     from robot/, smpl/ (x 50/30) and envs/ so the three stay frame-aligned.

Originals are copied to <dataset>/spawnfix_backup/{robot,smpl,envs}/ first.
"""
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S  # noqa: E402

E = S.E
STAND_TARGET = 0.045
CALM = 3.0           # rad/s
MAX_CROP = 15        # frames @ 30 fps


def main():
    root = Path(sys.argv[1]); clips = sys.argv[2:]
    fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names); fidx = S.foot_idx(names)
    bak = root / "spawnfix_backup"
    for c in clips:
        stem = c if c.startswith("pico1002sit_") else f"pico1002sit_{c}"
        for sub, ext in (("robot", ".pkl"), ("smpl", ".pkl"), ("envs", ".pkl")):
            (bak / sub).mkdir(parents=True, exist_ok=True)
            dst = bak / sub / (stem + ext)
            if not dst.exists():
                shutil.copy2(root / sub / (stem + ext), dst)
        # ---------- robot ----------
        rp = root / "robot" / f"{stem}.pkl"
        d = joblib.load(rp); key = next(iter(d)); rv = d[key]
        fps = float(rv["fps"])
        pos = E.fk_positions(fk, rv)
        low = pos[:, fidx, 2].min(1); pel = pos[:, names.index("pelvis"), 2]
        spd = np.abs(np.diff(rv["dof"], axis=0)).max(1) * fps
        n_crop = 0
        for i in range(MAX_CROP + 1):
            if spd[i:i + 5].max() < CALM:
                n_crop = i; break
        else:
            n_crop = MAX_CROP
        w = np.clip((uniform_filter1d(pel, 15, mode="nearest") - 0.70) / 0.08, 0.0, 1.0)
        # contact snapping: the supporting (= lowest) foot is on the floor in every standing / walking frame, so the
        # per-frame lowest-foot height (lightly smoothed, 5 frames) is the floor offset.  A rolling MINIMUM over 0.5 s
        # (first attempt) left the median clearance at 0.10 m in these clips because both feet keep moving.
        env = uniform_filter1d(low, 5, mode="nearest")
        shift = (w * np.maximum(env - STAND_TARGET, 0.0)).astype(np.float32)
        before = (float(np.median(low[pel > 0.74])), float(pel[0]), float(spd[:3].max()))
        rv["root_trans_offset"] = rv["root_trans_offset"].copy()
        rv["root_trans_offset"][:, 2] -= shift
        keep = slice(n_crop, None)
        for k, v in list(rv.items()):
            if isinstance(v, np.ndarray) and v.shape[:1] == (len(low),):
                rv[k] = v[keep]
        d[key] = rv
        joblib.dump(d, rp, compress=True)
        pos2 = E.fk_positions(fk, rv)
        low2 = pos2[:, fidx, 2].min(1); pel2 = pos2[:, names.index("pelvis"), 2]
        spd2 = np.abs(np.diff(rv["dof"], axis=0)).max(1) * fps
        after = (float(np.median(low2[pel2 > 0.74])), float(pel2[0]), float(spd2[:3].max()))
        # ---------- smpl (50 fps) ----------
        sp = root / "smpl" / f"{stem}.pkl"
        s = joblib.load(sp); skey = None if "pose_aa" in s else next(iter(s)); sv = s if skey is None else s[skey]
        T_s = sv["transl"].shape[0]
        # the aligned SMPL length is ceil((T_robot - 1) * smpl_fps / robot_fps) (checked on all 76 original pairs)
        T_r_new = len(low) - n_crop
        S_new = int(np.ceil((T_r_new - 1) * float(sv["fps"]) / fps - 1e-9)) + 0
        m = T_s - S_new
        for k, v in list(sv.items()):
            if isinstance(v, np.ndarray) and v.shape[:1] == (T_s,) and k not in ("original_pose_aa",):
                sv[k] = v[m:]
        joblib.dump(s, sp)
        # ---------- chair trajectory ----------
        ep = root / "envs" / f"{stem}.pkl"
        o = joblib.load(ep); okey = next(iter(o))
        for k in ("root_pos", "root_quat"):
            o[okey][k] = o[okey][k][n_crop:]
        joblib.dump(o, ep, compress=True)
        print(f"{stem[-8:]}: crop {n_crop:2d} fr (smpl {m}) | standing clearance {before[0]:.3f} -> {after[0]:.3f} | pelvis f0 {before[1]:.3f} -> {after[1]:.3f} | "
              f"first-frame speed {before[2]:.1f} -> {after[2]:.1f} rad/s | max root-z shift {shift.max():.3f} m | frames robot {len(low)}->{len(low2)}, smpl {T_s}->{sv['transl'].shape[0]}")


if __name__ == "__main__":
    main()
