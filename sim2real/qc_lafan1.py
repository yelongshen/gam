#!/usr/bin/env python3
"""Quality check for lafan1_trainset / lafan1_evalset (paired smpl + robot pkls).

For every clip it records and flags:
  SMPL  : NaN/Inf, fps/duration, whether non-root `pose_aa` is all zero, pelvis-local joint
          speed, largest single-frame joint jump (teleport), root height, foot floor offset,
          bone-length drift, idle fraction.
  ROBOT : NaN/Inf, fps/duration, dof range / speed / static fraction, root z range,
          root_rot quaternion norm, spikes, frame-0 floor penetration proxy.
  PAIR  : duration mismatch, root path-length ratio, and correlation between SMPL
          pelvis-local joint speed and robot dof speed (time-aligned) -- low correlation
          means the robot file does not move like the SMPL clip.

Usage:  python sim2real/qc_lafan1.py          -> sim2real/icl_hard_set/qc_lafan1.csv
"""
import csv
import glob
import os
import sys

import joblib
import numpy as np

ROOT = os.path.expanduser("~/ego_dataset")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icl_hard_set", "qc_lafan1.csv")


def first(d):
    return d[next(iter(d))] if isinstance(d, dict) and "dof" not in d else d


def resample(x, n):
    """Linear resample 1-D array x to length n."""
    if len(x) == n:
        return x
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def qc(split, name):
    r = {"split": split, "name": name, "flags": []}
    fl = r["flags"]
    s = joblib.load(f"{ROOT}/{split}/smpl/{name}.pkl")
    rb = first(joblib.load(f"{ROOT}/{split}/robot/{name}.pkl"))

    # ---------------- SMPL ----------------
    sfps = float(s.get("fps", 50))
    sj = np.asarray(s["smpl_joints"], dtype=np.float64)
    pa = np.asarray(s["pose_aa"], dtype=np.float64)
    tr = np.asarray(s["transl"], dtype=np.float64)
    Ts = len(sj)
    r["smpl_frames"], r["smpl_fps"], r["smpl_dur_s"] = Ts, sfps, round(Ts / sfps, 1)
    bad = any(not np.isfinite(a).all() for a in (sj, pa, tr))
    r["smpl_nan"] = int(bad)
    if bad:
        fl.append("smpl_nan")
    pa3 = pa.reshape(Ts, -1, 3)
    r["smpl_nonroot_pose_zero"] = int(np.abs(pa3[:, 1:]).max() < 1e-9)
    if r["smpl_nonroot_pose_zero"]:
        fl.append("smpl_pose_aa_nonroot_all_zero")
    loc = sj - sj[:, :1]
    lv = np.linalg.norm(np.diff(loc, axis=0), axis=2).mean(1) * sfps  # m/s
    r["smpl_local_speed_mean"] = round(float(lv.mean()), 3)
    r["smpl_idle_frac"] = round(float((lv < 0.05).mean()), 3)
    jump = np.linalg.norm(np.diff(sj, axis=0), axis=2).max(1)  # m per frame
    r["smpl_max_jump_m"] = round(float(jump.max()), 3)
    if jump.max() > 0.25:
        fl.append("smpl_teleport")
    # up axis: the axis with the smallest mean |velocity| of pelvis while standing is not
    # reliable, so use the axis whose range of the lowest joint is the smallest relative
    # to the others being large: LAFAN/SMPL filtered data is Z-up like eval_subset.
    up = 2
    foot = sj[:, [10, 11], up].min(1)
    r["smpl_foot_min_z"] = round(float(foot.min()), 3)
    r["smpl_foot_z_p5"] = round(float(np.percentile(foot, 5)), 3)
    if abs(np.percentile(foot, 5)) > 0.06:
        fl.append("smpl_floor_offset")
    r["smpl_pelvis_z0"] = round(float(sj[0, 0, up]), 3)
    # bone-length drift (should be rigid)
    bones = np.linalg.norm(sj[:, 1:] - sj[:, [0] * 23], axis=2)
    drift = float((bones.std(0) / (bones.mean(0) + 1e-9)).max())
    r["smpl_bone_drift_max"] = round(drift, 3)

    # ---------------- ROBOT ----------------
    rfps = float(rb.get("fps", 30))
    dof = np.asarray(rb["dof"], dtype=np.float64)
    root = np.asarray(rb["root_trans_offset"], dtype=np.float64)
    rot = np.asarray(rb["root_rot"], dtype=np.float64)
    Tr = len(dof)
    r["robot_frames"], r["robot_fps"], r["robot_dur_s"] = Tr, rfps, round(Tr / rfps, 1)
    badr = any(not np.isfinite(a).all() for a in (dof, root, rot))
    r["robot_nan"] = int(badr)
    if badr:
        fl.append("robot_nan")
    dd = np.rad2deg(dof)
    jv = np.abs(np.diff(dd, axis=0)) * rfps
    r["robot_dof_speed_mean"] = round(float(jv.mean()), 2)
    r["robot_dof_speed_p99"] = round(float(np.percentile(jv, 99)), 1)
    r["robot_dof_speed_max"] = round(float(jv.max()), 0)
    r["robot_dof_absmax_deg"] = round(float(np.abs(dd).max()), 1)
    r["robot_static_frac"] = round(float((jv.mean(1) < 1.0).mean()), 3)
    rom = dd.max(0) - dd.min(0)
    r["robot_rom_mean"] = round(float(rom.mean()), 1)
    if jv.mean() < 3.0 and r["smpl_local_speed_mean"] > 0.3:
        fl.append("robot_dof_nearly_static_vs_smpl_moving")
    if np.abs(dd).max() > 175:
        fl.append("robot_dof_near_pi")
    if jv.max() > 2500:
        fl.append("robot_dof_spike")
    qn = np.linalg.norm(rot, axis=1)
    r["robot_quat_norm_dev"] = round(float(np.abs(qn - 1).max()), 4)
    if np.abs(qn - 1).max() > 1e-2:
        fl.append("robot_quat_not_unit")
    r["robot_root_z0"] = round(float(root[0, 2]), 3)
    r["robot_root_z_min"] = round(float(root[:, 2].min()), 3)
    r["robot_root_z_max"] = round(float(root[:, 2].max()), 3)
    if root[0, 2] < 0.6 or root[0, 2] > 0.95:
        fl.append("robot_start_height_odd")
    if root[:, 2].max() > 1.4:
        fl.append("robot_root_z_high")

    # ---------------- PAIR ----------------
    r["dur_mismatch_s"] = round(abs(Ts / sfps - Tr / rfps), 2)
    if abs(Ts / sfps - Tr / rfps) > 0.5:
        fl.append("duration_mismatch")
    sp = np.linalg.norm(np.diff(tr, axis=0), axis=1).sum()
    rp = np.linalg.norm(np.diff(root, axis=0), axis=1).sum()
    r["root_path_ratio_robot_over_smpl"] = round(float(rp / (sp + 1e-9)), 3)
    if rp / (sp + 1e-9) < 0.7 or rp / (sp + 1e-9) > 1.4:
        fl.append("root_path_ratio_off")
    n = min(Ts, Tr * 2)
    a = resample(lv, 2000)
    b = resample(jv.mean(1), 2000)
    r["corr_smplspeed_vs_dofspeed"] = round(float(np.corrcoef(a, b)[0, 1]), 3) \
        if a.std() > 0 and b.std() > 0 else 0.0
    if r["corr_smplspeed_vs_dofspeed"] < 0.3:
        fl.append("robot_not_following_smpl")
    r["flags"] = ";".join(fl)
    return r


def main():
    rows = []
    for split in ("lafan1_trainset", "lafan1_evalset"):
        for rp in sorted(glob.glob(f"{ROOT}/{split}/robot/*.pkl")):
            name = os.path.basename(rp)[:-4]
            try:
                rows.append(qc(split, name))
            except Exception as exc:
                rows.append({"split": split, "name": name, "flags": f"load_error:{type(exc).__name__}:{exc}"})
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} clips)")
    from collections import Counter
    for split in ("lafan1_trainset", "lafan1_evalset"):
        sub = [r for r in rows if r["split"] == split]
        c = Counter(f for r in sub for f in r["flags"].split(";") if f)
        print(f"\n{split}: {len(sub)} clips; clips with >=1 flag: {sum(1 for r in sub if r['flags'])}")
        for f, k in c.most_common():
            print(f"  {k:3d}  {f}")


if __name__ == "__main__":
    sys.exit(main())
