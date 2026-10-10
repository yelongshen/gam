#!/usr/bin/env python3
"""Rule-based detector for hand/arm retargeting errors (paired smpl/ + robot/ pkls).

Idea: the retargeted G1 arm should reproduce the SMPL arm *shape*. Shape features that are
invariant to root orientation / global frame are compared frame by frame:

  elbow_ang_err   |elbow flexion angle(SMPL) - elbow flexion angle(G1 FK)|   (deg)
  reach_err       |wrist-shoulder distance / arm length| difference (SMPL vs G1)    (ratio)
  hand_dist_err   |wrist-to-wrist distance / shoulder width| difference            (ratio)

plus joint-limit features straight from the robot `dof`:

  arm_limit_frac  fraction of frames where any shoulder/elbow/wrist joint is within 2% of its
                  limit range of a hard limit (retargeter clipped it)
  wrist_limit_frac same, wrist joints only
  arm_spike_deg_s  99.9th percentile arm joint speed (deg/s) -- IK flips / discontinuities

A clip is flagged when any robust-aggregated feature exceeds a threshold. Thresholds are
chosen from the distribution over the whole set (median + k*MAD) and printed, so they can be
re-fit for other datasets. Output: <out>/arm_retarget_qc_<set>.csv

Usage (run with .venv_sim, needs mujoco):
    .venv_sim/bin/python sim2real/detect_arm_retarget_errors.py --set amass_evalset
"""
import argparse
import csv
import os

import joblib
import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.expanduser("~/ego_dataset")
XML = f"{REPO}/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icl_hard_set")
HW = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch",
      "left_ankle_roll", "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
      "right_ankle_pitch", "right_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
      "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
      "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch",
      "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
      "right_wrist_pitch", "right_wrist_yaw"]
ARM_IDX = list(range(15, 29))
WRIST_IDX = [19, 20, 21, 26, 27, 28]
# SMPL-24: 16/17 shoulders, 18/19 elbows, 20/21 wrists
SMPL = {"L": (16, 18, 20), "R": (17, 19, 21)}
ROBOT_BODIES = {"L": ("left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link"),
                "R": ("right_shoulder_pitch_link", "right_elbow_link", "right_wrist_yaw_link")}


def first(d):
    return d[next(iter(d))] if isinstance(d, dict) and "dof" not in d else d


def angle(a, b, c):
    """Angle at b between (a-b) and (c-b), degrees; arrays (T,3)."""
    u, v = a - b, c - b
    cs = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


class G1:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.adr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint")]
                    for n in HW]
        self.lim = np.array([self.m.jnt_range[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint")]
                             for n in HW])
        self.bid = {k: [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in v]
                    for k, v in ROBOT_BODIES.items()}

    def arm_points(self, dof):
        """(T, side, 3 pts, 3) arm points in the pelvis frame (root fixed at identity)."""
        out = np.zeros((len(dof), 2, 3, 3))
        for t in range(len(dof)):
            self.d.qpos[:] = 0
            self.d.qpos[3] = 1.0
            for i, a in enumerate(self.adr):
                self.d.qpos[a] = dof[t, i]
            mujoco.mj_kinematics(self.m, self.d)
            for si, k in enumerate("LR"):
                for pi, b in enumerate(self.bid[k]):
                    out[t, si, pi] = self.d.xpos[b]
        return out


def qc_clip(g, smpl_path, robot_path, stride=1):
    s = joblib.load(smpl_path)
    r = first(joblib.load(robot_path))
    sj = np.asarray(s["smpl_joints"], dtype=np.float64)
    sfps, rfps = float(s.get("fps", 50)), float(r.get("fps", 30))
    dof = np.asarray(r["dof"], dtype=np.float64)[::stride]
    T = len(dof)
    idx = np.minimum(np.round(np.arange(T) * stride / rfps * sfps).astype(int), len(sj) - 1)
    sj = sj[idx]
    pts = g.arm_points(dof)
    f = {}
    ea, ra = [], []
    for si, k in enumerate("LR"):
        a, b, c = SMPL[k]
        e_s = angle(sj[:, a], sj[:, b], sj[:, c])
        e_r = angle(pts[:, si, 0], pts[:, si, 1], pts[:, si, 2])
        ea.append(np.abs(e_s - e_r))
        len_s = np.linalg.norm(sj[:, a] - sj[:, b], axis=1) + np.linalg.norm(sj[:, b] - sj[:, c], axis=1)
        len_r = np.linalg.norm(pts[:, si, 0] - pts[:, si, 1], axis=1) + \
            np.linalg.norm(pts[:, si, 1] - pts[:, si, 2], axis=1)
        reach_s = np.linalg.norm(sj[:, c] - sj[:, a], axis=1) / len_s
        reach_r = np.linalg.norm(pts[:, si, 2] - pts[:, si, 0], axis=1) / len_r
        ra.append(np.abs(reach_s - reach_r))
    ea, ra = np.concatenate(ea), np.concatenate(ra)
    f["elbow_ang_err_p95"] = float(np.percentile(ea, 95))
    f["elbow_ang_err_mean"] = float(ea.mean())
    f["reach_err_p95"] = float(np.percentile(ra, 95))
    sw_s = np.linalg.norm(sj[:, 16] - sj[:, 17], axis=1)
    sw_r = np.linalg.norm(pts[:, 0, 0] - pts[:, 1, 0], axis=1)
    hd_s = np.linalg.norm(sj[:, 20] - sj[:, 21], axis=1) / (sw_s + 1e-9)
    hd_r = np.linalg.norm(pts[:, 0, 2] - pts[:, 1, 2], axis=1) / (sw_r + 1e-9)
    f["hand_dist_err_p95"] = float(np.percentile(np.abs(hd_s - hd_r), 95))
    lo, hi = g.lim[:, 0], g.lim[:, 1]
    rng = hi - lo
    at_lim = (dof < lo + 0.02 * rng) | (dof > hi - 0.02 * rng)
    f["arm_limit_frac"] = float(at_lim[:, ARM_IDX].any(1).mean())
    f["wrist_limit_frac"] = float(at_lim[:, WRIST_IDX].any(1).mean())
    sp = np.abs(np.diff(np.degrees(dof[:, ARM_IDX]), axis=0)) * rfps / stride
    f["arm_spike_deg_s"] = float(np.percentile(sp, 99.9)) if len(sp) else 0.0
    f["dur_s"] = T * stride / rfps
    return f


FEATS = ["elbow_ang_err_p95", "reach_err_p95", "hand_dist_err_p95", "arm_limit_frac",
         "wrist_limit_frac", "arm_spike_deg_s"]

_G = None
_STRIDE = 1
_SDIR = _RDIR = ""


def _init(sdir, rdir, stride):
    global _G, _STRIDE, _SDIR, _RDIR
    _G, _STRIDE, _SDIR, _RDIR = G1(), stride, sdir, rdir


def _work(n):
    try:
        return {"name": n, **qc_clip(_G, f"{_SDIR}/{n}.pkl", f"{_RDIR}/{n}.pkl", _STRIDE)}
    except Exception as exc:  # corrupt / odd clip
        return {"name": n, "error": f"{type(exc).__name__}: {exc}"}


def main():
    import json
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="amass_evalset")
    ap.add_argument("--k", type=float, default=4.0, help="flag if feature > median + k*MAD")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth robot frame (speed)")
    ap.add_argument("--thr-file", help="reuse thresholds fitted on another set (json); "
                                       "default: fit on this set and save")
    ap.add_argument("--names-file", help="only these clip names (one per line)")
    args = ap.parse_args()
    sdir, rdir = f"{ROOT}/{args.set}/smpl", f"{ROOT}/{args.set}/robot"
    names = sorted(fn[:-4] for fn in os.listdir(rdir) if fn.endswith(".pkl"))
    if args.names_file:
        keep = {l.strip() for l in open(args.names_file) if l.strip()}
        names = [n for n in names if n in keep]
    with mp.Pool(args.workers, initializer=_init, initargs=(sdir, rdir, args.stride)) as pool:
        res = pool.map(_work, names, chunksize=4)
    rows = [r for r in res if "error" not in r]
    for r in res:
        if "error" in r:
            print(f"[skip] {r['name']}: {r['error']}")
    os.makedirs(OUT, exist_ok=True)
    if args.thr_file:
        thr = json.load(open(args.thr_file))
    else:
        thr = {}
        for k in FEATS:
            v = np.array([r[k] for r in rows])
            med = np.median(v)
            mad = np.median(np.abs(v - med)) * 1.4826 + 1e-9
            thr[k] = float(med + args.k * mad)
        json.dump(thr, open(f"{OUT}/arm_retarget_thresholds_{args.set}.json", "w"), indent=2)
    for r in rows:
        r["flags"] = ";".join(k for k in FEATS if r[k] > thr[k])
        r["n_flags"] = len(r["flags"].split(";")) if r["flags"] else 0
        # severe = limit-pinned arms/wrists AND wrong arm shape (the S04/S09 signature)
        r["severe"] = int(r["arm_limit_frac"] >= 0.9 and r["wrist_limit_frac"] >= 0.5)
    rows.sort(key=lambda r: (-r["severe"], -r["n_flags"], -r["elbow_ang_err_p95"]))
    out = f"{OUT}/arm_retarget_qc_{args.set}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "severe", "n_flags", "flags"] + FEATS +
                           ["elbow_ang_err_mean", "dur_s"])
        w.writeheader()
        w.writerows(rows)
    print("thresholds:", {k: round(v, 3) for k, v in thr.items()})
    n = len(rows)
    print(f"{n} clips -> {out}")
    print(f"severe (arm_limit>=0.9 & wrist_limit>=0.5): {sum(r['severe'] for r in rows)} "
          f"({100 * sum(r['severe'] for r in rows) / max(n, 1):.1f}%)")
    for nf in (4, 3, 2, 1):
        print(f"  >= {nf} flags: {sum(r['n_flags'] >= nf for r in rows)}")
    print(f"{'clip':72s} sev n  " + " ".join(f"{k[:11]:>11s}" for k in FEATS))
    for r in rows[:15]:
        print(f"{r['name'][:70]:72s} {r['severe']}   {r['n_flags']}  " +
              " ".join(f"{r[k]:11.3f}" for k in FEATS))


if __name__ == "__main__":
    main()
