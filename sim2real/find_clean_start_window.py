#!/usr/bin/env python
"""Find the smallest window shift that gives each ICL hold-failure clip a clean start frame.

The clips are 8 s windows of full LAFAN sequences. For every hold clip this script
  1. locates the window inside the full source sequence (matching dof of frame 0),
  2. evaluates every source frame in [start - max_back, start + max_fwd] with
     SOLE-based contact (real foot collision geometry) and a CoM-in-support test,
  3. picks the clean frame closest to the original start (so as few frames as
     possible are moved) and keeps the window length by sliding the window.

Clean frame (must hold for --hold-frames consecutive frames):
  * lowest sole z      <= --sole-lo   (a foot is actually on the ground)
  * highest sole z     <= --sole-hi   (other foot not far in the air)
  * CoM xy dist to support hull <= --com-tol (support = foot points within --contact-tol of ground)
  * root tilt          <= --tilt
  * root speed         <= --root-speed

    .venv_sim/bin/python sim2real/find_clean_start_frame.py
    .venv_sim/bin/python sim2real/find_clean_start_frame.py --max-fwd-s 1.5 --max-back-s 1.0
"""
import argparse
import csv
import json
import os
import sys

import joblib
import mujoco
import numpy as np

sys.path.insert(0, os.path.expanduser("~/gam/sim2real"))
from detect_arm_retarget_errors import G1  # noqa: E402
from diag_frame0_contact import FEET, HOLD_CLIPS, geom_points, hull_dist  # noqa: E402

EGO = os.path.expanduser("~/ego_dataset")


def load_robot(p):
    rd = joblib.load(p)
    r = rd[next(iter(rd))] if "dof" not in rd else rd
    return {"dof": np.asarray(r["dof"], dtype=np.float64),
            "root": np.asarray(r["root_trans_offset"], dtype=np.float64),
            "quat": np.asarray(r["root_rot"], dtype=np.float64),  # xyzw
            "fps": float(r.get("fps", 30.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{EGO}/ICL_hardset_aligned")
    ap.add_argument("--full-dirs", nargs="*", default=[f"{EGO}/{s}/robot_v2" for s in ("lafan1_evalset", "lafan1_trainset")])
    ap.add_argument("--clips", nargs="*", default=HOLD_CLIPS)
    ap.add_argument("--sole-lo", type=float, default=0.02)
    ap.add_argument("--sole-hi", type=float, default=0.10)
    ap.add_argument("--contact-tol", type=float, default=0.02)
    ap.add_argument("--com-tol", type=float, default=0.02)
    ap.add_argument("--tilt", type=float, default=10.0)
    ap.add_argument("--root-speed", type=float, default=1.5)
    ap.add_argument("--hold-frames", type=int, default=3)
    ap.add_argument("--max-back-s", type=float, default=1.0, help="max shift earlier (s)")
    ap.add_argument("--max-fwd-s", type=float, default=1.5, help="max shift later (s)")
    ap.add_argument("--out", default="~/gam/sim2real/clean_start_frames.csv")
    a = ap.parse_args()

    g = G1()
    m, d = g.m, g.d
    body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in FEET]
    foot_geoms = [[i for i in range(m.ngeom) if m.geom_bodyid[i] == b] for b in body_ids]

    def metrics(r, t):
        d.qpos[:] = 0
        d.qpos[0:3] = r["root"][t]
        q = r["quat"][t]
        d.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        for i, adr in enumerate(g.adr):
            d.qpos[adr] = r["dof"][t, i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        P = [np.vstack([geom_points(m, d, gi) for gi in gs]) for gs in foot_geoms]
        zs = [p[:, 2].min() for p in P]
        allp = np.vstack(P)
        sup = allp[allp[:, 2] <= allp[:, 2].min() + a.contact_tol]
        off = hull_dist(d.subtree_com[1][:2], sup[:, :2])
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[0] ** 2 + q[1] ** 2), -1, 1)))
        return min(zs), max(zs), off, tilt

    rows = []
    print(f"{'clip':34s} {'orig_f':>6s} {'new_f':>6s} {'shift_s':>7s} {'clean':>5s} {'soleLo':>6s} "
          f"{'soleHi':>6s} {'com_off':>7s} {'tilt':>5s} {'v':>5s}")
    for n in a.clips:
        win = load_robot(f"{a.src}/robot/{n}.pkl")
        base = n.split("__")[0]
        best = None
        for dd in a.full_dirs:
            p = f"{dd}/{base}.pkl"
            if not os.path.exists(p):
                continue
            full = load_robot(p)
            err = np.abs(full["dof"][: len(full["dof"]) - len(win["dof"]) + 1] - win["dof"][0]).sum(1)
            o = int(err.argmin())
            if best is None or err[o] < best[2]:
                best = (full, o, err[o], dd)
        if best is None:
            raise SystemExit(f"full sequence for {base} not found in {a.full_dirs}")
        full, o, err, dd = best
        fps, W = full["fps"], len(win["dof"])
        # the ICL window was re-based (xy zeroed, z shifted); apply the same z shift to the full sequence
        dzs = win["root"][:, 2] - full["root"][o:o + W, 2]
        dz = float(np.median(dzs))
        full["root"] = full["root"].copy()
        full["root"][:, 2] += dz
        print(f"  [{n}] source={dd.split('/')[-2]}/{dd.split('/')[-1]} start_frame={o} match_err={err:.4f} "
              f"dz={dz:+.3f} (std {dzs.std():.4f})")
        lo = max(0, o - int(a.max_back_s * fps))
        hi = min(len(full["dof"]) - W, o + int(a.max_fwd_s * fps))
        idx = list(range(lo, hi + a.hold_frames))
        M = {t: metrics(full, t) for t in idx}
        vel = {t: (np.linalg.norm(full["root"][t] - full["root"][t - 1]) * fps if t > 0 else 0.0) for t in idx}
        ok_f = {t: (M[t][0] <= a.sole_lo and M[t][1] <= a.sole_hi and M[t][2] <= a.com_tol
                    and M[t][3] <= a.tilt and vel[t] <= a.root_speed) for t in idx}
        cands = [t for t in range(lo, hi + 1) if all(ok_f[t + k] for k in range(a.hold_frames))]
        clean = bool(cands)
        if clean:
            new = min(cands, key=lambda t: (abs(t - o), t))
        else:  # best effort: smallest normalised violation in range
            def score(t):
                return max(M[t][0] / a.sole_lo, M[t][1] / a.sole_hi, max(M[t][2], 0) / a.com_tol,
                           M[t][3] / a.tilt, vel[t] / a.root_speed)
            new = min(range(lo, hi + 1), key=lambda t: (score(t), abs(t - o)))
        mm = M[new]
        rows.append({"clip": n, "full_seq": base, "full_dir": os.path.basename(dd), "match_err": round(float(err), 4),
                     "orig_start_frame": o, "new_start_frame": new, "shift_frames": new - o,
                     "shift_s": round((new - o) / fps, 2), "clean": clean, "window_frames": W,
                     "sole_lo": round(mm[0], 3), "sole_hi": round(mm[1], 3), "com_off": round(mm[2], 3),
                     "tilt_deg": round(mm[3], 1), "root_speed": round(vel[new], 2),
                     "orig_sole_lo": round(M[o][0], 3) if o in M else "",
                     "orig_com_off": round(M[o][2], 3) if o in M else ""})
        print(f"{n[:34]:34s} {o:6d} {new:6d} {(new - o) / fps:7.2f} {str(clean):>5s} {mm[0]:6.3f} {mm[1]:6.3f} "
              f"{mm[2]:7.3f} {mm[3]:5.1f} {vel[new]:5.2f}" + ("" if clean else "  (best effort)"))

    out = os.path.expanduser(a.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump({r["clip"]: {k: r[k] for k in ("full_seq", "new_start_frame", "window_frames", "clean")} for r in rows},
              open(out.replace(".csv", ".json"), "w"), indent=1)
    print(f"\nwrote {out} (+ .json)")


if __name__ == "__main__":
    main()
