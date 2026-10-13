#!/usr/bin/env python
"""Per-run correlation between measured torque and the PD-predicted torque,
for waist_roll / waist_pitch.

    tau_est  = motor_torque.csv                       (measured on the robot)
    tau_sim  = kp*(q_target - q) - kd*dq              (what the PD loop should give)

`verify_pd_law.py` showed 27/29 joints match at corr >= 0.98 pooled, but
waist_roll/pitch do not. `diagnose_waist.py` traced that to the commanded
target exceeding the +-0.52 rad URDF limit. This breaks the result out per run
so the failure can be attributed to specific sessions/policies rather than to
the robot.

`waist_yaw` and `left_knee` are carried as controls: they share the waist chain
and the leg chain respectively and should stay clean in every run.

Usage:
  .venv_sim/bin/python sim2real/waist_corr_per_run.py
  .venv_sim/bin/python sim2real/waist_corr_per_run.py --sort corr
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_constants import KD, KP, list_runs, load_run  # noqa: E402

ROLL, PITCH, YAW, KNEE = 13, 14, 12, 3
LIMIT = 0.52  # rad, g1_29dof.urdf waist_roll / waist_pitch


def policy_of(run_dir):
    try:
        m = json.load(open(os.path.join(run_dir, "metadata.json")))
        p = m["robot_config"]["model_path"]
        return p.replace("policy/", "").replace("/model_decoder.onnx", "")
    except Exception:  # noqa: BLE001
        return "?"


def stats(run_dir):
    d = load_run(run_dir)
    q, dq, qt, tau = d["q"], d["dq"], d["q_target"], d["tau"]
    if any(np.isnan(v).any() for v in (q, dq, qt, tau)):
        return {"skip": "NaN"}
    mv = np.abs(dq).max(axis=1) > 0.05
    if mv.sum() < 200:
        return {"skip": "idle"}
    lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
    q, dq, qt, tau = q[lo:hi], dq[lo:hi], qt[lo:hi], tau[lo:hi]

    out = {"n": len(q), "policy": policy_of(run_dir)}
    for j, key in ((ROLL, "roll"), (PITCH, "pitch"), (YAW, "yaw"), (KNEE, "knee")):
        pred = KP[j] * (qt[:, j] - q[:, j]) - KD[j] * dq[:, j]
        t = tau[:, j]
        c = np.corrcoef(t, pred)[0, 1] if t.std() > 1e-9 and pred.std() > 1e-9 else np.nan
        s = float(pred @ t / (pred @ pred)) if (pred @ pred) > 1e-9 else np.nan
        out[f"{key}_corr"] = c
        out[f"{key}_slope"] = s
    out["roll_over"] = 100.0 * float((np.abs(qt[:, ROLL]) > LIMIT).mean())
    out["pitch_over"] = 100.0 * float((np.abs(qt[:, PITCH]) > LIMIT).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/grease/g1_robot_data")
    ap.add_argument("--sort", choices=["name", "corr", "policy"], default="policy")
    a = ap.parse_args()

    rows, skipped = [], []
    for r in list_runs(a.root):
        try:
            s = stats(r)
        except Exception as exc:  # noqa: BLE001
            skipped.append((os.path.relpath(r, a.root), f"unreadable: {exc}"))
            continue
        if "skip" in s:
            skipped.append((os.path.relpath(r, a.root), s["skip"]))
            continue
        s["run"] = os.path.relpath(r, a.root)
        rows.append(s)

    if a.sort == "corr":
        rows.sort(key=lambda x: x["pitch_corr"])
    elif a.sort == "policy":
        rows.sort(key=lambda x: (x["policy"], x["run"]))
    else:
        rows.sort(key=lambda x: x["run"])

    hdr = (f"{'run':<40}{'policy':<20}{'n':>6}"
           f"{'roll r':>8}{'roll sl':>9}{'roll>lim':>9}"
           f"{'pitch r':>9}{'pitch sl':>10}{'pitch>lim':>10}"
           f"{'yaw r':>7}{'knee r':>8}")
    print(hdr)
    print("-" * len(hdr))
    for s in rows:
        print(f"{s['run']:<40}{s['policy']:<20}{s['n']:>6}"
              f"{s['roll_corr']:>8.3f}{s['roll_slope']:>9.3f}{s['roll_over']:>8.1f}%"
              f"{s['pitch_corr']:>9.3f}{s['pitch_slope']:>10.3f}{s['pitch_over']:>9.1f}%"
              f"{s['yaw_corr']:>7.3f}{s['knee_corr']:>8.3f}")

    print(f"\n{len(rows)} runs analysed, {len(skipped)} skipped")
    for name, why in skipped:
        print(f"   skip {name}: {why}")

    # aggregate by policy
    print("\n=== by policy (frame-weighted) ===")
    print(f"{'policy':<22}{'runs':>6}{'frames':>9}"
          f"{'roll r':>9}{'pitch r':>9}{'roll>lim':>10}{'pitch>lim':>11}")
    pols = sorted({s["policy"] for s in rows})
    for p in pols:
        sub = [s for s in rows if s["policy"] == p]
        w = np.array([s["n"] for s in sub], dtype=float)
        w /= w.sum()
        print(f"{p:<22}{len(sub):>6}{int(sum(s['n'] for s in sub)):>9}"
              f"{np.dot(w, [s['roll_corr'] for s in sub]):>9.3f}"
              f"{np.dot(w, [s['pitch_corr'] for s in sub]):>9.3f}"
              f"{np.dot(w, [s['roll_over'] for s in sub]):>9.1f}%"
              f"{np.dot(w, [s['pitch_over'] for s in sub]):>10.1f}%")


if __name__ == "__main__":
    main()
