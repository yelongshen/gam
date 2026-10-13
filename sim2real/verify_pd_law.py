#!/usr/bin/env python
"""Step 0: verify the low-level PD law against measured torque, on real data.

Before any sim comparison is meaningful we must confirm that the commanded
target and gains we reconstruct offline actually reproduce the torque the robot
applied. The deploy binary sets `tau_ff = 0` and `dq_target = 0`
(`g1_deploy_onnx_ref.cpp:3147-3150`), so the motor-side law is simply

    tau_pred[i] = kp[i] * (q_target[i] - q[i]) - kd[i] * dq[i]

and `motor_torque.csv` holds the measured `tau_est`. What is left over,

    residual = tau_est - tau_pred

is friction + gearing loss + motor-model error + any gain scaling we failed to
reproduce. This also doubles as a **joint-order check**: if `motor_torque.csv`
were in a different order than `q.csv` (it is written WITHOUT the remap that
`q.csv` gets -- see deploy_constants.py), the correlation would collapse for
the asymmetric joints.

Usage:
  .venv_sim/bin/python sim2real/verify_pd_law.py                    # all runs
  .venv_sim/bin/python sim2real/verify_pd_law.py --max-runs 5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_constants import (  # noqa: E402
    EFFORT_LIMIT,
    JOINT_NAMES,
    KD,
    KP,
    list_runs,
    load_run,
)


def analyse(run_dir, active_only=True):
    d = load_run(run_dir)
    q, dq, qt, tau = d["q"], d["dq"], d["q_target"], d["tau"]

    tau_pred = KP[None, :] * (qt - q) - KD[None, :] * dq
    resid = tau - tau_pred

    if active_only:
        # Drop the idle head/tail: before the policy starts, q_target is the
        # init ramp rather than a policy action, which is a different regime.
        moving = np.abs(dq).max(axis=1) > 0.05
        if moving.sum() > 50:
            lo, hi = np.argmax(moving), len(moving) - np.argmax(moving[::-1])
            sl = slice(lo, hi)
            q, dq, qt, tau = q[sl], dq[sl], qt[sl], tau[sl]
            tau_pred, resid = tau_pred[sl], resid[sl]

    return dict(run=os.path.basename(run_dir), n=len(q), tau=tau,
                tau_pred=tau_pred, resid=resid, q=q, dq=dq, q_target=qt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--root", default="/home/grease/g1_robot_data")
    a = ap.parse_args()

    runs = list_runs(a.root)
    if a.max_runs:
        runs = runs[: a.max_runs]
    print(f"analysing {len(runs)} runs\n")

    all_tau, all_pred = [], []
    per_run = []
    for r in runs:
        try:
            res = analyse(r)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {os.path.basename(r)}: {exc}")
            continue
        all_tau.append(res["tau"])
        all_pred.append(res["tau_pred"])
        per_run.append(res)

    tau = np.concatenate(all_tau)
    pred = np.concatenate(all_pred)
    resid = tau - pred
    print(f"pooled frames: {len(tau)}  ({len(tau) * 0.02:.0f} s of robot time)\n")

    # global agreement
    c = np.corrcoef(tau.ravel(), pred.ravel())[0, 1]
    rmse = float(np.sqrt((resid**2).mean()))
    print(f"GLOBAL   corr(tau_est, tau_pred) = {c:.4f}   RMSE = {rmse:.3f} Nm")
    print(f"         tau_est  std = {tau.std():.3f} Nm")
    print(f"         residual std = {resid.std():.3f} Nm "
          f"({100 * resid.std() / tau.std():.1f}% of signal)\n")

    print(f"{'joint':24s}{'corr':>7s}{'slope':>8s}{'RMSE':>8s}"
          f"{'|tau|p95':>10s}{'resid p95':>11s}{'sat%':>7s}")
    print("-" * 75)
    rows = []
    for j in range(29):
        tj, pj = tau[:, j], pred[:, j]
        cj = np.corrcoef(tj, pj)[0, 1] if tj.std() > 1e-9 and pj.std() > 1e-9 else np.nan
        # least-squares slope tau_est ~ slope * tau_pred (1.0 = perfect gain)
        slope = float(pj @ tj / (pj @ pj)) if (pj @ pj) > 1e-9 else np.nan
        rj = tj - pj
        sat = 100.0 * float((np.abs(tj) > 0.95 * EFFORT_LIMIT[j]).mean())
        rows.append((JOINT_NAMES[j], cj, slope, float(np.sqrt((rj**2).mean())),
                     float(np.percentile(np.abs(tj), 95)),
                     float(np.percentile(np.abs(rj), 95)), sat))
        print(f"{JOINT_NAMES[j]:24s}{cj:7.3f}{slope:8.3f}{rows[-1][3]:8.2f}"
              f"{rows[-1][4]:10.2f}{rows[-1][5]:11.2f}{sat:7.1f}")

    bad = [r for r in rows if not np.isnan(r[1]) and r[1] < 0.8]
    print(f"\njoints with corr < 0.80: {len(bad)}")
    for r in bad:
        print(f"   {r[0]:24s} corr={r[1]:.3f} slope={r[2]:.3f}")

    slopes = np.array([r[2] for r in rows])
    print(f"\nslope (tau_est / tau_pred): median {np.nanmedian(slopes):.3f}, "
          f"range {np.nanmin(slopes):.3f}-{np.nanmax(slopes):.3f}")
    print("   slope ~1.0 => gains reproduced correctly")
    print("   slope <1.0 => real torque smaller than commanded (friction/derating)")


if __name__ == "__main__":
    main()
