#!/usr/bin/env python
"""Per-run QC + PD-law verification, with data-driven exclusion.

Two independent exclusion sources are combined:

1. **Documented** (`sim2real/online_eval_*_report.md`,
   `online_eval_policy_comparison.md` §6.3): the 09-19 session is confounded --
   degraded stream plus `reference/example/` instead of `real_example/` -- and
   `g1_run_0917` has no evaluation report at all.

2. **Measured**: per-run agreement between `tau_est` and the reconstructed PD
   law. A run whose torque does not obey
   `tau = kp(q_target - q) - kd*dq` is unusable for dynamics identification
   regardless of what any report says, because it means the gains, the joint
   order, or the logging itself differed for that run.

Note the two do NOT have to agree, and mostly should not: a degraded *stream*
corrupts the reference the policy was chasing, but the robot's own
(q, dq, tau) still obey the actuator law. Only (2) invalidates a run for
dynamics work. Both are reported so the difference is visible.

Usage:
  .venv_sim/bin/python sim2real/qc_runs_pd.py
  .venv_sim/bin/python sim2real/qc_runs_pd.py --exclude-docs   # also drop 09-17/09-19
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_constants import (  # noqa: E402
    JOINT_NAMES,
    KD,
    KP,
    list_runs,
    load_run,
)

# Joints excluded from the PD score: the waist roll/pitch pair is commanded far
# outside its +-0.52 rad URDF limit (measured: +-1.71 / +-1.94 rad), so the law
# cannot hold there and would swamp any per-run signal. Kept in the data, just
# not used to judge a run.
WAIST_RP = [13, 14]
SCORE_JOINTS = [j for j in range(29) if j not in WAIST_RP]

# Documented problem sessions (see module docstring).
DOC_EXCLUDE_SUBSTR = ("g1_run_0919", "20260919", "g1_run_0917", "09182026")


def run_stats(run_dir):
    d = load_run(run_dir)
    q, dq, qt, tau = d["q"], d["dq"], d["q_target"], d["tau"]
    if len(q) < 200:
        return None

    moving = np.abs(dq).max(axis=1) > 0.05
    if moving.sum() < 100:
        return None
    lo, hi = int(np.argmax(moving)), len(moving) - int(np.argmax(moving[::-1]))
    sl = slice(lo, hi)
    q, dq, qt, tau = q[sl], dq[sl], qt[sl], tau[sl]

    pred = KP[None, :] * (qt - q) - KD[None, :] * dq
    t_s, p_s = tau[:, SCORE_JOINTS], pred[:, SCORE_JOINTS]
    corr = float(np.corrcoef(t_s.ravel(), p_s.ravel())[0, 1])
    slope = float((p_s * t_s).sum() / (p_s * p_s).sum())
    rmse = float(np.sqrt(((t_s - p_s) ** 2).mean()))

    # data-integrity flags
    frozen = float((np.abs(np.diff(q, axis=0)).max(axis=1) == 0).mean())
    nan_frac = float(np.isnan(q).mean() + np.isnan(dq).mean()
                     + np.isnan(qt).mean() + np.isnan(tau).mean())
    # A NaN anywhere makes corr/slope NaN, which silently poisons the pooled
    # statistics if it is not caught here.
    bad_num = bool(np.isnan(corr) or np.isnan(slope) or np.isnan(rmse)
                   or nan_frac > 0)

    return dict(run=run_dir, n=len(q), corr=corr, slope=slope, rmse=rmse,
                frozen=frozen, nan=nan_frac, bad_num=bad_num,
                dq_max=float(np.abs(dq).max()), tau_max=float(np.abs(tau).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/grease/g1_robot_data")
    ap.add_argument("--min-corr", type=float, default=0.95)
    ap.add_argument("--slope-tol", type=float, default=0.15,
                    help="|slope-1| above this fails the run")
    ap.add_argument("--exclude-docs", action="store_true")
    a = ap.parse_args()

    rows = []
    for r in list_runs(a.root):
        try:
            s = run_stats(r)
        except Exception as exc:  # noqa: BLE001
            print(f"  unreadable: {os.path.basename(r)} ({exc})")
            continue
        if s:
            rows.append(s)

    print(f"{'run':<42}{'n':>7}{'corr':>8}{'slope':>8}{'RMSE':>8}"
          f"{'frozen%':>9}{'dq_max':>8}  verdict")
    print("-" * 100)
    keep, drop = [], []
    for s in sorted(rows, key=lambda x: x["corr"]):
        rel = os.path.relpath(s["run"], a.root)
        doc_bad = any(k in s["run"] for k in DOC_EXCLUDE_SUBSTR)
        pd_bad = (s["corr"] < a.min_corr) or (abs(s["slope"] - 1.0) > a.slope_tol)
        integ_bad = s["frozen"] > 0.5 or s["nan"] > 0 or s["bad_num"]

        why = []
        if pd_bad:
            why.append("PD-FAIL")
        if integ_bad:
            why.append("INTEGRITY")
        if doc_bad and a.exclude_docs:
            why.append("DOC")
        verdict = ",".join(why) if why else "ok"
        if why:
            drop.append(s)
        else:
            keep.append(s)
        tag = "  [doc-flagged]" if doc_bad and not a.exclude_docs else ""
        print(f"{rel:<42}{s['n']:>7}{s['corr']:>8.4f}{s['slope']:>8.3f}"
              f"{s['rmse']:>8.2f}{100*s['frozen']:>9.1f}{s['dq_max']:>8.1f}"
              f"  {verdict}{tag}")

    print(f"\nkeep {len(keep)} runs, drop {len(drop)}")
    if drop:
        print("dropped:")
        for s in drop:
            print(f"   {os.path.relpath(s['run'], a.root)}")

    # pooled PD verification on the kept runs
    print("\n=== pooled PD law on KEPT runs ===")
    T, P = [], []
    for s in keep:
        d = load_run(s["run"])
        q, dq, qt, tau = d["q"], d["dq"], d["q_target"], d["tau"]
        moving = np.abs(dq).max(axis=1) > 0.05
        lo, hi = int(np.argmax(moving)), len(moving) - int(np.argmax(moving[::-1]))
        sl = slice(lo, hi)
        T.append(tau[sl])
        P.append(KP[None, :] * (qt[sl] - q[sl]) - KD[None, :] * dq[sl])
    tau = np.concatenate(T)
    pred = np.concatenate(P)
    ts, ps = tau[:, SCORE_JOINTS], pred[:, SCORE_JOINTS]
    print(f"frames {len(tau)} ({len(tau)*0.02:.0f} s)")
    print(f"corr {np.corrcoef(ts.ravel(), ps.ravel())[0,1]:.4f}   "
          f"RMSE {np.sqrt(((ts-ps)**2).mean()):.3f} Nm   "
          f"resid/signal {100*(ts-ps).std()/ts.std():.1f}%")

    print(f"\n{'joint':24s}{'corr':>8}{'slope':>8}{'RMSE':>8}")
    for j in range(29):
        tj, pj = tau[:, j], pred[:, j]
        c = np.corrcoef(tj, pj)[0, 1] if tj.std() > 1e-9 else np.nan
        sl_ = float(pj @ tj / (pj @ pj)) if (pj @ pj) > 1e-9 else np.nan
        mark = "   <-- waist r/p, over-limit cmd" if j in WAIST_RP else ""
        print(f"{JOINT_NAMES[j]:24s}{c:8.3f}{sl_:8.3f}"
              f"{np.sqrt(((tj-pj)**2).mean()):8.2f}{mark}")


if __name__ == "__main__":
    main()
