#!/usr/bin/env python3
"""Per-RUN (not pooled) MSE/RMSE/corr of tau_est vs nominal tau_sim, to find
which individual runs in a session are driving a bad pooled result.

Usage:
    .venv_sim/bin/python model_eval/sim2real_phaseB_per_run.py g1_run_0914
    .venv_sim/bin/python model_eval/sim2real_phaseB_per_run.py g1_run_0914 --joint waist_roll
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES, KDS, KPS  # noqa: E402
from model_eval.sim2real_phaseB_refit import ROOT, load_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--joint", default=None,
                     help="restrict detail rows to one joint name (default: ALL-29 mean only)")
    args = ap.parse_args()

    session_dir = os.path.join(ROOT, args.session)
    run_dirs = sorted(glob.glob(os.path.join(session_dir, "g1_deploy_run_*")))
    if not run_dirs:
        run_dirs = [session_dir] if os.path.isdir(session_dir) else []

    jidx = JOINT_NAMES.index(args.joint) if args.joint else None

    print(f"=== {args.session}: {len(run_dirs)} run dirs found ===\n")
    header = f"{'run':32s} {'n':>6s} {'mean|RMSE|':>11s} {'max RMSE (joint)':>28s} {'min corr (joint)':>28s}"
    if jidx is not None:
        header += f"  {'  '+args.joint+' RMSE':>16s} {args.joint+' corr':>12s}"
    print(header)

    for rd in run_dirs:
        out = load_run(rd)
        name = os.path.basename(rd)
        if out is None:
            print(f"{name:32s}  (unusable / <20 Mode-2 samples)")
            continue
        e, dq, tau = out
        finite = np.all(np.isfinite(e), 1) & np.all(np.isfinite(dq), 1) & np.all(np.isfinite(tau), 1)
        e, dq, tau = e[finite], dq[finite], tau[finite]
        n = len(e)
        if n < 20:
            print(f"{name:32s}  (only {n} finite samples, skipped)")
            continue

        tau_sim = e * KPS[None, :] - dq * KDS[None, :]
        err = tau - tau_sim
        rmse = np.sqrt(np.mean(err ** 2, axis=0))
        corr = np.array([
            np.corrcoef(tau[:, j], tau_sim[:, j])[0, 1] if tau[:, j].std() > 1e-9 else np.nan
            for j in range(tau.shape[1])
        ])

        worst_rmse_j = int(np.nanargmax(rmse))
        worst_corr_j = int(np.nanargmin(corr))
        row = (f"{name:32s} {n:6d} {rmse.mean():11.4f} "
               f"{rmse[worst_rmse_j]:12.4f} ({JOINT_NAMES[worst_rmse_j]:14s}) "
               f"{corr[worst_corr_j]:12.4f} ({JOINT_NAMES[worst_corr_j]:14s})")
        if jidx is not None:
            row += f"  {rmse[jidx]:16.4f} {corr[jidx]:12.4f}"
        print(row)


if __name__ == "__main__":
    raise SystemExit(main())
