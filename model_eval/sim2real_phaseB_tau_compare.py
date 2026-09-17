#!/usr/bin/env python3
"""MSE / RMSE / correlation of tau_est (real, logged) vs tau_sim (nominal PD
law: Kp_nom*(q_target-q) - Kd_nom*dq) for every joint, pooled across all runs
in a session. This is the BASELINE (no correction) comparison referenced in
`sim2real/phaseB_actuator.md` sec 1 / `sim2real/phaseC1_damping_scan.md` sec 5.

Usage:
    .venv_sim/bin/python model_eval/sim2real_phaseB_tau_compare.py g1_run_0905 g1_run_0908
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES, KDS, KPS  # noqa: E402
from model_eval.sim2real_phaseB_refit import ROOT, fit_session_raw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    args = ap.parse_args()

    for s in args.sessions:
        E, DQ, TAU, used = fit_session_raw(s)
        if E is None:
            print(f"[{s}] no usable runs found under {os.path.join(ROOT, s)}")
            continue

        tau_sim = E * KPS[None, :] - DQ * KDS[None, :]
        err = TAU - tau_sim
        mse = np.mean(err ** 2, axis=0)
        rmse = np.sqrt(mse)
        corr = np.array([
            np.corrcoef(TAU[:, j], tau_sim[:, j])[0, 1] for j in range(TAU.shape[1])
        ])

        print(f"\n=== {s} ===  ({TAU.shape[0]} Mode-2 samples pooled from "
              f"{len(used)} runs)")
        print(f"{'joint':16s} {'MSE':>10s} {'RMSE':>10s} {'corr':>8s}")
        for j, name in enumerate(JOINT_NAMES):
            print(f"{name:16s} {mse[j]:10.4f} {rmse[j]:10.4f} {corr[j]:8.4f}")
        print(f"{'ALL-29 MEAN':16s} {mse.mean():10.4f} {rmse.mean():10.4f} "
              f"{np.nanmean(corr):8.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
