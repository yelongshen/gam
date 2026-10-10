#!/usr/bin/env python3
"""Kp/Kd free-fit (tau ~= Kp*e - Kd*dq), ALL 29 joints, pooled across a set
of run dirs. Same method as sim2real_phaseE_gain_refit_per_run.py /
sim2real_single_run_quicklook.py's fit section, just: (a) all joints instead
of a shortlist, and (b) works on any run-dir list, not just g1_robot_data
session globs -- so it works directly on g1_run_0918's non-standard layout.

Usage:
    .venv_sim/bin/python model_eval/sim2real_gain_refit_all_joints.py \
        --run-dirs RUN_DIR [RUN_DIR ...] [--all-samples]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES, KDS, KPS, action_to_q_target  # noqa: E402
from model_eval.sim2real_single_run_quicklook import load  # noqa: E402


def load_run(run_dir, only_mode2):
    q, t = load(run_dir, "q")
    dq, _ = load(run_dir, "dq")
    act, _ = load(run_dir, "action")
    tau, _ = load(run_dir, "motor_torque")
    mode, _ = load(run_dir, "encoder_mode", ncols=1)
    n = min(len(q), len(dq), len(act), len(tau), len(mode))
    q, dq, act, tau, mode = q[:n], dq[:n], act[:n], tau[:n], mode[:n, 0]

    qt = action_to_q_target(act)
    e = qt - q
    finite = (np.all(np.isfinite(e), 1) & np.all(np.isfinite(dq), 1) &
              np.all(np.isfinite(tau), 1))
    if only_mode2:
        finite &= (mode == 2)
    if finite.sum() < 20:
        return None
    return e[finite], dq[finite], tau[finite]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--all-samples", action="store_true",
                     help="use all samples, not just encoder_mode==2")
    args = ap.parse_args()

    e_list, dq_list, tau_list = [], [], []
    for rd in args.run_dirs:
        out = load_run(rd, only_mode2=not args.all_samples)
        if out is None:
            print(f"[skip] {os.path.basename(rd)}: too few usable samples")
            continue
        e, dq, tau = out
        print(f"[ok]   {os.path.basename(rd)}: {len(e)} samples")
        e_list.append(e); dq_list.append(dq); tau_list.append(tau)

    if not e_list:
        print("no usable data")
        return 1

    e = np.concatenate(e_list)
    dq = np.concatenate(dq_list)
    tau = np.concatenate(tau_list)
    print(f"\npooled: {len(e)} samples across {len(e_list)} runs\n")

    print(f"{'joint':15s} {'Kp_fit':>8s} {'Kp/nom':>7s} {'Kd_fit':>8s} {'Kd/nom':>7s} "
          f"{'R2':>6s} {'RMSEnom':>8s} {'RMSEfit':>8s}")
    rows = []
    for j in range(29):
        A = np.stack([e[:, j], -dq[:, j]], axis=1)
        t_ = tau[:, j]
        coef, *_ = np.linalg.lstsq(A, t_, rcond=None)
        resid = t_ - A @ coef
        ss = ((t_ - t_.mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / ss if ss > 0 else np.nan
        tau_nom = e[:, j] * KPS[j] - dq[:, j] * KDS[j]
        rmse_nom = np.sqrt(np.mean((t_ - tau_nom) ** 2))
        rmse_fit = np.sqrt(np.mean(resid ** 2))
        rows.append((j, coef[0], coef[0] / KPS[j], coef[1], coef[1] / KDS[j], r2, rmse_nom, rmse_fit))

    for j, kp, kp_r, kd, kd_r, r2, rn, rf in rows:
        print(f"{JOINT_NAMES[j]:15s} {kp:8.3f} {kp_r:7.3f} {kd:8.4f} {kd_r:7.3f} "
              f"{r2:6.3f} {rn:8.3f} {rf:8.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
