#!/usr/bin/env python
"""Why do waist_roll / waist_pitch violate the PD law?

After excluding the one NaN-corrupted run, 27 of 29 joints obey
`tau = kp(q_target-q) - kd*dq` at corr >= 0.98, but waist_roll (0.42) and
waist_pitch (0.63) do not. Two hypotheses, tested here against each other:

H1  OVER-LIMIT COMMAND. The policy commands +-1.71 / +-1.94 rad against a
    +-0.52 rad URDF limit. Past the stop the joint cannot move, so the PD
    error keeps growing while the real torque is bounded by the stop reaction.
    Prediction: restricting to in-limit frames should restore the correlation.

H2  PARALLEL / DIFFERENTIAL MECHANISM. On G1 the waist roll and pitch axes are
    driven by two motors through a differential, so each *motor* torque is a
    linear combination of the two *joint* PD errors, not a 1:1 map.
    Prediction: regressing tau on BOTH joints' errors jointly should fit far
    better than either alone, with significant off-diagonal terms.

The two are not mutually exclusive; the point is to see which one carries the
variance.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_constants import KD, KP, list_runs, load_run  # noqa: E402

ROLL, PITCH, YAW = 13, 14, 12
LIMIT = 0.52  # rad, from g1_29dof.urdf for both waist_roll and waist_pitch


def gather():
    Q, DQ, QT, TAU = [], [], [], []
    for r in list_runs():
        try:
            d = load_run(r)
        except Exception:  # noqa: BLE001
            continue
        if any(np.isnan(v).any() for v in (d["q"], d["dq"], d["q_target"], d["tau"])):
            continue
        mv = np.abs(d["dq"]).max(axis=1) > 0.05
        if mv.sum() < 200:
            continue
        lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
        Q.append(d["q"][lo:hi]); DQ.append(d["dq"][lo:hi])
        QT.append(d["q_target"][lo:hi]); TAU.append(d["tau"][lo:hi])
    return (np.concatenate(Q), np.concatenate(DQ),
            np.concatenate(QT), np.concatenate(TAU))


def fit_report(name, X, y):
    """Least squares y ~ X, report R^2 and coefficients."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return r2, coef


def main():
    q, dq, qt, tau = gather()
    print(f"pooled {len(q)} frames ({len(q)*0.02:.0f} s), NaN run excluded\n")

    err = qt - q
    for j, nm in ((ROLL, "waist_roll"), (PITCH, "waist_pitch"), (YAW, "waist_yaw")):
        pred = KP[j] * err[:, j] - KD[j] * dq[:, j]
        c = np.corrcoef(tau[:, j], pred)[0, 1]
        over = 100.0 * float((np.abs(qt[:, j]) > LIMIT).mean())
        atlim = 100.0 * float((np.abs(q[:, j]) > 0.95 * LIMIT).mean())
        print(f"{nm:14s} corr={c:6.3f}  cmd beyond +-{LIMIT} rad: {over:5.1f}% "
              f"of frames   |q| at >95% of limit: {atlim:4.1f}%")

    # ---------------- H1: restrict to in-limit commands --------------------
    print("\n--- H1: does restricting to IN-LIMIT commands rescue the fit? ---")
    print(f"{'joint':14s}{'subset':>26}{'n':>9}{'corr':>8}{'slope':>8}")
    for j, nm in ((ROLL, "waist_roll"), (PITCH, "waist_pitch")):
        pred_all = KP[j] * err[:, j] - KD[j] * dq[:, j]
        for label, m in (("all frames", np.ones(len(q), bool)),
                         ("|q_target| < 0.52", np.abs(qt[:, j]) <= LIMIT),
                         ("|q_target| < 0.30", np.abs(qt[:, j]) <= 0.30),
                         ("|q| < 0.4*limit", np.abs(q[:, j]) < 0.4 * LIMIT)):
            if m.sum() < 500:
                continue
            t_, p_ = tau[m, j], pred_all[m]
            c = np.corrcoef(t_, p_)[0, 1]
            s = float(p_ @ t_ / (p_ @ p_))
            print(f"{nm:14s}{label:>26}{m.sum():>9}{c:8.3f}{s:8.3f}")

    # ---------------- H2: differential coupling ----------------------------
    print("\n--- H2: is tau a COMBINATION of both waist axes? ---")
    print("regressing measured tau on [kp*err_roll, kp*err_pitch, kd*dq_roll, kd*dq_pitch]")
    X = np.column_stack([
        KP[ROLL] * err[:, ROLL], KP[PITCH] * err[:, PITCH],
        -KD[ROLL] * dq[:, ROLL], -KD[PITCH] * dq[:, PITCH],
    ])
    print(f"\n{'target':14s}{'R2 single':>11}{'R2 joint':>10}"
          f"{'c_errRoll':>11}{'c_errPitch':>12}{'c_dqRoll':>10}{'c_dqPitch':>11}")
    for j, nm in ((ROLL, "waist_roll"), (PITCH, "waist_pitch")):
        x_single = np.column_stack([KP[j] * err[:, j], -KD[j] * dq[:, j]])
        r2_s, _ = fit_report(nm, x_single, tau[:, j])
        r2_j, coef = fit_report(nm, X, tau[:, j])
        print(f"{nm:14s}{r2_s:11.3f}{r2_j:10.3f}"
              f"{coef[0]:11.3f}{coef[1]:12.3f}{coef[2]:10.3f}{coef[3]:11.3f}")

    print("\n  If the off-diagonal coefficient is comparable to the diagonal one,")
    print("  the two axes are mechanically coupled (H2). If the joint fit barely")
    print("  improves on the single fit, coupling is not the explanation.")

    # sanity: the same test on a joint known to be clean
    x_single = np.column_stack([KP[YAW] * err[:, YAW], -KD[YAW] * dq[:, YAW]])
    r2_s, _ = fit_report("waist_yaw", x_single, tau[:, YAW])
    print(f"\n  control (waist_yaw, single-axis fit): R2 = {r2_s:.3f}")


if __name__ == "__main__":
    main()
