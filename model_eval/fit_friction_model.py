#!/usr/bin/env python3
"""Fit a friction / damping correction model to the real torque residual.

THE GAP
-------
`verify_pd_law.py` established that the commanded PD law

    tau_pd = kp (q_target - q) - kd * dq

reproduces the measured `tau_est` at corr 0.998 with a 6.3% residual. That
residual is what a simulator does not model. This script characterises it per
joint and then tries to FIT it, so the fitted term can be added to the sim.

    residual = tau_est - tau_pd          [Nm]

CANDIDATE MODELS (nested, so each column shows what the extra term buys)
------------------------------------------------------------------------
  M1 coulomb            a*sign(dq)
  M2 + viscous          a*sign(dq) + b*dq
  M3 + stribeck         .. + c*exp(-|dq|/vs)*sign(dq)     (stiction at low speed)
  M4 + load-dependent   .. + d*|tau_pd|*sign(dq)          (friction grows with load)
  M5 + position         .. + e*q + f                      (gravity/calibration offset)
  M6 + accel            .. + g*ddq                        (missing inertia/armature)

An earlier attempt fitted only M2 and got R^2 = 0.02-0.05, which was reported
as "friction does not explain the residual". That conclusion was premature: M2
is the weakest model in this list, and it was fitted per-joint on pooled data
with no cross-validation.

VALIDATION
----------
Leave-one-RUN-out. Splitting randomly would leak: consecutive frames inside a
run are near-duplicates, so a random split lets the model memorise the run and
report an R^2 that does not transfer. Every number below is on held-out runs.

Usage:
  .venv_sim/bin/python model_eval/fit_friction_model.py
  .venv_sim/bin/python model_eval/fit_friction_model.py --joints 4 10 18
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim2real"))

from deploy_constants import (  # noqa: E402
    CONTROL_DT,
    EFFORT_LIMIT,
    JOINT_NAMES,
    KD,
    KP,
    list_runs,
    load_run,
)

V_STRIBECK = 0.1  # rad/s, scale of the stiction decay


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.stack([np.convolve(x[:, j], k, mode="same")
                     for j in range(x.shape[1])], axis=1)


def load_data(max_runs, smooth_win=5):
    """Per-run arrays of (q, dq, ddq, tau_pd, residual)."""
    out = []
    for r in list_runs()[:max_runs]:
        try:
            d = load_run(r)
        except Exception:  # noqa: BLE001
            continue
        q, dq, qt, tau = d["q"], d["dq"], d["q_target"], d["tau"]
        if any(np.isnan(v).any() for v in (q, dq, qt, tau)):
            continue
        mv = np.abs(dq).max(axis=1) > 0.05
        if mv.sum() < 300:
            continue
        lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
        q, dq, qt, tau = q[lo:hi], dq[lo:hi], qt[lo:hi], tau[lo:hi]
        ddq = np.gradient(smooth(dq, smooth_win), CONTROL_DT, axis=0)
        tau_pd = KP * (qt - q) - KD * dq
        out.append(dict(name=os.path.relpath(r, "/home/grease/g1_robot_data"),
                        q=q, dq=dq, ddq=ddq, tau_pd=tau_pd,
                        tau=tau, res=tau - tau_pd))
    return out


def design(d, j, model):
    """Feature matrix for joint j under the named model."""
    dq = d["dq"][:, j]
    s = np.sign(dq)
    cols = [s]                                    # M1 coulomb
    if model >= 2:
        cols.append(dq)                           # viscous
    if model >= 3:
        cols.append(np.exp(-np.abs(dq) / V_STRIBECK) * s)   # stribeck
    if model >= 4:
        cols.append(np.abs(d["tau_pd"][:, j]) * s)          # load-dependent
    if model >= 5:
        cols += [d["q"][:, j], np.ones(len(dq))]            # pos + offset
    if model >= 6:
        cols.append(d["ddq"][:, j])                         # inertia error
    return np.column_stack(cols)


def loro_r2(data, j, model, lam=1e-3):
    """Leave-one-run-out R^2 for joint j, plus the pooled coefficients."""
    r2s = []
    for held in range(len(data)):
        tr = [d for i, d in enumerate(data) if i != held]
        te = data[held]
        X = np.vstack([design(d, j, model) for d in tr])
        y = np.concatenate([d["res"][:, j] for d in tr])
        A = X.T @ X + lam * np.eye(X.shape[1])
        coef = np.linalg.solve(A, X.T @ y)
        Xte, yte = design(te, j, model), te["res"][:, j]
        pred = Xte @ coef
        ss_res = ((yte - pred) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum()
        r2s.append(1 - ss_res / max(ss_tot, 1e-12))
    # coefficients fitted on everything, for reporting
    X = np.vstack([design(d, j, model) for d in data])
    y = np.concatenate([d["res"][:, j] for d in data])
    coef = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    return np.asarray(r2s), coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runs", type=int, default=30)
    ap.add_argument("--joints", type=int, nargs="*", default=None)
    ap.add_argument("--smooth", type=int, default=5)
    a = ap.parse_args()

    data = load_data(a.max_runs, a.smooth)
    n = sum(len(d["q"]) for d in data)
    print(f"runs {len(data)}   frames {n}  ({n*CONTROL_DT:.0f} s)\n")

    joints = a.joints if a.joints else list(range(29))

    # ---------------------------------------------------------- the gap
    print("=" * 96)
    print("1. THE GAP PER JOINT:  residual = tau_est - tau_pd   [Nm]")
    print("=" * 96)
    print(f"{'joint':22s}{'tau_est RMS':>12}{'resid RMS':>11}{'resid p95':>11}"
          f"{'resid/tau':>11}{'% effort':>10}{'corr(res,dq)':>14}")
    print("-" * 96)
    RES = {}
    for j in joints:
        res = np.concatenate([d["res"][:, j] for d in data])
        tau = np.concatenate([d["tau"][:, j] for d in data])
        dq = np.concatenate([d["dq"][:, j] for d in data])
        RES[j] = (res, tau, dq)
        c = np.corrcoef(res, dq)[0, 1] if dq.std() > 1e-9 else np.nan
        print(f"{JOINT_NAMES[j]:22s}{np.sqrt((tau**2).mean()):12.3f}"
              f"{np.sqrt((res**2).mean()):11.3f}"
              f"{np.percentile(np.abs(res),95):11.3f}"
              f"{np.sqrt((res**2).mean())/max(np.sqrt((tau**2).mean()),1e-9):11.3f}"
              f"{100*np.sqrt((res**2).mean())/EFFORT_LIMIT[j]:10.2f}{c:14.3f}")

    # ------------------------------------------------- nested model fits
    print("\n" + "=" * 96)
    print("2. FITTING THE RESIDUAL — leave-one-run-out R^2 (higher is better)")
    print("=" * 96)
    print("   M1 coulomb | M2 +viscous | M3 +stribeck | M4 +load | M5 +pos/offset | M6 +ddq")
    print(f"\n{'joint':22s}{'M1':>8}{'M2':>8}{'M3':>8}{'M4':>8}{'M5':>8}{'M6':>8}"
          f"{'best':>7}{'resid after':>13}")
    print("-" * 96)
    summary = {}
    for j in joints:
        row, best_r2, best_m = [], -9e9, 0
        for m in (1, 2, 3, 4, 5, 6):
            r2s, _ = loro_r2(data, j, m)
            med = float(np.median(r2s))
            row.append(med)
            if med > best_r2:
                best_r2, best_m = med, m
        res = RES[j][0]
        after = np.sqrt((res ** 2).mean()) * np.sqrt(max(1 - best_r2, 0.0))
        summary[j] = (best_m, best_r2, after)
        print(f"{JOINT_NAMES[j]:22s}" + "".join(f"{v:8.3f}" for v in row)
              + f"{best_m:7d}{after:13.3f}")

    # ------------------------------------------------------------ verdict
    print("\n" + "=" * 96)
    print("3. VERDICT")
    print("=" * 96)
    good = [j for j in joints if summary[j][1] > 0.5]
    part = [j for j in joints if 0.2 < summary[j][1] <= 0.5]
    poor = [j for j in joints if summary[j][1] <= 0.2]
    print(f"  R^2 > 0.5 (model explains most of the gap): {len(good)}/{len(joints)}")
    for j in good:
        m, r2, after = summary[j]
        print(f"      {JOINT_NAMES[j]:22s} M{m}  R2 {r2:.3f}  "
              f"resid {np.sqrt((RES[j][0]**2).mean()):.3f} -> {after:.3f} Nm")
    print(f"  0.2 < R^2 <= 0.5 (partial): {len(part)}/{len(joints)}")
    for j in part:
        print(f"      {JOINT_NAMES[j]:22s} M{summary[j][0]}  R2 {summary[j][1]:.3f}")
    print(f"  R^2 <= 0.2 (not explained): {len(poor)}/{len(joints)}")

    if good:
        print("\n  Fitted coefficients for the joints that DO fit "
              "(pooled over all runs):")
        names = ["coulomb[Nm]", "viscous[Nms/rad]", "stribeck[Nm]",
                 "load_coef[-]", "pos[Nm/rad]", "offset[Nm]", "inertia[kg m^2]"]
        for j in good:
            m = summary[j][0]
            _, coef = loro_r2(data, j, m)
            terms = " ".join(f"{names[k]}={coef[k]:+.4f}"
                             for k in range(len(coef)))
            print(f"      {JOINT_NAMES[j]:22s} M{m}: {terms}")

    print("\n  HOW TO USE: add the fitted term as an extra joint torque in sim,")
    print("  or map coulomb -> joint friction and viscous -> joint damping in")
    print("  the actuator config. Re-run")
    print("      model_eval/sim2real_phaseC1b_onestep_qdq.py --base float")
    print("  and check whether ddq correlation rises above today's 0.46-0.49.")

    print("\n  CAVEAT: residual and dq are both measured, so a fit can absorb")
    print("  sensor noise correlated between them. M6 in particular can soak up")
    print("  ddq differentiation error rather than real inertia mismatch --")
    print("  prefer the simplest model within ~0.05 R^2 of the best.")


if __name__ == "__main__":
    main()
