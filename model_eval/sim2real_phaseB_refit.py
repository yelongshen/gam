#!/usr/bin/env python3
"""Re-run the Phase B `Kp_eff`/`Kd_eff` free-fit (`sim2real/phaseB_actuator.md`
section 3b) against NEW deploy sessions, and compare against the nominal
(commanded) gains from `data_process/g1_params.py`.

Fits, per joint, over ALL Mode-2 samples pooled across every run directory in
a session:

    tau_est = Kp_eff * (q_target - q) - Kd_eff * dq + offset

exactly the recipe used for the `aug11` session in phaseB_actuator.md sec 3b.

Usage:
    .venv_sim/bin/python model_eval/sim2real_phaseB_refit.py g1_run_0905 g1_run_0908
    .venv_sim/bin/python model_eval/sim2real_phaseB_refit.py g1_run_0905 g1_run_0908 --csv out.csv
"""
import argparse
import csv as csv_mod
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import (  # noqa: E402
    JOINT_NAMES, KDS, KPS, action_to_q_target,
)

ROOT = "/home/grease/g1_robot_data"
_TCOL, _DATA0 = 2, 5


def _read_csv(path, ncol):
    t, v = [], []
    with open(path) as f:
        r = csv_mod.reader(f)
        next(r)
        for row in r:
            if len(row) < _DATA0 + ncol:
                continue
            try:
                ts = float(row[_TCOL]) / 1000.0
                vals = [float(x) for x in row[_DATA0:_DATA0 + ncol]]
            except ValueError:
                continue
            t.append(ts)
            v.append(vals)
    return np.asarray(t), np.asarray(v)


def load_run(run_dir):
    """Load one g1_deploy_run_* directory, mode-2 rows only.
    Returns (e, dq, tau) each (T,29), or None if unusable."""
    needed = ["q", "dq", "action", "motor_torque", "encoder_mode"]
    paths = {n: os.path.join(run_dir, n + ".csv") for n in needed}
    if not all(os.path.exists(p) for p in paths.values()):
        return None

    t0, q = _read_csv(paths["q"], 29)
    t1, dq = _read_csv(paths["dq"], 29)
    t2, action = _read_csv(paths["action"], 29)
    t3, tau = _read_csv(paths["motor_torque"], 29)
    t4, em = _read_csv(paths["encoder_mode"], 1)

    n = min(len(t0), len(t1), len(t2), len(t3), len(t4))
    if n < 20:
        return None
    q, dq, action, tau, em = q[:n], dq[:n], action[:n], tau[:n], em[:n, 0]

    mask = em == 2
    if mask.sum() < 20:
        return None

    q_target = action_to_q_target(action)
    e = q_target - q
    return e[mask], dq[mask], tau[mask]


def fit_session(session_name):
    session_dir = os.path.join(ROOT, session_name)
    run_dirs = sorted(glob.glob(os.path.join(session_dir, "g1_deploy_run_*")))
    if not run_dirs:
        # session_name may itself be a single run directory
        run_dirs = [session_dir] if os.path.isdir(session_dir) else []

def fit_session_raw(session_name):
    """Load + pool Mode-2 (e, dq, tau) across all runs in a session.
    Returns (E, DQ, TAU, used) or (None, None, None, used) if unusable."""
    session_dir = os.path.join(ROOT, session_name)
    run_dirs = sorted(glob.glob(os.path.join(session_dir, "g1_deploy_run_*")))
    if not run_dirs:
        run_dirs = [session_dir] if os.path.isdir(session_dir) else []

    all_e, all_dq, all_tau = [], [], []
    used = []
    for rd in run_dirs:
        out = load_run(rd)
        if out is None:
            continue
        e, dq, tau = out
        all_e.append(e)
        all_dq.append(dq)
        all_tau.append(tau)
        used.append((os.path.basename(rd), len(e)))

    if not all_e:
        return None, None, None, used

    E = np.concatenate(all_e, axis=0)
    DQ = np.concatenate(all_dq, axis=0)
    TAU = np.concatenate(all_tau, axis=0)

    finite = np.all(np.isfinite(E), axis=1) & np.all(np.isfinite(DQ), axis=1) \
        & np.all(np.isfinite(TAU), axis=1)
    E, DQ, TAU = E[finite], DQ[finite], TAU[finite]
    return E, DQ, TAU, used


def fit_session(session_name):
    E, DQ, TAU, used = fit_session_raw(session_name)
    if E is None:
        return None, used

    n_joints = E.shape[1]
    kp_fit = np.zeros(n_joints)
    kd_fit = np.zeros(n_joints)
    r2 = np.zeros(n_joints)
    n_samples = E.shape[0]

    for j in range(n_joints):
        # tau = Kp*e - Kd*dq + c  ==  A @ [Kp, -Kd, c] ... solve as
        # tau = Kp*e + Kd'*dq + c   with Kd' = -Kd_eff
        A = np.stack([E[:, j], DQ[:, j], np.ones(n_samples)], axis=1)
        coef, *_ = np.linalg.lstsq(A, TAU[:, j], rcond=None)
        kp_j, kdp_j, _c = coef
        kd_j = -kdp_j
        pred = A @ coef
        ss_res = np.sum((TAU[:, j] - pred) ** 2)
        ss_tot = np.sum((TAU[:, j] - TAU[:, j].mean()) ** 2)
        r2_j = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        kp_fit[j] = kp_j
        kd_fit[j] = kd_j
        r2[j] = r2_j

    return dict(kp_fit=kp_fit, kd_fit=kd_fit, r2=r2, n=n_samples), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="e.g. g1_run_0905 g1_run_0908")
    ap.add_argument("--csv", default=None, help="optional path to write a CSV")
    args = ap.parse_args()

    results = {}
    for s in args.sessions:
        fit, used = fit_session(s)
        if fit is None:
            print(f"[{s}] no usable runs found under {os.path.join(ROOT, s)}")
            continue
        print(f"\n=== {s} ===  ({fit['n']} Mode-2 samples pooled from "
              f"{len(used)} runs)")
        for name, n in used:
            print(f"    {name}: {n} samples")
        results[s] = fit

    if not results:
        return 1

    header = ["joint", "Kp_nom", "Kd_nom"]
    for s in results:
        header += [f"{s}_Kp_fit", f"{s}_Kp_ratio", f"{s}_Kd_fit",
                   f"{s}_Kd_ratio", f"{s}_R2"]

    rows = []
    print("\n" + "  ".join(f"{h:>16s}" for h in header))
    for i, name in enumerate(JOINT_NAMES):
        row = [name, f"{KPS[i]:.4f}", f"{KDS[i]:.4f}"]
        for s, fit in results.items():
            kp_r = fit["kp_fit"][i] / KPS[i]
            kd_r = fit["kd_fit"][i] / KDS[i]
            row += [f"{fit['kp_fit'][i]:.4f}", f"{kp_r:.3f}",
                    f"{fit['kd_fit'][i]:.4f}", f"{kd_r:.3f}",
                    f"{fit['r2'][i]:.4f}"]
        rows.append(row)
        print("  ".join(f"{v:>16s}" for v in row))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv_mod.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"\nWrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
