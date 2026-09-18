#!/usr/bin/env python3
"""Per-RUN free-fit of (Kp_eff, Kd_eff) for a single joint, and the resulting
residual, compared against the nominal-gain residual.

Motivation (see sim2real/phaseE_waist_roll_chatter.md §14): §2/§3 of that document
rank runs by `tau_est` vs. *nominal*-gain `tau_sim` RMSE. But §1 already established
that the roll subgroup needs Kd_eff ~= 0.5x nominal, so a nominal-gain residual
conflates two different things:

  (a) the known, session-invariant "2x 5020" gain-model error, and
  (b) genuine chatter / aliasing episodes.

Re-fitting (Kp, Kd) per run separates them: a run whose residual collapses once it
gets its own best-fit gains was only ever showing (a); a run that stays high after
refitting has a real (b).

Usage:
    .venv_sim/bin/python model_eval/sim2real_phaseE_gain_refit_per_run.py \
        --sessions g1_run_0905 g1_run_0908 g1_run_0914 --joint waist_roll
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES, KDS, KPS  # noqa: E402
from model_eval.sim2real_phaseB_refit import ROOT, load_run  # noqa: E402

MIN_SAMPLES = 50


def fit_run(e, dq, tau):
    """Least-squares fit of tau ~ kp*e - kd*dq. Returns (kp, kd, r2)."""
    A = np.stack([e, -dq], axis=1)
    coef, *_ = np.linalg.lstsq(A, tau, rcond=None)
    resid = tau - A @ coef
    ss_tot = ((tau - tau.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return coef[0], coef[1], r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--joint", default="waist_roll")
    args = ap.parse_args()

    j = JOINT_NAMES.index(args.joint)
    kp_nom, kd_nom = KPS[j], KDS[j]

    for sess in args.sessions:
        run_dirs = sorted(glob.glob(os.path.join(ROOT, sess, "g1_deploy_run_*")))
        if not run_dirs:
            print(f"[{sess}] no runs found under {os.path.join(ROOT, sess)}")
            continue

        print(f"\n=== {sess} — {args.joint} "
              f"(nominal Kp={kp_nom:.3f}, Kd={kd_nom:.4f}) ===")
        print(f"{'run':34s} {'n':>6s} {'Kp_fit':>8s} {'Kd_fit':>8s} "
              f"{'Kp/nom':>7s} {'Kd/nom':>7s} {'R2':>6s} "
              f"{'RMSE_nom':>9s} {'RMSE_fit':>9s} {'|dq|p99':>8s}")

        pooled = []
        for rd in run_dirs:
            out = load_run(rd)
            if out is None:
                continue
            e, dq, tau = out
            m = (np.isfinite(e[:, j]) & np.isfinite(dq[:, j]) & np.isfinite(tau[:, j]))
            e_, d_, t_ = e[m, j], dq[m, j], tau[m, j]
            if len(e_) < MIN_SAMPLES:
                continue
            pooled.append((e_, d_, t_))

            kp, kd, r2 = fit_run(e_, d_, t_)
            tau_nom = e_ * kp_nom - d_ * kd_nom
            tau_fit = e_ * kp - d_ * kd
            print(f"{os.path.basename(rd):34s} {len(e_):6d} {kp:8.3f} {kd:8.4f} "
                  f"{kp / kp_nom:7.3f} {kd / kd_nom:7.3f} {r2:6.3f} "
                  f"{np.sqrt(((t_ - tau_nom) ** 2).mean()):9.3f} "
                  f"{np.sqrt(((t_ - tau_fit) ** 2).mean()):9.3f} "
                  f"{np.percentile(np.abs(d_), 99):8.2f}")

        if pooled:
            e_ = np.concatenate([p[0] for p in pooled])
            d_ = np.concatenate([p[1] for p in pooled])
            t_ = np.concatenate([p[2] for p in pooled])
            kp, kd, r2 = fit_run(e_, d_, t_)
            tau_nom = e_ * kp_nom - d_ * kd_nom
            tau_fit = e_ * kp - d_ * kd
            print(f"{'POOLED':34s} {len(e_):6d} {kp:8.3f} {kd:8.4f} "
                  f"{kp / kp_nom:7.3f} {kd / kd_nom:7.3f} {r2:6.3f} "
                  f"{np.sqrt(((t_ - tau_nom) ** 2).mean()):9.3f} "
                  f"{np.sqrt(((t_ - tau_fit) ** 2).mean()):9.3f} "
                  f"{np.percentile(np.abs(d_), 99):8.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
