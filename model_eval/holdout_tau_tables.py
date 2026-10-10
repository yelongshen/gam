#!/usr/bin/env python3
"""Hold-out tables: tau_est vs tau_pd, and tau_est vs tau_pd + fitted M6 term.

Split is BY RUN (every 5th clean run is held out). Fitting on train runs only;
every number printed is computed on the held-out runs. Frame-level random splits
would leak, since consecutive frames are near-duplicates.

Table 1  tau_est  vs  tau_pd                      (the un-corrected PD law)
Table 2  tau_est  vs  tau_pd + M6(fitted residual)  (M6 = coulomb + viscous +
         stribeck + load-dependent + position/offset + ddq; see fit_friction_model.py)

Columns
  est RMS / model RMS   RMS of each torque [Nm]
  corr                  Pearson correlation of tau_est with the model torque
  gap RMS               sqrt(mean((tau_est - model)^2))   [Nm]
  bias                  mean(tau_est - model)             [Nm]
  gain                  least-squares slope of tau_est on model
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim2real"))
sys.path.insert(0, os.path.join(ROOT, "model_eval"))

from deploy_constants import JOINT_NAMES  # noqa: E402
from fit_friction_model import design, load_data  # noqa: E402

MODEL = 6
LAM = 1e-3
HOLDOUT_EVERY = 5


def stats(est, mod):
    gap = est - mod
    c = np.corrcoef(est, mod)[0, 1] if est.std() > 1e-9 and mod.std() > 1e-9 else np.nan
    g = float(mod @ est / (mod @ mod)) if (mod @ mod) > 1e-9 else np.nan
    return (float(np.sqrt((est ** 2).mean())), float(np.sqrt((mod ** 2).mean())),
            c, float(np.sqrt((gap ** 2).mean())), float(gap.mean()), g)


def table(title, est, mod, joints):
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'joint':22s}{'tau_est RMS':>12}{'model RMS':>11}{'corr':>8}"
          f"{'gap RMS[Nm]':>13}{'bias[Nm]':>10}{'gain':>7}")
    print("-" * 100)
    rows = []
    for j in joints:
        s = stats(est[j], mod[j])
        rows.append(s)
        print(f"{JOINT_NAMES[j]:22s}{s[0]:12.3f}{s[1]:11.3f}{s[2]:8.3f}"
              f"{s[3]:13.3f}{s[4]:+10.3f}{s[5]:7.3f}")
    arr = np.array(rows, dtype=float)
    print("-" * 100)
    print(f"{'MEAN over joints':22s}{np.nanmean(arr[:,0]):12.3f}{np.nanmean(arr[:,1]):11.3f}"
          f"{np.nanmean(arr[:,2]):8.3f}{np.nanmean(arr[:,3]):13.3f}"
          f"{np.nanmean(arr[:,4]):+10.3f}{np.nanmean(arr[:,5]):7.3f}")
    print()
    return arr


def main():
    data = load_data(60, 5)
    test_idx = [i for i in range(len(data)) if i % HOLDOUT_EVERY == 0]
    train = [d for i, d in enumerate(data) if i not in test_idx]
    test = [d for i, d in enumerate(data) if i in test_idx]
    n_tr = sum(len(d["q"]) for d in train)
    n_te = sum(len(d["q"]) for d in test)
    print(f"runs: {len(data)} clean  ->  {len(train)} train / {len(test)} HELD-OUT")
    print(f"frames: train {n_tr}  held-out {n_te} ({n_te*0.02:.0f} s)")
    print("held-out runs:", ", ".join(d["name"].split('/')[-1] for d in test), "\n")

    joints = list(range(29))
    est = {j: np.concatenate([d["tau"][:, j] for d in test]) for j in joints}
    pd = {j: np.concatenate([d["tau_pd"][:, j] for d in test]) for j in joints}
    fit = {}
    for j in joints:
        X = np.vstack([design(d, j, MODEL) for d in train])
        y = np.concatenate([d["res"][:, j] for d in train])
        coef = np.linalg.solve(X.T @ X + LAM * np.eye(X.shape[1]), X.T @ y)
        Xte = np.vstack([design(d, j, MODEL) for d in test])
        fit[j] = pd[j] + Xte @ coef

    t1 = table("TABLE 1   tau_est (real)  vs  tau_pd   -- HELD-OUT runs", est, pd, joints)
    t2 = table("TABLE 2   tau_est (real)  vs  tau_pd + M6 fit   -- HELD-OUT runs", est, fit, joints)

    print("=" * 100)
    print("CHANGE FROM TABLE 1 TO TABLE 2 (held-out)")
    print("=" * 100)
    print(f"{'joint':22s}{'gap before':>12}{'gap after':>11}{'reduction':>11}"
          f"{'corr before':>13}{'corr after':>12}")
    print("-" * 82)
    worse = 0
    for k, j in enumerate(joints):
        b, a = t1[k, 3], t2[k, 3]
        if a > b:
            worse += 1
        print(f"{JOINT_NAMES[j]:22s}{b:12.3f}{a:11.3f}{100*(b-a)/b:10.1f}%"
              f"{t1[k,2]:13.3f}{t2[k,2]:12.3f}")
    print("-" * 82)
    print(f"{'MEAN gap':22s}{t1[:,3].mean():12.3f}{t2[:,3].mean():11.3f}"
          f"{100*(t1[:,3].mean()-t2[:,3].mean())/t1[:,3].mean():10.1f}%")
    print(f"\njoints where the fit made the held-out gap WORSE: {worse}/29")


if __name__ == "__main__":
    main()
