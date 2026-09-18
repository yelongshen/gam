#!/usr/bin/env python3
"""corr(dq_logged, dq_fd) for ALL 29 joints, on any single run directory or
a session of `g1_deploy_run_*` subdirectories.

This is the all-joint version of the Phase E S6 diagnostic
(`sim2real_phaseE_dq_aliasing.py`), but works mode-agnostically (no Mode-2
filter) so it also says something about idle/no-command logs like
`g1_run_0917` (see the g1_run_0917 idle-sway analysis).

Usage:
    .venv_sim/bin/python model_eval/sim2real_dq_corr_all_joints.py \
        /home/grease/g1_robot_data/g1_run_0917/g1_deploy_run_09182026_092516

    # restrict to a time window (seconds, relative to run start):
    .venv_sim/bin/python model_eval/sim2real_dq_corr_all_joints.py RUN_DIR --t0 8 --t1 16

    # a whole session of g1_deploy_run_* subdirs, pooled per joint:
    .venv_sim/bin/python model_eval/sim2real_dq_corr_all_joints.py \
        /home/grease/g1_robot_data/g1_run_0914 --session
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES  # noqa: E402
from model_eval.sim2real_single_run_quicklook import load  # noqa: E402


def dq_fd_corr(run_dir, t0=None, t1=None):
    """Return (corr[29], std_log[29], std_fd[29]) for one run directory,
    or None if the run has too few usable samples."""
    q, t = load(run_dir, "q")
    dq, _ = load(run_dir, "dq")
    n = min(len(q), len(dq))
    q, dq, t = q[:n], dq[:n], t[:n]

    if t0 is not None or t1 is not None:
        t_s = (t - t[0]) / 1000.0
        m = np.ones(len(t_s), dtype=bool)
        if t0 is not None:
            m &= t_s >= t0
        if t1 is not None:
            m &= t_s < t1
        q, dq, t = q[m], dq[m], t[m]

    if len(q) < 22:
        return None

    dt_s = np.diff(t) / 1000.0
    ok = dt_s > 1e-6
    dq_fd = np.diff(q, axis=0) / np.where(ok, dt_s, np.nan)[:, None]
    a, b = dq[:-1], dq_fd

    corr = np.full(29, np.nan)
    std_log = np.full(29, np.nan)
    std_fd = np.full(29, np.nan)
    for j in range(29):
        g = np.isfinite(a[:, j]) & np.isfinite(b[:, j])
        if g.sum() < 20:
            continue
        aj, bj = a[g, j], b[g, j]
        std_log[j], std_fd[j] = aj.std(), bj.std()
        if std_log[j] < 1e-9 or std_fd[j] < 1e-9:
            continue
        corr[j] = np.corrcoef(aj, bj)[0, 1]
    return corr, std_log, std_fd


def print_table(corr, std_log, std_fd, title):
    order = np.argsort(np.nan_to_num(corr, nan=1.0))  # worst (most negative) first
    print(f"\n=== {title} ===")
    print(f"  {'joint':15s} {'corr':>7s} {'std_log':>8s} {'std_fd':>8s} {'ratio':>7s}")
    for j in order:
        if not np.isfinite(corr[j]):
            continue
        ratio = std_log[j] / max(std_fd[j], 1e-9)
        flag = "  <-- worst" if j == order[0] else ""
        print(f"  {JOINT_NAMES[j]:15s} {corr[j]:7.3f} {std_log[j]:8.3f} "
              f"{std_fd[j]:8.3f} {ratio:7.2f}{flag}")
    n_neg = int(np.sum(corr < 0))
    n_lt05 = int(np.sum(corr < 0.5))
    print(f"  -> {n_neg}/29 joints negative, {n_lt05}/29 below 0.5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--t0", type=float, default=None, help="window start, seconds")
    ap.add_argument("--t1", type=float, default=None, help="window end, seconds")
    ap.add_argument("--session", action="store_true",
                     help="run_dir is a session containing g1_deploy_run_* subdirs; "
                          "report each sub-run and a pooled-by-concat summary")
    args = ap.parse_args()

    if args.session:
        subdirs = sorted(glob.glob(os.path.join(args.run_dir, "g1_deploy_run_*")))
        if not subdirs:
            subdirs = [args.run_dir]
        corrs = []
        for d in subdirs:
            out = dq_fd_corr(d, args.t0, args.t1)
            if out is None:
                print(f"\n=== {os.path.basename(d)} === (too few samples, skipped)")
                continue
            corr, std_log, std_fd = out
            print_table(corr, std_log, std_fd, os.path.basename(d))
            corrs.append(corr)
        if corrs:
            allc = np.stack(corrs)
            print("\n=== summary across sub-runs (min / median / max corr per joint) ===")
            order = np.argsort(np.nanmin(allc, axis=0))
            print(f"  {'joint':15s} {'min':>7s} {'median':>7s} {'max':>7s} {'#<0':>5s} {'#<0.5':>6s}")
            for j in order:
                col = allc[:, j]
                if np.all(np.isnan(col)):
                    continue
                print(f"  {JOINT_NAMES[j]:15s} {np.nanmin(col):7.3f} "
                      f"{np.nanmedian(col):7.3f} {np.nanmax(col):7.3f} "
                      f"{int(np.nansum(col < 0)):5d} {int(np.nansum(col < 0.5)):6d}")
    else:
        meta_path = os.path.join(args.run_dir, "metadata.json")
        title = os.path.basename(args.run_dir.rstrip("/"))
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            title += f" ({meta['robot_config']['model_path']})"
        out = dq_fd_corr(args.run_dir, args.t0, args.t1)
        if out is None:
            print("too few usable samples in this window/run")
            return 1
        print_table(*out, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
