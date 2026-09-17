#!/usr/bin/env python3
"""Diagnostics for the `waist_roll` velocity-aliasing / chatter phenomenon
documented in `sim2real/phaseE_waist_roll_chatter.md`.

Compares the firmware-logged `dq` ("dq_logged"/"dq_est") against an
independent velocity estimate derived purely from the logged `q` via a
forward finite difference ("dq_fd" = (q[i+1]-q[i])/dt), for every joint,
restricted to Mode-2 samples. A strong positive correlation between the two
is the "healthy" signature every joint should show; a correlation dropping
toward zero or negative is the aliasing/chatter signature seen on
`waist_roll`.

Usage:
    # Per-joint table, pooled per session (one column per session/date):
    .venv_sim/bin/python model_eval/sim2real_phaseE_dq_aliasing.py \
        --sessions aug11 g1_run_0905 g1_run_0908 g1_run_0914

    # Split one session into "good" vs "bad" run subsets:
    .venv_sim/bin/python model_eval/sim2real_phaseE_dq_aliasing.py \
        --sessions g1_run_0914 --split-bad-runs run8 run20

    # Per-run breakdown (not pooled) for one session:
    .venv_sim/bin/python model_eval/sim2real_phaseE_dq_aliasing.py \
        --sessions g1_run_0914 --per-run
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES  # noqa: E402
from model_eval.sim2real_phaseB_refit import ROOT, _read_csv  # noqa: E402

AUG11_DIR = "/home/grease/g1_robot_data/g1_real_deploy_logs"


def session_run_dirs(session):
    if session == "aug11":
        return [AUG11_DIR]
    session_dir = os.path.join(ROOT, session)
    dirs = sorted(glob.glob(os.path.join(session_dir, "g1_deploy_run_*")))
    return dirs if dirs else ([session_dir] if os.path.isdir(session_dir) else [])


def load_qdq(run_dir):
    """Return (t, q, dq, em) or None if unusable (missing files / too short)."""
    paths = {n: os.path.join(run_dir, n + ".csv") for n in ("q", "dq", "encoder_mode")}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    t, q = _read_csv(paths["q"], 29)
    _, dq = _read_csv(paths["dq"], 29)
    te, em = _read_csv(paths["encoder_mode"], 1)
    n = min(len(t), len(te))
    if n < 25:
        return None
    t, q, dq, em = t[:n], q[:n], dq[:n], em[:n, 0]
    if (em == 2).sum() < 20:
        return None
    return t, q, dq, em


def pooled_corr_all_joints(run_dirs):
    """corr(dq_logged, dq_fd) per joint, pooled across all Mode-2 samples in
    every run dir given."""
    all_dl = [[] for _ in range(29)]
    all_df = [[] for _ in range(29)]
    for rd in run_dirs:
        out = load_qdq(rd)
        if out is None:
            continue
        t, q, dq, em = out
        dt = np.diff(t)
        mask2 = em[:-1] == 2
        for j in range(29):
            dq_fd = np.diff(q[:, j]) / dt
            all_dl[j].append(dq[:-1, j][mask2])
            all_df[j].append(dq_fd[mask2])

    corrs = np.full(29, np.nan)
    for j in range(29):
        if not all_dl[j]:
            continue
        dl = np.concatenate(all_dl[j])
        df = np.concatenate(all_df[j])
        if df.std() > 1e-9:
            corrs[j] = np.corrcoef(dl, df)[0, 1]
    return corrs


def per_run_corr_one_joint(run_dir, jidx):
    out = load_qdq(run_dir)
    if out is None:
        return None
    t, q, dq, em = out
    dt = np.diff(t)
    mask2 = em[:-1] == 2
    dq_fd = np.diff(q[:, jidx]) / dt
    dl, df = dq[:-1, jidx][mask2], dq_fd[mask2]
    if df.std() < 1e-9:
        return None
    return np.corrcoef(dl, df)[0, 1], dl.std(), df.std(), len(dl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True,
                     help="'aug11', 'g1_run_0905', 'g1_run_0908', 'g1_run_0914', ...")
    ap.add_argument("--joint", default="waist_roll")
    ap.add_argument("--per-run", action="store_true",
                     help="print one row per run dir instead of pooling")
    ap.add_argument("--split-bad-runs", nargs="*", default=None,
                     help="run name substrings (e.g. run8 run20) to report separately "
                          "from the rest of the session")
    args = ap.parse_args()

    jidx = JOINT_NAMES.index(args.joint)

    if args.per_run:
        for s in args.sessions:
            print(f"=== {s}: per-run corr(dq_logged, dq_fd) for {args.joint} ===")
            for rd in session_run_dirs(s):
                res = per_run_corr_one_joint(rd, jidx)
                name = os.path.basename(rd)
                if res is None:
                    print(f"  {name:40s}  (unusable)")
                    continue
                corr, sl, sf, n = res
                print(f"  {name:40s}  n={n:5d}  corr={corr:7.3f}  "
                      f"std_logged={sl:7.3f}  std_fd={sf:6.3f}  ratio={sl/sf:5.2f}")
            print()
        return 0

    if args.split_bad_runs is not None:
        for s in args.sessions:
            all_runs = session_run_dirs(s)
            bad = [r for r in all_runs if any(b in os.path.basename(r) for b in args.split_bad_runs)]
            good = [r for r in all_runs if r not in bad]
            c_good = pooled_corr_all_joints(good)
            c_bad = pooled_corr_all_joints(bad)
            c_all = pooled_corr_all_joints(all_runs)
            print(f"=== {s}: good ({len(good)} runs) vs bad ({len(bad)} runs: {args.split_bad_runs}) ===")
            print(f"{'joint':16s} {'good':>10s} {'bad':>10s} {'all':>10s}")
            for j in np.argsort(c_good):
                print(f"{JOINT_NAMES[j]:16s} {c_good[j]:10.3f} {c_bad[j]:10.3f} {c_all[j]:10.3f}")
            print()
        return 0

    # default: one pooled column per session, all 29 joints, sorted worst-first
    results = {s: pooled_corr_all_joints(session_run_dirs(s)) for s in args.sessions}
    header = f"{'joint':16s} " + " ".join(f"{s:>20s}" for s in args.sessions)
    print(header)
    worst = np.nanmin(np.stack([results[s] for s in args.sessions]), axis=0)
    for j in np.argsort(worst):
        row = f"{JOINT_NAMES[j]:16s} " + " ".join(f"{results[s][j]:20.3f}" for s in args.sessions)
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
