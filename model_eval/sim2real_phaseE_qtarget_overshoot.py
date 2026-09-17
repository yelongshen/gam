#!/usr/bin/env python3
"""For every joint and every session/policy, what fraction of Mode-2 samples have
a reconstructed `q_target` beyond that joint's physical `range` in `g1_29dof.xml`?

Answers two questions from `sim2real/phaseE_waist_roll_chatter.md` sec 12-13:
  1. Is out-of-range q_target unique to `sonic_no_vr_ll_062k`, or does it also
     happen under `policy/low_latency` (the `aug11` session)?
  2. Is it unique to `waist_roll`, or do other joints also get commanded beyond
     their physical range?

Usage:
    .venv_sim/bin/python model_eval/sim2real_phaseE_qtarget_overshoot.py \
        --sessions aug11 g1_run_0905 g1_run_0908 g1_run_0914
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_joint_ranges import JOINT_RANGES  # noqa: E402
from data_process.g1_params import JOINT_NAMES, action_to_q_target  # noqa: E402
from model_eval.sim2real_phaseB_refit import ROOT, _read_csv  # noqa: E402

AUG11_DIR = "/home/grease/g1_robot_data/g1_real_deploy_logs"
LO = np.array([JOINT_RANGES[n][0] for n in JOINT_NAMES])
HI = np.array([JOINT_RANGES[n][1] for n in JOINT_NAMES])


def session_run_dirs(session):
    if session == "aug11":
        return [AUG11_DIR]
    session_dir = os.path.join(ROOT, session)
    dirs = sorted(glob.glob(os.path.join(session_dir, "g1_deploy_run_*")))
    return dirs if dirs else ([session_dir] if os.path.isdir(session_dir) else [])


def overshoot_stats(run_dirs):
    """Returns (frac_over[29], max_overshoot_mag[29], max_overshoot_ratio[29])
    pooled across all Mode-2 samples in run_dirs. overshoot_ratio = how many
    multiples of the joint's own half-range the worst q_target reached."""
    all_over = [[] for _ in range(29)]
    half_range = (HI - LO) / 2.0
    worst_mag = np.zeros(29)

    for rd in run_dirs:
        paths = {n: os.path.join(rd, n + ".csv") for n in ("q", "action", "encoder_mode")}
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        t, q = _read_csv(paths["q"], 29)
        _, action = _read_csv(paths["action"], 29)
        te, em = _read_csv(paths["encoder_mode"], 1)
        n = min(len(t), len(te))
        if n < 25:
            continue
        q, action, em = q[:n], action[:n], em[:n, 0]
        mask2 = em == 2
        if mask2.sum() < 20:
            continue
        qtgt = action_to_q_target(action)[mask2]
        finite = np.all(np.isfinite(qtgt), axis=1)
        qtgt = qtgt[finite]
        if len(qtgt) < 20:
            continue
        over = (qtgt < LO[None, :]) | (qtgt > HI[None, :])
        for j in range(29):
            all_over[j].append(over[:, j])
        excess = np.maximum(qtgt - HI[None, :], LO[None, :] - qtgt)
        excess = np.maximum(excess, 0)
        worst_mag = np.maximum(worst_mag, excess.max(axis=0))

    frac = np.full(29, np.nan)
    for j in range(29):
        if all_over[j]:
            cat = np.concatenate(all_over[j])
            frac[j] = cat.mean()
    ratio = worst_mag / half_range
    return frac, worst_mag, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    args = ap.parse_args()

    results = {}
    for s in args.sessions:
        frac, worst_mag, ratio = overshoot_stats(session_run_dirs(s))
        results[s] = (frac, worst_mag, ratio)

    for s in args.sessions:
        frac, worst_mag, ratio = results[s]
        print(f"=== {s}: fraction of Mode-2 samples with q_target outside physical range ===")
        order = np.argsort(-np.nan_to_num(frac))
        for j in order:
            if frac[j] <= 0 and worst_mag[j] <= 0.001:
                continue
            print(f"  {JOINT_NAMES[j]:16s}  frac_over={frac[j]*100:6.2f}%   "
                  f"worst_overshoot={worst_mag[j]:7.3f} rad beyond range "
                  f"({ratio[j]:5.2f}x the joint's own half-range)")
        if np.all(frac <= 0):
            print("  (no joint ever exceeds its physical range in this session)")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
