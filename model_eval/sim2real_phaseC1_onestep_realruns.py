#!/usr/bin/env python3
"""Re-run the phaseC1_damping_scan.md sec 6 one-step gain-scale test on REAL
online deploy-run sessions (g1_run_0905, g1_run_0911, ...), not just the
original `aug11` single-session log.

Unlike `aug11` (loaded via data_process/load_sim2real_session.py, a single
continuous recording), these sessions are directories of independent
`g1_deploy_run_*` runs. This script:
  1. loads q/dq/action/motor_torque/encoder_mode for every run in a session,
  2. keeps only CONTIGUOUS Mode-2 stretches within each run (so q_real_next
     is never taken across a mode transition or a run boundary),
  3. concatenates all such (q, dq, q_target, q_next) pairs across runs,
  4. runs the same four gain configs as phaseC1_damping_scan.md sec 6.2-6.4:
       - baseline (nominal, uncalibrated)      <- training ground truth
       - Phase B calibrated (dof_damping)       <- proxy for REAL, unscaled
       - calibrated + scaled 1.5/1.5 (upstream) <- predicted REAL* under scaling
       - calibrated + scaled Kp=1.0/Kd=1.5      <- measurement-derived variant
  5. reports error_before = RMS(q_real_next - q_sim_next[baseline])  and
     error_after[cfg]   = RMS(q_sim_next[cfg] - q_sim_next[baseline])
     exactly as phaseC1_damping_scan.md sec 6.2 defines them.

Usage:
    .venv_sim/bin/python model_eval/sim2real_phaseC1_onestep_realruns.py \
        g1_run_0905 g1_run_0911
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES, action_to_q_target  # noqa: E402
from model_eval.sim2real_phaseB_refit import ROOT, _read_csv  # noqa: E402
from model_eval.sim2real_phaseC1_onestep_prediction import (  # noqa: E402
    FS, build_model, one_step_predict, scaled_gains,
)

TARGET_JOINTS = ("L_ankle_pitch", "R_ankle_pitch")


def load_run_contiguous_mode2(run_dir):
    """Return list of (q_seg, dq_seg, qtgt_seg) arrays, one per contiguous
    Mode-2 stretch in this run (each stretch length >= 2 so a q_next exists)."""
    needed = ["q", "dq", "action", "encoder_mode"]
    paths = {n: os.path.join(run_dir, n + ".csv") for n in needed}
    if not all(os.path.exists(p) for p in paths.values()):
        return []
    _, q = _read_csv(paths["q"], 29)
    _, dq = _read_csv(paths["dq"], 29)
    _, action = _read_csv(paths["action"], 29)
    _, em = _read_csv(paths["encoder_mode"], 1)
    n = min(len(q), len(dq), len(action), len(em))
    if n < 3:
        return []
    q, dq, action, em = q[:n], dq[:n], action[:n], em[:n, 0]
    qtgt = action_to_q_target(action)

    finite = np.all(np.isfinite(q), 1) & np.all(np.isfinite(dq), 1) & np.all(np.isfinite(qtgt), 1)
    mask = (em == 2) & finite

    segments = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= 2:  # need at least one (t, t+1) pair
            segments.append((q[i:j], dq[i:j], qtgt[i:j]))
        i = j
    return segments


def load_session(session_name):
    session_dir = os.path.join(ROOT, session_name)
    run_dirs = sorted(glob.glob(os.path.join(session_dir, "g1_deploy_run_*")))
    all_q, all_dq, all_qtgt, all_qnext = [], [], [], []
    used = []
    for rd in run_dirs:
        segs = load_run_contiguous_mode2(rd)
        total = 0
        for q_seg, dq_seg, qtgt_seg in segs:
            all_q.append(q_seg[:-1])
            all_dq.append(dq_seg[:-1])
            all_qtgt.append(qtgt_seg[:-1])
            all_qnext.append(q_seg[1:])
            total += len(q_seg) - 1
        used.append((os.path.basename(rd), total, len(segs)))
    if not all_q:
        return None
    return (np.concatenate(all_q), np.concatenate(all_dq),
            np.concatenate(all_qtgt), np.concatenate(all_qnext), used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    args = ap.parse_args()

    cand_sets = {
        "baseline (nominal, = training ground truth)":
            dict(extra_damping={}, kp_scale=None, kd_scale=None),
        "Phase B calibrated (proxy for REAL, unscaled)":
            dict(extra_damping={4: 1.19, 10: 1.09, 14: 0.98}, kp_scale=None, kd_scale=None),
        "calibrated + scaled 1.5/1.5 (predicted REAL*, upstream)":
            dict(extra_damping={4: 1.19, 10: 1.09, 14: 0.98},
                 kp_scale={4: 1.5, 10: 1.5}, kd_scale={4: 1.5, 10: 1.5}),
        "calibrated + scaled Kp=1.0/Kd=1.5 (predicted REAL*, measured)":
            dict(extra_damping={4: 1.19, 10: 1.09, 14: 0.98},
                 kp_scale=None, kd_scale={4: 1.5, 10: 1.5}),
    }
    highlight_idx = [JOINT_NAMES.index(j) for j in TARGET_JOINTS]

    for session in args.sessions:
        loaded = load_session(session)
        if loaded is None:
            print(f"[{session}] no usable Mode-2 data found")
            continue
        q_real, dq_real, qt_real, q_real_next, used = loaded
        print(f"\n=== {session} ===  ({len(q_real)} contiguous Mode-2 one-step "
              f"pairs from {len(used)} runs)")
        for name, total, nseg in used:
            print(f"    {name}: {total} pairs ({nseg} contiguous segments)")

        raw_q_sim_next = {}
        for name, cfg in cand_sets.items():
            model = build_model(cfg["extra_damping"])
            kp, kd = scaled_gains(cfg["kp_scale"], cfg["kd_scale"])
            q_sim_next, _ = one_step_predict(model, q_real, dq_real, qt_real, 1 / FS,
                                              kp=kp, kd=kd)
            raw_q_sim_next[name] = q_sim_next
            print(f"    done: {name}")

        names = list(cand_sets)
        ground_truth = raw_q_sim_next[names[0]]

        def rms_deg(a, b):
            return np.degrees(np.sqrt(((a - b) ** 2).mean(0)))

        error_before = rms_deg(q_real_next, ground_truth)
        error_after = {n: rms_deg(raw_q_sim_next[n], ground_truth) for n in names[1:]}

        print(f"\n  [{session}] q1step-style RMS (deg) vs training ground truth:")
        header = f"    {'joint':16s} {'q_real_next (actual)':>24s} " + \
                 " ".join(f"{n[:24]:>24s}" for n in names[1:])
        print(header)
        for j, jname in zip(highlight_idx, TARGET_JOINTS):
            row = f"    {jname:16s} {error_before[j]:24.4f} "
            row += " ".join(f"{error_after[n][j]:24.4f}" for n in names[1:])
            print(row)
        print(f"    {'ALL-29 MEAN':16s} {error_before.mean():24.4f} " +
              " ".join(f"{error_after[n].mean():24.4f}" for n in names[1:]))

        print(f"\n  [{session}] % change vs error_before (negative = closer to training):")
        for j, jname in zip(highlight_idx, TARGET_JOINTS):
            for n in names[1:]:
                delta = error_after[n][j] - error_before[j]
                pct = 100 * delta / error_before[j]
                verdict = "CLOSER" if delta < 0 else "FARTHER"
                print(f"    {jname:16s} | {n:50s} | {pct:+7.1f}%  -> {verdict}")
        for n in names[1:]:
            delta = error_after[n].mean() - error_before.mean()
            pct = 100 * delta / error_before.mean()
            verdict = "CLOSER" if delta < 0 else "FARTHER"
            print(f"    {'ALL-29 MEAN':16s} | {n:50s} | {pct:+7.1f}%  -> {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
