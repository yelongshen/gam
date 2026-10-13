#!/usr/bin/env python3
"""Characterize the transient right at the encoder_mode 0->2 transition
(start of active motion tracking), to compare "how aggressively" different
runs/robots begin tracking, independent of which policy is loaded.

Usage:
    .venv_sim/bin/python model_eval/sim2real_mode_transition_check.py \
        --run-dirs RUN_DIR [RUN_DIR ...] --label "new robot"
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES  # noqa: E402
from model_eval.sim2real_single_run_quicklook import load  # noqa: E402

PRE, POST = 5, 25  # samples before/after the 0->2 transition to inspect


def find_transitions(mode):
    """Indices i where mode[i-1] != 2 and mode[i] == 2 (rising edges into mode 2)."""
    m2 = (mode == 2)
    edges = np.nonzero(m2[1:] & ~m2[:-1])[0] + 1
    # also count i==0 if it starts directly in mode 2 (no visible transition -- skip, uninformative)
    return edges


def analyze_run(run_dir):
    q, t = load(run_dir, "q")
    dq, _ = load(run_dir, "dq")
    act, _ = load(run_dir, "action")
    tau, _ = load(run_dir, "motor_torque")
    mode, _ = load(run_dir, "encoder_mode", ncols=1)
    n = min(len(q), len(dq), len(act), len(tau), len(mode))
    if n < PRE + POST + 2:
        return []
    q, dq, act, tau, mode = q[:n], dq[:n], act[:n], tau[:n], mode[:n, 0]

    out = []
    for idx in find_transitions(mode):
        lo, hi = idx - PRE, idx + POST
        if lo < 0 or hi > n:
            continue
        window_dq = dq[lo:hi]
        window_tau = tau[lo:hi]
        window_act = act[lo:hi]
        pre_act = act[max(0, idx - PRE):idx]
        if len(pre_act) == 0:
            continue
        action_jump = np.abs(window_act[0] - pre_act[-1]).max()  # first post-transition action vs last pre
        out.append(dict(
            max_abs_dq=np.abs(window_dq).max(),
            rms_dq=np.sqrt((window_dq ** 2).mean()),
            max_abs_tau=np.abs(window_tau).max(),
            action_jump=action_jump,
            worst_joint_dq=JOINT_NAMES[int(np.abs(window_dq).max(0).argmax())],
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    all_events = []
    for rd in args.run_dirs:
        events = analyze_run(rd)
        for e in events:
            print(f"[{os.path.basename(rd)}] max|dq|={e['max_abs_dq']:7.3f} rad/s  "
                  f"rms(dq)={e['rms_dq']:6.3f}  max|tau|={e['max_abs_tau']:7.2f} N.m  "
                  f"action_jump={e['action_jump']:6.3f}  worst={e['worst_joint_dq']}")
        all_events.extend(events)

    if not all_events:
        print(f"[{args.label}] no 0->2 transitions found in these runs")
        return 1

    max_dq = np.array([e['max_abs_dq'] for e in all_events])
    rms_dq = np.array([e['rms_dq'] for e in all_events])
    max_tau = np.array([e['max_abs_tau'] for e in all_events])
    jump = np.array([e['action_jump'] for e in all_events])

    print(f"\n=== {args.label}: {len(all_events)} mode-0->2 transitions across "
          f"{len(args.run_dirs)} run dirs ===")
    print(f"max|dq|      : mean={max_dq.mean():6.3f}  median={np.median(max_dq):6.3f}  "
          f"p90={np.percentile(max_dq,90):6.3f}  max={max_dq.max():6.3f}")
    print(f"rms(dq)      : mean={rms_dq.mean():6.3f}  median={np.median(rms_dq):6.3f}  "
          f"p90={np.percentile(rms_dq,90):6.3f}")
    print(f"max|tau|     : mean={max_tau.mean():7.2f}  median={np.median(max_tau):7.2f}  "
          f"p90={np.percentile(max_tau,90):7.2f}")
    print(f"action_jump  : mean={jump.mean():6.3f}  median={np.median(jump):6.3f}  "
          f"p90={np.percentile(jump,90):6.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
