#!/usr/bin/env python3
"""At the encoder_mode 0->2 transition, compute the position ERROR
(q_target - q, via reconstructed action_to_q_target) right at onset, to see
whether a bigger onset error (not a gain/damping difference) explains the
bigger post-transition dq spike.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES, action_to_q_target  # noqa: E402
from model_eval.sim2real_single_run_quicklook import load  # noqa: E402
from model_eval.sim2real_mode_transition_check import find_transitions  # noqa: E402

PRE, POST = 2, 8


def analyze_run(run_dir):
    q, t = load(run_dir, "q")
    dq, _ = load(run_dir, "dq")
    act, _ = load(run_dir, "action")
    mode, _ = load(run_dir, "encoder_mode", ncols=1)
    n = min(len(q), len(dq), len(act), len(mode))
    if n < PRE + POST + 2:
        return []
    q, dq, act, mode = q[:n], dq[:n], act[:n], mode[:n, 0]
    qt = action_to_q_target(act)
    e = qt - q

    out = []
    for idx in find_transitions(mode):
        if idx - PRE < 0 or idx + POST > n:
            continue
        e_onset = e[idx]  # error at the very first mode-2 sample
        out.append(dict(e_onset=e_onset, run=os.path.basename(run_dir)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    events = []
    for rd in args.run_dirs:
        events.extend(analyze_run(rd))

    if not events:
        print(f"[{args.label}] no transitions found")
        return 1

    e_all = np.stack([ev["e_onset"] for ev in events])
    print(f"=== {args.label}: {len(events)} transitions ===")
    print(f"{'joint':15s} {'mean|e|(deg)':>13s} {'max|e|(deg)':>12s}")
    order = np.argsort(-np.abs(e_all).mean(0))
    for j in order[:10]:
        print(f"{JOINT_NAMES[j]:15s} {np.degrees(np.abs(e_all[:, j]).mean()):13.2f} "
              f"{np.degrees(np.abs(e_all[:, j]).max()):12.2f}")
    print(f"{'ALL-29 mean':15s} {np.degrees(np.abs(e_all).mean()):13.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
