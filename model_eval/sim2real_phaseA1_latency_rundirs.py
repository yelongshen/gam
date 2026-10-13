#!/usr/bin/env python3
"""Phase A1 latency (actuator/tracking lag: q_target[j] -> q[j]) computed
directly from a list of run dirs, reusing the cross-correlation method from
sim2real_phaseA_latency.py -- but without requiring load_sim2real_session's
named-session (paired human capture) infrastructure, so it works on any
g1_run_* directory, including g1_run_0918.

NOTE: this only answers A1. A2 (end-to-end: human SMPL -> robot motion) needs
a timestamped human-side stream to align against; the smpl_20260919 CSVs
(joint_pos.csv etc.) have no timestamp columns, only raw indexed rows, so A2
is NOT reconstructable from that data without an external time reference.

Usage:
    .venv_sim/bin/python model_eval/sim2real_phaseA1_latency_rundirs.py \
        --run-dirs RUN_DIR [RUN_DIR ...] --label "g1_run_0918"
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES, action_to_q_target  # noqa: E402
from model_eval.sim2real_single_run_quicklook import load  # noqa: E402

FS = 50.0
MAX_LAG_S = 0.40


def xcorr(a, b, fs=FS, max_lag_s=MAX_LAG_S, highpass=True):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if highpass:
        a, b = np.diff(a), np.diff(b)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = int(max_lag_s * fs)
    lags = np.arange(-n, n + 1)
    cs = np.empty(len(lags))
    for i, L in enumerate(lags):
        if L >= 0:
            x, y = a[L:], (b[:len(b) - L] if L else b)
        else:
            x, y = a[:L], b[-L:]
        m = min(len(x), len(y))
        cs[i] = float(np.dot(x[:m], y[:m]) / m) if m else -1.0
    k = int(np.argmax(cs))
    lag = lags[k] / fs
    if 0 < k < len(cs) - 1:
        d = cs[k - 1] - 2 * cs[k] + cs[k + 1]
        if abs(d) > 1e-12:
            lag = (lags[k] + 0.5 * (cs[k - 1] - cs[k + 1]) / d) / fs
    return lag, cs[k], (k == 0 or k == len(cs) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--min_corr", type=float, default=0.30)
    ap.add_argument("--all-samples", action="store_true",
                     help="use all samples, not just encoder_mode==2")
    args = ap.parse_args()

    qt_list, q_list = [], []
    for rd in args.run_dirs:
        q, t = load(rd, "q")
        act, _ = load(rd, "action")
        mode, _ = load(rd, "encoder_mode", ncols=1)
        n = min(len(q), len(act), len(mode))
        if n < 50:
            continue
        q, act, mode = q[:n], act[:n], mode[:n, 0]
        qt = action_to_q_target(act)
        m = np.all(np.isfinite(q), 1) & np.all(np.isfinite(qt), 1)
        if not args.all_samples:
            m &= (mode == 2)
        if m.sum() < 50:
            continue
        # keep contiguous runs of True to preserve time-series structure for xcorr
        qt_list.append(qt[m]); q_list.append(q[m])
        print(f"[ok] {os.path.basename(rd)}: {m.sum()} samples")

    if not qt_list:
        print("no usable data"); return 1

    qt = np.concatenate(qt_list)
    q = np.concatenate(q_list)

    cz = np.array([np.corrcoef(qt[:, j], q[:, j])[0, 1] for j in range(29)])
    print(f"\n=== {args.label}: pooled {len(qt)} samples across {len(qt_list)} runs ===")
    print(f"reconstruction check: median corr(q_target,q) = {np.nanmedian(cz):.3f}, "
          f"median|q_target-q| = {np.median(np.abs(qt - q)):.4f} rad\n")

    print('--- A1  actuator lag:  q_target[j] -> q[j] ---')
    print(f"{'joint':16s} {'lag(ms)':>9s} {'corr':>7s} {'cmd_std':>9s}  note")
    rows = []
    for j in range(29):
        cmd_std = qt[:, j].std()
        lag, c, rail = xcorr(qt[:, j], q[:, j])
        note = ""
        if rail:
            note = "RAIL (unreliable)"
        elif c < args.min_corr:
            note = "weak corr, excluded"
        else:
            rows.append(lag)
        print(f"{JOINT_NAMES[j]:16s} {lag*1000:9.1f} {c:7.3f} {cmd_std:9.4f}  {note}")

    if rows:
        rows = np.array(rows)
        print(f"\nmedian A1 lag over {len(rows)} well-excited/reliable joints: "
              f"{np.median(rows)*1000:.1f} ms  (mean {rows.mean()*1000:.1f} ms)")
    else:
        print("\nno joints passed reliability filter")


if __name__ == "__main__":
    raise SystemExit(main())
