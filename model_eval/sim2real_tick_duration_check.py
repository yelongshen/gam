#!/usr/bin/env python3
"""Measure the REAL per-tick duration of the control loop directly from the
state_logger CSVs, using the fact that `time_ms`/`time_monotonic_ms` is
stamped once per tick, at the end of that tick's fully synchronous pipeline
(gather -> encode -> policy infer -> command -> log). If any tick's compute
exceeds the nominal 20ms (50 Hz) period, the loop cannot catch up -- it just
runs late -- so the GAP between consecutive log rows IS the real tick
duration, directly measurable with no extra instrumentation.

Usage:
    .venv_sim/bin/python model_eval/sim2real_tick_duration_check.py \
        --run-dirs RUN_DIR [RUN_DIR ...] --label "g1_run_0918"
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_eval.sim2real_single_run_quicklook import load  # noqa: E402

NOMINAL_DT_MS = 20.0


def tick_gaps(run_dir):
    """Return array of inter-row gaps in ms, using time_monotonic_ms (col 3,
    robust to any wall-clock adjustments) via the `t` return value of load(),
    which is actually time_ms (col 1). Use raw genfromtxt for monotonic col."""
    q, t_relative = load(run_dir, "q")
    # `load()` only exposes column 1 (time_ms); that's fine here, it's
    # relative and monotonic within a run (no epoch jumps possible).
    if len(t_relative) < 3:
        return None
    gaps = np.diff(t_relative)
    return gaps[np.isfinite(gaps) & (gaps > 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    all_gaps = []
    for rd in args.run_dirs:
        g = tick_gaps(rd)
        if g is None:
            print(f"[skip] {os.path.basename(rd)}: too few rows")
            continue
        all_gaps.append(g)
        over20 = (g > NOMINAL_DT_MS + 1.0).mean() * 100  # +1ms slack for jitter
        print(f"[{os.path.basename(rd)}] n={len(g)}  median={np.median(g):.2f}ms  "
              f"p99={np.percentile(g,99):.2f}ms  max={g.max():.2f}ms  "
              f"%>21ms={over20:.1f}%")

    if not all_gaps:
        print("no usable data"); return 1

    g = np.concatenate(all_gaps)
    print(f"\n=== {args.label}: pooled {len(g)} tick gaps across {len(all_gaps)} runs ===")
    print(f"median gap        : {np.median(g):.3f} ms  (nominal {NOMINAL_DT_MS:.0f} ms)")
    print(f"mean gap          : {g.mean():.3f} ms")
    print(f"p90 / p99 / max   : {np.percentile(g,90):.2f} / {np.percentile(g,99):.2f} / {g.max():.2f} ms")
    for thr in [21, 25, 30, 40, 50]:
        pct = (g > thr).mean() * 100
        print(f"fraction of ticks > {thr:3d} ms : {pct:6.2f}%")


if __name__ == "__main__":
    raise SystemExit(main())
