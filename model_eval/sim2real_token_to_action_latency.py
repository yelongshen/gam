#!/usr/bin/env python3
"""Encoder-input -> policy-output latency, measured directly from the
robot-side state_logger CSVs (no external human-capture timestamps needed).

WHAT THIS MEASURES AND WHAT IT DOES NOT
----------------------------------------
`token_state.csv` is the encoder's latent output (the policy's actual
"observation of the incoming motion" after GatherObservations()/encoder
inference) -- i.e. the closest thing to "input" available in these logs.
`action.csv` is the raw policy output for that same tick.

Both are logged once per 50 Hz control tick, AFTER that tick's full pipeline
(gather -> encode -> policy inference -> action) has already completed. So:

  - This CANNOT measure sub-tick/wall-clock compute latency (e.g. "the
    policy.forward() call took 3ms") -- that information isn't in the CSVs,
    only in the deploy binary's console timing log, which isn't part of the
    state_logger output.
  - What IS measurable: whether a CHANGE in the encoder's token output at
    tick t is reflected in the policy's action at tick t, t+1, t+2, ... i.e.
    an integer-tick lag between "new information enters the token stream"
    and "policy output visibly responds to it". Resolution is exactly
    1 tick = 20 ms @ 50 Hz -- cannot resolve anything finer.

Method: cross-correlate the token stream's rate-of-change (mean |diff| across
all 64 token dims, to get a single scalar "how much new information arrived
this tick" signal) against the action stream's rate-of-change (mean |diff|
across all 29 action dims), same xcorr method as Phase A.

Usage:
    .venv_sim/bin/python model_eval/sim2real_token_to_action_latency.py \
        --run-dirs RUN_DIR [RUN_DIR ...] --label "g1_run_0918"
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_eval.sim2real_single_run_quicklook import load  # noqa: E402
from model_eval.sim2real_phaseA1_latency_rundirs import xcorr  # noqa: E402

FS = 50.0
MAX_LAG_TICKS = 10  # +/- 200 ms search window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--all-samples", action="store_true")
    args = ap.parse_args()

    tok_list, act_list = [], []
    for rd in args.run_dirs:
        try:
            tok, t = load(rd, "token_state", ncols=64)
        except Exception as e:
            print(f"[skip] {os.path.basename(rd)}: {e}")
            continue
        act, _ = load(rd, "action")
        mode, _ = load(rd, "encoder_mode", ncols=1)
        n = min(len(tok), len(act), len(mode))
        if n < 60:
            continue
        tok, act, mode = tok[:n], act[:n], mode[:n, 0]
        m = np.all(np.isfinite(tok), 1) & np.all(np.isfinite(act), 1)
        if not args.all_samples:
            m &= (mode == 2)
        if m.sum() < 60:
            continue
        tok_list.append(tok[m]); act_list.append(act[m])
        print(f"[ok] {os.path.basename(rd)}: {m.sum()} samples")

    if not tok_list:
        print("no usable token_state/action data")
        return 1

    tok = np.concatenate(tok_list)
    act = np.concatenate(act_list)

    # scalar "novelty" signal per tick: how much did the token/action vector
    # change since the previous tick (mean abs diff across dims).
    tok_novelty = np.concatenate([[0.0], np.abs(np.diff(tok, axis=0)).mean(1)])
    act_novelty = np.concatenate([[0.0], np.abs(np.diff(act, axis=0)).mean(1)])

    print(f"\n=== {args.label}: pooled {len(tok)} samples across {len(tok_list)} runs ===")
    print(f"token novelty: mean={tok_novelty.mean():.5f} std={tok_novelty.std():.5f}")
    print(f"action novelty: mean={act_novelty.mean():.5f} std={act_novelty.std():.5f}")

    lag, corr, rail = xcorr(tok_novelty, act_novelty, fs=FS,
                             max_lag_s=MAX_LAG_TICKS / FS, highpass=False)
    print(f"\ntoken novelty -> action novelty lag: {lag*1000:+.1f} ms "
          f"({lag*FS:+.2f} ticks)  corr={corr:.3f}{'  [RAIL -- unreliable]' if rail else ''}")
    print("(NEGATIVE lag = action novelty follows token novelty by |lag| -- the "
          "causally expected direction for an encoder->policy pipeline. Verified "
          "from xcorr()'s index arithmetic: for L<0, cs[L] = mean(tok[i] * act[i+|L|]), "
          "i.e. tok(t) is matched against act(t+|L|).)")


if __name__ == "__main__":
    raise SystemExit(main())
