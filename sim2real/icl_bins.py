#!/usr/bin/env python3
"""Per-attempt-block means of the multi-attempt ICL evals (logs_eval/*EVAL_icl_*_r30{keep,clr}).

Blocks of 5 attempts (1-5, 6-10, ..., 26-30) for success rate and progress rate, keep vs clr side by side.
Usage: python sim2real/icl_bins.py [--block 5] [--glob '*EVAL_icl_*_r30*']
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=5)
ap.add_argument("--glob", default="*EVAL_icl_*_r30*")
a = ap.parse_args()
os.chdir(os.path.expanduser("~/GR00T-WholeBodyControl/logs_eval"))
runs = {}
for d in sorted(glob.glob(a.glob)):
    try:
        r = json.load(open(d + "/metrics_eval.json"))["retry"]
    except Exception:
        continue
    runs[re.sub(r"^\d+_\d+-EVAL_icl_", "", d)] = r


def blocks(x):
    x = np.asarray(x, dtype=float)
    return [x[i:i + a.block].mean() for i in range(0, len(x), a.block)]


nb = len(blocks(next(iter(runs.values()))["success_rate_per_attempt"]))
hdr = " ".join(f"{i * a.block + 1:>2d}-{(i + 1) * a.block:<2d}" .rjust(7) for i in range(nb))
for metric, key in (("SUCCESS RATE", "success_rate_per_attempt"), ("PROGRESS RATE", "progress_rate_per_attempt")):
    print(f"\n{metric}: mean over each block of {a.block} attempts (35 clips)")
    print(f"{'run':20s} {hdr}   | last-vs-first block")
    for n, r in runs.items():
        b = blocks(r[key])
        print(f"{n:20s} " + " ".join(f"{v:7.3f}" for v in b) + f"   | {b[-1] - b[0]:+.3f}")
print("\nkeep minus clr (same checkpoint), success rate per block:")
for n in runs:
    if n.endswith("_keep"):
        c = n[:-5] + "_clr"
        if c in runs:
            d = np.array(blocks(runs[n]["success_rate_per_attempt"])) - np.array(blocks(runs[c]["success_rate_per_attempt"]))
            print(f"{n[:-5]:20s} " + " ".join(f"{v:+7.3f}" for v in d) + f"   | mean {d.mean():+.3f}")
