#!/usr/bin/env python3
"""Consolidate a PICO session into ONE npz (newest frame of each 4-frame chunk) for fast analysis.

    .venv_teleop/bin/python gear_sonic/scripts/consolidate_pico_session.py <session_dir> [--out <npz>]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunk_pico_session import load_session  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("session_dir")
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or a.session_dir.rstrip("/") + "_session.npz"
d, files = load_session(a.session_dir)
np.savez(out, **{k: v for k, v in d.items() if not k.startswith("_")}, _t=d["_t"])
print("files", len(files), "frames", len(d["_t"]), "duration_s", float(d["_t"][-1]), "->", out)
print("keys", [k for k in d])
