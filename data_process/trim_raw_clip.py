#!/usr/bin/env python3
"""Trim the start off a raw PICO clip npz: cut everything before the subject first has a foot on the floor.

Used for clips that START in the air or elevated (e.g. a drop from a platform, or a jump already in progress): the reference
spawns off the ground and cannot be tracked.  The cut is placed at the first frame where the lowest foot joint is within
`--land` m of the clip's floor (5th-percentile floor, as in tag_clips.world_joints).

  .venv_teleop/bin/python data_process/trim_raw_clip.py <src.npz> <dst.npz> [--land 0.05]
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, "/home/grease/gam/gear_sonic/scripts")
from tag_clips import world_joints  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--land", type=float, default=0.05)
a = ap.parse_args()
W, q, t = world_joints(a.src)
low = W[:, [10, 11], 2].min(axis=1)
idx = np.where(low < a.land)[0]
i0 = int(idx[0]) if len(idx) else 0
d = dict(np.load(a.src, allow_pickle=True))
T = len(t)
out = {k: (v[i0:] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T else v) for k, v in d.items()}
if "clip_duration_s" in out:
    out["clip_duration_s"] = np.float32(t[-1] - t[i0])
os.makedirs(os.path.dirname(a.dst), exist_ok=True)
np.savez_compressed(a.dst, **out)
print(f"{os.path.basename(a.src)}: trimmed first {i0} of {T} frames ({t[i0]:.2f} s); lowest foot there {low[i0]:.3f} m, was {low[0]:.3f} m at the start; "
      f"{t[-1] - t[i0]:.1f} s remain")
