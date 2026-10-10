import json
import os

import numpy as np

m = json.load(open(os.path.expanduser("~/ego_dataset/ICL_hardset/manifest.json")))["clips"]
r = np.array([c["robot_frames"] for c in m])
s = np.array([c["smpl_frames"] for c in m])
print(f"robot frames @30fps (dof, 29 joints): min {r.min()} p25 {np.percentile(r, 25):.0f} median {np.median(r):.0f} p75 {np.percentile(r, 75):.0f} max {r.max()} mean {r.mean():.0f} total {r.sum()}")
print(f"smpl frames  @50fps                  : min {s.min()} median {np.median(s):.0f} max {s.max()} total {s.sum()}")
edges = [0, 100, 150, 200, 230, 250, 300, 400, 700]
h, _ = np.histogram(r, bins=edges)
print("\nrobot-frame histogram:")
for a, b, n in zip(edges[:-1], edges[1:], h):
    print(f"  {a:>4}-{b:<4} {n:2d} {'#' * n}")
print("\nper source (robot frames):")
for src in sorted({(c["source"], c["split"]) for c in m}):
    v = np.array([c["robot_frames"] for c in m if (c["source"], c["split"]) == src])
    print(f"  {src[0]:16s} {src[1]:11s} n={len(v):2d} min {v.min():4d} median {np.median(v):4.0f} max {v.max():4d}")
print("\nall clips (robot frames):")
for c in sorted(m, key=lambda c: c["robot_frames"]):
    print(f"  {c['robot_frames']:4d}  {c['name'][:58]}")
