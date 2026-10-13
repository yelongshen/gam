import json
import os
from collections import defaultdict

import numpy as np

m = json.load(open(os.path.expanduser("~/ego_dataset/ICL_hardset/manifest.json")))["clips"]
d = np.array([c["dur_s"] for c in m])
print(f"all {len(d)} clips: min {d.min():.1f} p25 {np.percentile(d, 25):.1f} median {np.median(d):.1f} p75 {np.percentile(d, 75):.1f} max {d.max():.1f} mean {d.mean():.1f}  total {d.sum():.0f}s ({d.sum() / 60:.1f} min)")
g = defaultdict(list)
for c in m:
    g[(c["source"], c["split"])].append(c["dur_s"])
for k, v in sorted(g.items()):
    v = np.array(v)
    print(f"  {k[0]:16s} {k[1]:11s} n={len(v):2d} min {v.min():5.1f} med {np.median(v):5.1f} max {v.max():5.1f}")
edges = [0, 3, 5, 7, 8.5, 10, 15, 25, 200]
h, _ = np.histogram(d, bins=edges)
print("\nhistogram (s):")
for a, b, n in zip(edges[:-1], edges[1:], h):
    print(f"  {a:>5}-{b:<5} {n:2d} {'#' * n}")
print("\nlongest:", [(c["name"][:40], c["dur_s"]) for c in sorted(m, key=lambda c: -c["dur_s"])[:5]])
print("shortest:", [(c["name"][:40], c["dur_s"]) for c in sorted(m, key=lambda c: c["dur_s"])[:5]])
