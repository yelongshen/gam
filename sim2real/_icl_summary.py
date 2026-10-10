import glob
import json
import os
import re

import numpy as np

os.chdir(os.path.expanduser("~/GR00T-WholeBodyControl/logs_eval"))
runs = {}
for d in sorted(glob.glob("*EVAL_icl_*_r30*")):
    try:
        j = json.load(open(d + "/metrics_eval.json"))
    except Exception:
        continue
    runs[re.sub(r"^\d+_\d+-EVAL_icl_", "", d)] = j["retry"]
k0 = next(iter(runs.values()))
print("retry keys:", {k: (type(v).__name__ + (str(len(v)) if hasattr(v, "__len__") else "")) for k, v in k0.items()})
print(f"\n{'run':20s} {'att1':>6s} {'mean30':>7s} {'max':>6s} {'last5':>6s}  per-attempt 1..10")
for n, r in runs.items():
    s = np.array(r["success_rate_per_attempt"])
    print(f"{n:20s} {s[0]:6.3f} {s.mean():7.3f} {s.max():6.3f} {s[-5:].mean():6.3f}  " + " ".join(f"{x:.2f}" for x in s[:10]))
for n, r in runs.items():
    extras = {k: v for k, v in r.items() if k not in ("success_rate_per_attempt",) and not isinstance(v, (list, dict))}
    print(n, extras)
    break
