import os, sys
import numpy as np
ROOT = "/home/grease/gam"
sys.path.insert(0, os.path.join(ROOT, "sim2real"))
from deploy_constants import KP, KD, list_runs, load_run
rows = []
for r in list_runs():
    try:
        d = load_run(r)
    except Exception as e:  # noqa: BLE001
        print("ERR", r, e); continue
    q, dq, qt, tau = d["q"], d["dq"], d["q_target"], d["tau"]
    if any(np.isnan(v).any() for v in (q, dq, qt, tau)):
        continue
    pd = KP * (qt - q) - KD * dq
    s = (pd * tau).sum(0) / np.maximum((pd ** 2).sum(0), 1e-9)
    top = r.split("g1_robot_data/")[1]
    off = [(j, round(float(s[j]), 2)) for j in range(29) if abs(s[j] - 1) > 0.15]
    print(f"{top:45s} slope min {s.min():.2f} max {s.max():.2f}  joints with |slope-1|>0.15: {off}")
