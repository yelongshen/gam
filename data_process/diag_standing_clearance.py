import sys
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
root = Path("/home/grease/ego_dataset/picoset_20261002_sit")
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names); fidx = S.foot_idx(names)
print("foot bodies:", [names[i] for i in fidx])
for tag, sub in (("snapped", "robot"), ("original", "robot_orig_backup")):
    per = {0.78: [], 0.80: [], 0.82: []}
    stancelow = []
    for p in sorted((root / sub).glob("*.pkl")):
        rv = joblib.load(p); rv = rv[next(iter(rv))]
        pos = E.fk_positions(fk, rv)
        low = pos[:, fidx, 2].min(1); pel = pos[:, names.index("pelvis"), 2]
        for th in per:
            m = pel > th
            if m.sum() > 15: per[th].append(np.median(low[m]))
        stancelow.append(np.percentile(low, 5))
    print(tag, " | ".join(f"pelvis>{th}: per-clip median-of-median {np.median(v):.3f} p90 {np.percentile(v, 90):.3f} max {np.max(v):.3f} (n={len(v)})" for th, v in per.items()), "| p5 of low over whole clip: median %.3f" % np.median(stancelow))
