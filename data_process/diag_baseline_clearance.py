import sys
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names); fidx = S.foot_idx(names)
for tag, d in (("0930", "/home/grease/ego_dataset/picoset_20260930/robot"), ("0929", "/home/grease/ego_dataset/picoset_20260929/robot")):
    meds, pels = [], []
    for p in sorted(Path(d).glob("*.pkl"))[:70]:
        rv = joblib.load(p); rv = rv[next(iter(rv))]
        pos = E.fk_positions(fk, rv)
        low = pos[:, fidx, 2].min(1); pel = pos[:, names.index("pelvis"), 2]
        m = pel > 0.74
        if m.sum() > 30:
            meds.append(np.median(low[m])); pels.append(np.median(pel[m]))
    print(f"{tag}: n={len(meds)}  lowest ankle-link z while standing/walking: median {np.median(meds):.3f}  p10 {np.percentile(meds,10):.3f} p90 {np.percentile(meds,90):.3f} | pelvis z median {np.median(pels):.3f}")
