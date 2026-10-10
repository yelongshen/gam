import json, sys
from pathlib import Path
import joblib, numpy as np
sel = {
 "picoset_20260928": ["clip_018", "clip_021", "clip_003", "clip_005_006", "clip_013"],
 "picoset_20260929": ["clip_054", "clip_010", "clip_053", "clip_011", "clip_046", "clip_077"],
 "picoset_20260930": ["145524_clip_034", "145524_021", "145524_clip_021", "145524_clip_022", "162422_clip_016", "145524_clip_012"],
}
sel["picoset_20260930"] = ["145524_clip_034", "145524_clip_021", "145524_clip_022", "162422_clip_016", "145524_clip_012"]
PELV = np.array([0.003, -0.351, 0.012])
screen = json.load(open("/tmp/screen_retarget_start.json"))
print("clip | human: lowest toe z at f0 / median first 1 s | pelvis z f0 | human foot z range first 3 s || robot: lowest ankle z f0 | pelvis z f0")
import sys as _s
_s.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names); fidx = S.foot_idx(names)
for ds, cl in sel.items():
    pref = {"picoset_20260928": "pico0928_", "picoset_20260929": "pico0929_", "picoset_20260930": "pico0930_"}[ds]
    for c in cl:
        k = pref + c
        p = Path(f"/home/grease/ego_dataset/{ds}/smpl/{k}.pkl")
        if not p.exists():
            print(k, "missing"); continue
        s = joblib.load(p); s = s[next(iter(s))] if "smpl_joints" not in s else s
        J = np.asarray(s["smpl_joints"], float) - PELV + np.asarray(s["transl"], float)[:, None, :]
        toe = J[:, [10, 11], 2].min(1); pel = J[:, 0, 2]
        rv = joblib.load(f"/home/grease/ego_dataset/{ds}/robot/{k}.pkl"); rv = rv[next(iter(rv))]
        pos = E.fk_positions(fk, rv); low = pos[:, fidx, 2].min(1); rp = pos[:, names.index("pelvis"), 2]
        print(f"{k[-18:]:>18s} | {toe[0]:5.2f} / {np.median(toe[:50]):5.2f} | {pel[0]:5.2f} | [{toe[:150].min():5.2f},{toe[:150].max():5.2f}] || {low[0]:5.2f} | {rp[0]:5.2f}")
