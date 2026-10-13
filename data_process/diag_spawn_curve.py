import sys, json
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
root = Path("/home/grease/ego_dataset/picoset_20261002_sit")
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names); fidx = S.foot_idx(names)
PELV_OFF = np.array([0.003, -0.351, 0.012])
for cid in ("clip_024", "clip_032", "clip_005"):
    p = root / "robot" / f"pico1002sit_{cid}.pkl"
    rv = joblib.load(p); rv = rv[next(iter(rv))]
    pos = E.fk_positions(fk, rv)
    lz = pos[:, [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")], 2]
    pel = pos[:, names.index("pelvis"), 2]
    s = joblib.load(root / "smpl" / p.name); s = s[next(iter(s))] if "smpl_joints" not in s else s
    J = np.asarray(s["smpl_joints"], float) - PELV_OFF + np.asarray(s["transl"], float)[:, None, :]
    hf = J[:, [10, 11], 2]; hp = J[:, 0, 2]
    dof = rv["dof"]; spd = np.abs(np.diff(dof, axis=0)).max(1) * 30
    i30 = np.arange(0, 120, 8)                 # robot frames @30
    i50 = (i30 * 50 / 30).astype(int)          # human frames @50
    print(f"== {cid}")
    print(" t[s]         ", " ".join(f"{i / 30:5.2f}" for i in i30))
    print(" robot L foot ", " ".join(f"{x:5.2f}" for x in lz[i30, 0]))
    print(" robot R foot ", " ".join(f"{x:5.2f}" for x in lz[i30, 1]))
    print(" robot pelvis ", " ".join(f"{x:5.2f}" for x in pel[i30]))
    print(" max dof speed", " ".join(f"{x:5.1f}" for x in spd[np.minimum(i30, len(spd) - 1)]))
    print(" human L toe  ", " ".join(f"{x:5.2f}" for x in hf[i50, 0]))
    print(" human R toe  ", " ".join(f"{x:5.2f}" for x in hf[i50, 1]))
    print(" human pelvis ", " ".join(f"{x:5.2f}" for x in hp[i50]))
