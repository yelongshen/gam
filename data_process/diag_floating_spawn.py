import sys, glob, csv
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
root = Path("/home/grease/ego_dataset/picoset_20261002_sit")
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names); fidx = S.foot_idx(names)
early = ["001", "008", "009", "024", "027", "032", "044", "045", "046"]
meta = {r["clip"][:-4]: r for r in csv.DictReader(open("/home/grease/g1_robot_data/pico_raw/20261002_151150_sit_clips/clips.csv"))}
print("clip  start_s dur_s | robot f0: clear pelvis | settle frame (clear<0.06 & dof speed<3) | smpl f0 foot z, pelvis z | smpl clearance after 1s")
for c in early + ["002", "003", "004"]:
    name = f"pico1002sit_clip_{c}"
    rv = joblib.load(root / "robot" / f"{name}.pkl"); rv = rv[next(iter(rv))]
    pos = E.fk_positions(fk, rv)
    low = pos[:, fidx, 2].min(1); pel = pos[:, names.index("pelvis"), 2]
    sp = np.r_[0, np.abs(np.diff(rv["dof"], axis=0)).max(1) * 30]
    ok = np.where((low < 0.06) & (sp < 3.0))[0]
    settle = int(ok[0]) if len(ok) else -1
    sm = joblib.load(root / "smpl" / f"{name}.pkl"); sm = sm[next(iter(sm))] if "smpl_joints" not in sm else sm
    J = np.asarray(sm["smpl_joints"], float) - np.array([0.003, -0.351, 0.012]) + np.asarray(sm["transl"], float)[:, None, :]
    fz = J[:, [10, 11], 2].min(1); pz = J[:, 0, 2]
    m = meta[c if "clip_" + c in meta else "clip_" + c]
    print(f"{c}   {float(m['start_s']):7.1f} {float(m['dur_s']):5.1f} | {low[0]:.3f} {pel[0]:.3f} | frame {settle:3d} ({settle / 30:.2f}s) | {fz[0]:.3f} {pz[0]:.3f} | {np.median(fz[50:100]):.3f}  maxspeed f0-5: {sp[:6].max():.1f}")
print("\nfirst 12 frames of clip 024: robot pelvis z / foot clearance / max dof speed")
rv = joblib.load(root / "robot" / "pico1002sit_clip_024.pkl"); rv = rv[next(iter(rv))]
pos = E.fk_positions(fk, rv); low = pos[:, fidx, 2].min(1); pel = pos[:, names.index("pelvis"), 2]
sp = np.r_[0, np.abs(np.diff(rv["dof"], axis=0)).max(1) * 30]
for t in range(0, 40, 3): print(f"  f{t:2d}: pel {pel[t]:.3f} clear {low[t]:.3f} dofspeed {sp[t]:.1f}")
