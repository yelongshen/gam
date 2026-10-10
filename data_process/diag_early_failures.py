import sys, json, glob
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
root = Path("/home/grease/ego_dataset/picoset_20261002_sit")
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names); fidx = S.foot_idx(names)
early = {"024", "032", "008", "009", "001", "044", "045", "027", "046"}
rows = []
for p in sorted((root / "robot").glob("*.pkl")):
    rv = joblib.load(p); rv = rv[next(iter(rv))]
    pos = E.fk_positions(fk, rv)
    low = pos[:, fidx, 2].min(1); pel = pos[:, names.index("pelvis"), 2]
    env = json.loads((root / "envs" / f"{p.stem}.json").read_text())
    cx, cy, top = env["seat_center"]
    d0 = float(np.hypot(pos[0, names.index("pelvis"), 0] - cx, pos[0, names.index("pelvis"), 1] - cy))
    vz = np.abs(np.diff(pos[:6, names.index("pelvis"), 2])).max() * 30
    dof = rv["dof"]
    rows.append((p.stem[-3:], float(low[0]), float(pel[0]), d0, float(top), vz, float(np.abs(np.diff(dof[:5], axis=0)).max() * 30)))
a = np.array([r[1:] for r in rows]); isearly = np.array([r[0] in early for r in rows])
lab = ["foot clearance f0 [m]", "pelvis z f0 [m]", "pelvis->chair distance f0 [m]", "seat top [m]", "pelvis vz first frames [m/s]", "max dof speed first frames [rad/s]"]
print(f"{'':36s}{'early-failing (n=%d)' % isearly.sum():>22s}{'others (n=%d)' % (~isearly).sum():>20s}")
for j, l in enumerate(lab):
    print(f"{l:36s}{np.median(a[isearly, j]):12.3f} [{a[isearly, j].min():.3f},{a[isearly, j].max():.3f}]  {np.median(a[~isearly, j]):8.3f} [{a[~isearly, j].min():.3f},{a[~isearly, j].max():.3f}]")
