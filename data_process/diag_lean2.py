import sys
from pathlib import Path
import joblib, numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
root = Path("/home/grease/ego_dataset/picoset_20261002_sit")
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names)
early = {"clip_%s" % x for x in ("001", "008", "009", "024", "027", "032", "044", "045", "046")}
rows = []
for p in sorted((root / "robot").glob("*.pkl")):
    cid = p.stem.replace("pico1002sit_", "")
    bp = root / "spawnfix_backup" / "robot" / p.name
    rv = joblib.load(bp if bp.exists() else p); rv = rv[next(iter(rv))]
    pos = E.fk_positions(fk, rv)
    v = pos[:60, names.index("torso_link")] - pos[:60, names.index("pelvis")]
    lean = np.degrees(np.arccos(np.clip((v / np.linalg.norm(v, axis=1, keepdims=True))[:, 2], -1, 1))).mean()
    eul = R.from_quat(np.asarray(rv["root_rot"])[0]).as_euler("zyx", degrees=True)
    dof = np.asarray(rv["dof"])[:60]
    # waist pitch (index 14) mean, hip pitch L/R (0, 6)
    rows.append((cid, lean, eul[0], np.degrees(dof[:, 14].mean()), np.degrees(dof[:, 0].mean()), np.degrees(dof[:, 6].mean()), np.degrees(dof[:, 3].mean())))
a = np.array([r[1:] for r in rows]); isE = np.array([r[0] in early for r in rows])
print("                      early(n=9) mean / others(n=67) mean")
for j, nm in enumerate(["torso lean [deg]", "root yaw f0 [deg]", "waist_pitch dof [deg]", "L hip_pitch dof [deg]", "R hip_pitch dof [deg]", "L knee dof [deg]"]):
    print(f"{nm:24s} {a[isE, j].mean():8.1f} / {a[~isE, j].mean():8.1f}    (early range [{a[isE, j].min():.0f},{a[isE, j].max():.0f}]  others range [{a[~isE, j].min():.0f},{a[~isE, j].max():.0f}])")
lean = a[:, 0]; yaw = a[:, 1]
print("corr(lean, |yaw+80|) over all 76 clips: %.2f ;  corr(lean, yaw): %.2f" % (np.corrcoef(lean, np.abs(yaw + 80))[0, 1], np.corrcoef(lean, yaw)[0, 1]))
print("clips with lean > 25 deg:", [(r[0][-3:], round(r[1])) for r in rows if r[1] > 25], " (early set:", sorted(c[-3:] for c in early), ")")
