import sys
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
root = Path("/home/grease/ego_dataset/picoset_20261002_sit")
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names)
early = {"clip_%s" % x for x in ("001", "008", "009", "024", "027", "032", "044", "045", "046")}
PEL = np.array([0.003, -0.351, 0.012])
def tilt(v):   # angle of vector from vertical (deg) and its horizontal direction
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(v[..., 2], -1, 1)))
rows = []
for p in sorted((root / "robot").glob("*.pkl")):
    cid = p.stem.replace("pico1002sit_", "")
    # use the ORIGINAL (pre-fix) data for the 9 so the comparison is untouched by my edits
    bp = root / "spawnfix_backup" / "robot" / p.name
    rv = joblib.load(bp if bp.exists() else p); rv = rv[next(iter(rv))]
    pos = E.fk_positions(fk, rv)
    n = int(2.0 * 30)
    spine = pos[:n, names.index("torso_link")] - pos[:n, names.index("pelvis")]
    rt = tilt(spine).mean()
    # human
    sp = root / "spawnfix_backup" / "smpl" / p.name
    s = joblib.load(sp if sp.exists() else root / "smpl" / p.name); s = s[next(iter(s))] if "smpl_joints" not in s else s
    J = np.asarray(s["smpl_joints"], float)[: int(2.0 * 50)]
    hs = J[:, 9] - J[:, 0]     # spine3 - pelvis (smpl joints are pelvis-pinned, world-oriented)
    ht = tilt(hs).mean()
    # sagittal lean sign: component of the spine along robot-forward? use pelvis pitch from root_rot instead
    rows.append((cid, rt, ht))
print("mean tilt of the pelvis->head (robot) / pelvis->neck (human) vector from vertical over the first 2 s [deg]")
for r in rows:
    if r[0] in early:
        print(f"{r[0]}  EARLY  robot {r[1]:5.1f}   human {r[2]:5.1f}   diff {r[1] - r[2]:+5.1f}")
oth = np.array([[r[1], r[2]] for r in rows if r[0] not in early])
print("others: robot median %.1f [%.1f,%.1f]   human median %.1f [%.1f,%.1f]   diff median %+.1f" % (np.median(oth[:, 0]), oth[:, 0].min(), oth[:, 0].max(), np.median(oth[:, 1]), oth[:, 1].min(), oth[:, 1].max(), np.median(oth[:, 0] - oth[:, 1])))
