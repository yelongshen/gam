import sys
from pathlib import Path
import joblib, numpy as np
from scipy.spatial.transform import Rotation as R
root = Path("/home/grease/ego_dataset/picoset_20261002_sit/robot")
early = {"clip_%s" % x for x in ("001", "008", "009", "024", "027", "032", "044", "045", "046")}
rows = []
for p in sorted(root.glob("*.pkl")):
    cid = p.stem.replace("pico1002sit_", "")
    rv = joblib.load(p); rv = rv[next(iter(rv))]
    aa = np.asarray(rv["pose_aa"])[:, 0]; q = np.asarray(rv["root_rot"])       # root_rot xyzw
    r1 = R.from_rotvec(aa); r2 = R.from_quat(q)
    ang = np.degrees((r1.inv() * r2).magnitude())
    # heading (yaw) of the root and tilt
    eul = r2.as_euler("zyx", degrees=True)
    rows.append((cid, float(ang[0]), float(ang[:90].max()), float(eul[0, 0]), float(eul[0, 1]), float(eul[0, 2])))
print("clip      |angle(pose_aa[0]) vs root_rot| frame0 / max first 90 | root yaw, pitch, roll at frame 0 [deg]")
for r in rows:
    if r[0] in early:
        print(f"{r[0]}  EARLY {r[1]:7.3f} {r[2]:7.3f} | {r[3]:7.1f} {r[4]:6.1f} {r[5]:6.1f}")
oth = [r for r in rows if r[0] not in early]
print("others: frame0 median %.3f max %.3f | first-90 max median %.3f max %.3f" % (np.median([r[1] for r in oth]), max(r[1] for r in oth), np.median([r[2] for r in oth]), max(r[2] for r in oth)))
print("others: root pitch at f0 median %.1f range [%.1f, %.1f] ; roll median %.1f range [%.1f, %.1f]" % (np.median([r[4] for r in oth]), min(r[4] for r in oth), max(r[4] for r in oth), np.median([r[5] for r in oth]), min(r[5] for r in oth), max(r[5] for r in oth)))
