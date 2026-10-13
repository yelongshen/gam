import sys
from pathlib import Path
import joblib, numpy as np
root = Path("/home/grease/ego_dataset/picoset_20261002_sit")
early = {"clip_%s" % x for x in ("001", "008", "009", "024", "027", "032", "044", "045", "046")}
lean = {}
rows = []
for p in sorted((root / "smpl").glob("*.pkl")):
    cid = p.stem.replace("pico1002sit_", "")
    sp = root / "spawnfix_backup" / "smpl" / p.name
    s = joblib.load(sp if sp.exists() else p); s = s[next(iter(s))] if "smpl_joints" not in s else s
    J = np.asarray(s["smpl_joints"], float)[:100]
    pel = J[:, 0]
    # wrist (20,21), elbow (18,19), shoulder (16,17): horizontal distance from pelvis, and height relative to pelvis
    d = {}
    for nm, i in (("L wrist", 20), ("R wrist", 21), ("L elbow", 18), ("R elbow", 19)):
        v = J[:, i] - pel
        d[nm] = (float(np.linalg.norm(v[:, :2], axis=1).mean()), float(v[:, 2].mean()))
    rows.append((cid, d))
print("human, mean over first 2 s: horizontal distance pelvis->joint [m] / height above pelvis [m]")
for c, d in rows:
    if c in early:
        print(c, "EARLY", " ".join(f"{k}: {v[0]:.2f}/{v[1]:+.2f}" for k, v in d.items()))
for k in ("L wrist", "R wrist", "L elbow", "R elbow"):
    a = np.array([d[k][0] for c, d in rows if c not in early]); h = np.array([d[k][1] for c, d in rows if c not in early])
    e = np.array([d[k][0] for c, d in rows if c in early]); eh = np.array([d[k][1] for c, d in rows if c in early])
    print(f"{k}: early horiz {e.mean():.2f} height {eh.mean():+.2f} | others horiz median {np.median(a):.2f} [{a.min():.2f},{a.max():.2f}] height median {np.median(h):+.2f}")
