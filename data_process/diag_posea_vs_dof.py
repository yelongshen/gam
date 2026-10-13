import sys, glob
from pathlib import Path
import joblib, numpy as np
root = Path("/home/grease/ego_dataset/picoset_20261002_sit/robot")
early = {"clip_%s" % x for x in ("001", "008", "009", "024", "027", "032", "044", "045", "046")}
rows = []
for p in sorted(root.glob("*.pkl")):
    cid = p.stem.replace("pico1002sit_", "")
    rv = joblib.load(p); rv = rv[next(iter(rv))]
    aa = np.asarray(rv["pose_aa"]); dof = np.asarray(rv["dof"])
    # joint rotation magnitudes from pose_aa (body 1..29) vs |dof|
    mag = np.linalg.norm(aa[:, 1:30], axis=-1)
    diff = np.abs(mag - np.abs(dof))
    # wrapped difference: rotvec magnitude is in [0, pi]
    wrapped = np.minimum(np.abs(np.abs(dof) - mag), np.abs(np.abs(dof) % (2 * np.pi) - mag))
    j = int(np.argmax(wrapped[0]))
    rows.append((cid, float(wrapped[0].max()), j, float(wrapped[:90].max()), float(np.abs(dof).max()), float(np.abs(dof[0]).max())))
print("clip      max |  |pose_aa joint| - |dof| |  at frame 0 (joint idx) | first 90 frames | max|dof| over clip")
for r in rows:
    if r[0] in early:
        print(f"{r[0]}  {'EARLY':5s} {r[1]:.4f} (j{r[2]})   {r[3]:.4f}   {r[4]:.2f}")
oth = np.array([r[1] for r in rows if r[0] not in early]); print("others: median %.4f max %.4f ; frames-0..90 max median %.4f" % (np.median(oth), oth.max(), np.median([r[3] for r in rows if r[0] not in early])))
