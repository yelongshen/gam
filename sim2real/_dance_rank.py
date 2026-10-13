import glob
import os

import joblib
import numpy as np

R = "/home/grease/ego_dataset/eval_subset"
have = {"dance_vouge_shake_it_babe_360_R_002__A318_M",
        "dance_hiphop_mike_tyson_R_fast_001__A319_M",
        "dance_latino_kick_kick_padeburee_doubled_R_002__A314"}
rows = []
for f in sorted(glob.glob(f"{R}/robot/*dance*.pkl")):
    n = os.path.basename(f)[:-4]
    if n in have:
        continue
    r = joblib.load(f)
    r = r[next(iter(r))]
    d = np.rad2deg(np.asarray(r["dof"]))
    fps = float(r.get("fps", 30))
    root = np.asarray(r["root_trans_offset"])
    jv = np.abs(np.diff(d, axis=0)) * fps
    rom = d.max(0) - d.min(0)
    rows.append((jv.mean(), jv.max(), rom[15:].mean(), rom[:12].mean(), len(d) / fps,
                 np.linalg.norm(np.diff(root, axis=0), axis=1).sum(),
                 root[:, 2].max() - root[:, 2].min(), n))
rows.sort(reverse=True)
print("joint mean/peak  armROM/legROM  dur  path  hrange  name")
for r in rows[:14]:
    print("%.1f/%.0f  %.0f/%.0f  %.1fs %.1fm %.2f  %s" % r)
print(len(rows), "dance clips not yet in manifest")
