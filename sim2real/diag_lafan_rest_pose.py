import glob
import os

import joblib
import numpy as np

R = os.path.expanduser("~/ego_dataset")
rows = []
for f in sorted(glob.glob(f"{R}/lafan1_evalset/smpl/*.pkl"))[::3] + sorted(glob.glob(f"{R}/lafan1_trainset/smpl/*.pkl"))[::3]:
    d = joblib.load(f)
    sj = np.asarray(d["smpl_joints"], dtype=np.float64)
    out = []
    for fr in (0, 5, 20, 60):
        j = sj[fr]
        def elev(a, b):
            v = j[a] - j[b]
            return np.degrees(np.arcsin(v[2] / np.linalg.norm(v)))
        out.append((elev(18, 16), elev(19, 17), elev(4, 1), elev(9, 0)))
    rows.append((os.path.basename(f)[:-4], out))
print("name | per frame (0,5,20,60): [L-arm elev, R-arm elev, L-shin elev, spine elev] deg (T-pose => arms ~0, legs ~-90, spine ~+90)")
for n, o in rows:
    print(f"{n:30s} " + "  ".join("[" + " ".join(f"{x:5.0f}" for x in t) + "]" for t in o))
