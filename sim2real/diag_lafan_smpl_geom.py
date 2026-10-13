import os

import joblib
import numpy as np

R = os.path.expanduser("~/ego_dataset")
NAMES = {0: "pelvis", 1: "Lhip", 2: "Rhip", 3: "spine1", 6: "spine2", 9: "chest", 12: "neck", 13: "Lcollar",
         14: "Rcollar", 15: "head", 16: "Lsh", 17: "Rsh", 18: "Lelb", 19: "Relb", 20: "Lwr", 21: "Rwr"}


def geom(path, label):
    d = joblib.load(path)
    sj = np.asarray(d["smpl_joints"], dtype=np.float64)
    T = len(sj)
    L = lambda a, b: np.linalg.norm(sj[:, a] - sj[:, b], axis=1)
    print(f"\n== {label}  ({T} frames)")
    print("  median bone lengths (m): upperarm L/R %.3f/%.3f  forearm L/R %.3f/%.3f  shoulder width %.3f  hip width %.3f"
          % (np.median(L(16, 18)), np.median(L(17, 19)), np.median(L(18, 20)), np.median(L(19, 21)),
             np.median(L(16, 17)), np.median(L(1, 2))))
    print("  collar->shoulder L/R %.3f/%.3f   neck->chest %.3f  pelvis->spine1 %.3f spine1->spine2 %.3f spine2->chest %.3f"
          % (np.median(L(13, 16)), np.median(L(14, 17)), np.median(L(12, 9)), np.median(L(0, 3)),
             np.median(L(3, 6)), np.median(L(6, 9))))
    # side consistency: left-right vectors
    hip = sj[:, 1] - sj[:, 2]
    sh = sj[:, 16] - sj[:, 17]
    print("  cos(hipL-R, shoulderL-R): median %.3f (should be ~+1)" % np.median((hip * sh).sum(1) / (np.linalg.norm(hip, axis=1) * np.linalg.norm(sh, axis=1))))
    for a, b, nm in [(18, 16, "Lelb-Lsh"), (20, 18, "Lwr-Lelb")]:
        v = sj[:, a] - sj[:, b]
        print(f"  mean {nm} vec (frames median): {np.round(np.median(v, axis=0), 3)}")
    # index 0 neighbourhood: where is the pelvis & spine relative to hips (lean)
    print("  median pelvis->chest vec:", np.round(np.median(sj[:, 9] - sj[:, 0], axis=0), 3),
          " pelvis->neck:", np.round(np.median(sj[:, 12] - sj[:, 0], axis=0), 3))
    print("  hip midpoint - pelvis:", np.round(np.median((sj[:, 1] + sj[:, 2]) / 2 - sj[:, 0], axis=0), 3))
    # per-joint local-speed sanity: mean joint speed per joint (m/s)
    loc = sj - sj[:, :1]
    sp = np.linalg.norm(np.diff(loc, axis=0), axis=2).mean(0) * float(d.get("fps", 50))
    print("  mean speed (m/s): " + " ".join(f"{NAMES[i]}={sp[i]:.2f}" for i in NAMES if i in (16, 17, 18, 19, 20, 21, 13, 14, 12, 9)))


geom(f"{R}/eval_subset/smpl/walk_180_R_003__A332_M.pkl", "eval_subset walk_180 (AMASS-style SMPL)")
geom(f"{R}/eval_subset/smpl/dance_hiphop_mike_tyson_R_fast_001__A319_M.pkl", "eval_subset dance_hiphop")
geom(f"{R}/lafan1_evalset/smpl/dance2_subject3.pkl", "LAFAN dance2_subject3")
geom(f"{R}/lafan1_evalset/smpl/walk1_subject1.pkl", "LAFAN walk1_subject1")
