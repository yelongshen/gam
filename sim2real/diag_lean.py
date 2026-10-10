import os

import joblib
import numpy as np

R = os.path.expanduser("~/ego_dataset")


def lean(path, label, sl=slice(None)):
    d = joblib.load(path)
    sj = np.asarray(d["smpl_joints"], dtype=np.float64)[sl]
    # body frame per frame: lateral = R hip - L hip, up ~ pelvis->spine2, forward = up x lateral... (sign fixed below)
    up = sj[:, 6] - sj[:, 0]
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    lat = sj[:, 2] - sj[:, 1]
    lat -= (lat * up).sum(1, keepdims=True) * up
    lat /= np.linalg.norm(lat, axis=1, keepdims=True)
    fwd = np.cross(lat, up)  # points forward for a Z-up right-handed body frame with lat=R-L? sign fixed by foot below
    foot = (sj[:, 10] + sj[:, 11]) / 2 - (sj[:, 7] + sj[:, 8]) / 2          # ankle -> toe, points forward
    if np.median((fwd * foot).sum(1)) < 0:
        fwd = -fwd
    out = []
    for a, b, nm in [(0, 3, "pelvis->spine1"), (3, 6, "spine1->spine2"), (6, 9, "spine2->chest"), (9, 12, "chest->neck"),
                     (0, 6, "pelvis->spine2"), (0, 12, "pelvis->neck")]:
        v = sj[:, b] - sj[:, a]
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        # forward lean angle relative to world vertical (Z-up)
        lean_w = np.degrees(np.arctan2((v * fwd).sum(1), v[:, 2]))
        out.append(f"{nm}: {np.median(lean_w):+5.1f}")
    print(f"{label:34s} " + " | ".join(out))


lean(f"{R}/eval_subset/smpl/walk_180_R_003__A332_M.pkl", "eval_subset walk_180 (SMPL)")
lean(f"{R}/eval_subset/smpl/dance_hiphop_mike_tyson_R_fast_001__A319_M.pkl", "eval_subset dance_hiphop")
lean(f"{R}/eval_subset/smpl/jog_ff_start_180_R_002__A192_M.pkl", "eval_subset jog")
lean(f"{R}/lafan1_evalset/smpl/dance2_subject3.pkl", "LAFAN dance2_subject3", slice(10000, 11000))
lean(f"{R}/lafan1_evalset/smpl/walk1_subject1.pkl", "LAFAN walk1_subject1", slice(2000, 4000))
lean(f"{R}/lafan1_evalset/smpl/run1_subject5.pkl", "LAFAN run1_subject5", slice(2000, 4000))
