import glob
import os

import joblib
import numpy as np

R = os.path.expanduser("~/ego_dataset")


def first(d):
    return d[next(iter(d))] if isinstance(d, dict) and "dof" not in d else d


print("clip | robot_v2 root z: first / min / max | SMPL pelvis z (transl): first / min / max | lowest foot z in SMPL (relative to its first-frame)")
for split in ("lafan1_evalset", "lafan1_trainset"):
    print("==", split)
    for f in sorted(glob.glob(f"{R}/{split}/robot_v2/*.pkl")):
        n = os.path.basename(f)[:-4]
        r = first(joblib.load(f))
        z = np.asarray(r["root_trans_offset"])[:, 2]
        s = joblib.load(f"{R}/{split}/smpl/{n}.pkl")
        tz = np.asarray(s["transl"])[:, 2]
        sj = np.asarray(s["smpl_joints"])
        foot = (sj[:, 7, 2] - sj[:, 0, 2]) + tz        # ankle world z
        flag = "  <-- z0 high" if z[0] > 1.0 else ""
        print(f"{n:28s} {z[0]:5.2f} {z.min():5.2f} {z.max():5.2f} | {tz[0]:5.2f} {tz.min():5.2f} {tz.max():5.2f} | ankle z min {foot.min():5.2f} p1 {np.percentile(foot, 1):5.2f} first {foot[0]:5.2f}{flag}")
