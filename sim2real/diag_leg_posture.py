import glob
import os

import joblib
import numpy as np

R = os.path.expanduser("~/ego_dataset")


def first(d):
    return d[next(iter(d))] if isinstance(d, dict) and "dof" not in d else d


def row(p, label):
    r = first(joblib.load(p))
    d = np.degrees(np.asarray(r["dof"], dtype=np.float64))
    z = np.asarray(r["root_trans_offset"])[:, 2]
    print(f"{label:46s} root z mean {z.mean():.2f} | hip_pitch {d[:, [0, 6]].mean():6.1f} knee {d[:, [3, 9]].mean():6.1f} ankle_pitch {d[:, [4, 10]].mean():6.1f}")


for n in ["dance_hiphop_mike_tyson_R_fast_001__A319_M", "walk_180_R_003__A332_M", "dance_vouge_shake_it_babe_360_R_002__A318_M",
          "jog_ff_start_180_R_002__A192_M"]:
    row(f"{R}/eval_subset/robot/{n}.pkl", "eval_subset " + n[:30])
for f in sorted(glob.glob(f"{R}/amass_evalset/robot/*.pkl"))[:3]:
    row(f, "amass " + os.path.basename(f)[:34])
row("/tmp/gmr_test_ml/dance2_subject3.pkl", "GMR lafan dance2 window (foot rot weight 1)")
