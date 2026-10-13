import glob
import os

import joblib
import numpy as np

R = os.path.expanduser("~/ego_dataset")


def first(d):
    return d[next(iter(d))] if isinstance(d, dict) and "dof" not in d else d


def elbow_stats(files, label):
    fr = []
    mn = []
    for f in files:
        r = first(joblib.load(f))
        e = np.degrees(np.asarray(r["dof"], dtype=np.float64)[:, [18, 25]])
        fr.append((e < -10).mean())
        mn.append(e.min())
    print(f"{label:34s} clips {len(files):3d}  frac frames elbow<-10deg: mean {np.mean(fr):.3f} p90 {np.percentile(fr, 90):.3f}  min over clips {np.min(mn):.0f}")


elbow_stats(sorted(glob.glob(f"{R}/eval_subset/robot/*.pkl"))[::4], "eval_subset (reference)")
elbow_stats(sorted(glob.glob(f"{R}/amass_evalset/robot/*.pkl"))[::3], "amass_evalset")
elbow_stats(["/tmp/gmr_test_ml/dance2_subject3.pkl"], "GMR posonly dance2 window")
