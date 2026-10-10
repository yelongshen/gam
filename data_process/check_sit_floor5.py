import glob
import numpy as np
import joblib
PELVIS_OFFSET = np.array([0.003, -0.351, 0.012])
fs = sorted(glob.glob("/home/grease/ego_dataset/picoset_20261002_sit/smpl/*.pkl"))
per, n_std = [], []
for f in fs:
    x = joblib.load(f); x = x[next(iter(x))] if "smpl_joints" not in x else x
    J = np.asarray(x["smpl_joints"], float) - PELVIS_OFFSET + np.asarray(x["transl"], float)[:, None, :]
    fmin = J[:, [10, 11], 2].min(1)
    fmin = fmin - np.percentile(fmin, 5)
    z = J[:, 0, 2] - 0  # pelvis (floor-corrected the same way)
    z = z - np.percentile(J[:, [10, 11], 2].min(1), 5)
    std = z > 0.9
    if std.sum() > 25:
        per.append(np.median(fmin[std])); n_std.append(int(std.sum()))
per = np.array(per)
print(f"clips with standing frames: {len(per)}/{len(fs)}  (standing = pelvis z > 0.9 m)")
print("per-clip MEDIAN lowest-foot height while standing: mean %.3f  median %.3f  std %.3f  min %.3f  p10 %.3f  p90 %.3f  max %.3f" %
      (per.mean(), np.median(per), per.std(), per.min(), np.percentile(per, 10), np.percentile(per, 90), per.max()))
print("clips with per-clip value > 0.05 m:", int((per > 0.05).sum()), " > 0.10 m:", int((per > 0.10).sum()))
