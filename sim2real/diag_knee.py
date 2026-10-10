import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.expanduser("~/gam/sim2real"))
from detect_arm_retarget_errors import angle  # noqa: E402

s = joblib.load(os.path.expanduser("~/ego_dataset/lafan1_smpl_filtered_FPS30/dance2_subject3.pkl"))
sj = np.asarray(s["smpl_joints"], dtype=np.float64)[3600:4500]
for nm, (a, b, c) in {"L knee": (1, 4, 7), "R knee": (2, 5, 8)}.items():
    flex = 180 - angle(sj[:, a], sj[:, b], sj[:, c])
    print(f"SMPL(LAFAN) {nm}: flexion mean {flex.mean():.0f} deg  p10/p90 {np.percentile(flex, 10):.0f}/{np.percentile(flex, 90):.0f}")
d = joblib.load("/tmp/gmr_test_ml/dance2_subject3.pkl")
r = d[next(iter(d))]
k = np.degrees(np.asarray(r["dof"], dtype=np.float64)[:, [3, 9]])
print(f"G1 retarget knee: mean {k.mean():.0f}  p10/p90 {np.percentile(k, 10):.0f}/{np.percentile(k, 90):.0f}")
# thigh/shin lengths in LAFAN vs G1 knee-bend needed
print("LAFAN thigh/shin length:", np.median(np.linalg.norm(sj[:, 1] - sj[:, 4], axis=1)).round(3), np.median(np.linalg.norm(sj[:, 4] - sj[:, 7], axis=1)).round(3))
