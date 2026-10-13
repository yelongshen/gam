import os
import pickle
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.expanduser("~/gam/sim2real"))
from detect_arm_retarget_errors import G1, HW, angle  # noqa: E402

d = pickle.load(open("/tmp/gmr_test/dance2_subject3.pkl", "rb"))
names = d["link_body_list"]
lbp = d["local_body_pos"]  # (T,N,3) root-local (identity root)
dof = d["dof_pos"]
s = joblib.load(os.path.expanduser("~/ego_dataset/lafan1_smpl_filtered_FPS30/dance2_subject3.pkl"))
sj = np.asarray(s["smpl_joints"], dtype=np.float64)[3600:4500]
print([n for n in names if "shoulder" in n or "elbow" in n or "wrist" in n][:12])
g = G1()
print("t | SMPL elbow interior (deg) | G1 elbow joint flexion | 180-flex | FK angle(sh_pitch,elbow,wrist_yaw) | FK angle(sh_yaw,elbow,wrist_yaw)")
for t in range(0, 900, 75):
    es = angle(sj[t:t + 1, 16], sj[t:t + 1, 18], sj[t:t + 1, 20])[0]
    fl = np.degrees(dof[t, HW.index("left_elbow")])
    p = {n: lbp[t, i] for i, n in enumerate(names)}
    a1 = angle(p["left_shoulder_pitch_link"][None], p["left_elbow_link"][None], p["left_wrist_yaw_link"][None])[0]
    a2 = angle(p["left_shoulder_yaw_link"][None], p["left_elbow_link"][None], p["left_wrist_yaw_link"][None])[0]
    print(f"{t:4d} | {es:6.1f} | {fl:6.1f} | {180 - fl:6.1f} | {a1:6.1f} | {a2:6.1f}")
