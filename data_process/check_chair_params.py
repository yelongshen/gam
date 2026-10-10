import os, sys, glob, csv
import numpy as np
os.environ["VIS_ROBOT_DIR"] = "/home/grease/GMR/picoset_20261002_sit_retargeted_g1_fps30"
os.environ["VIS_SMPL_DIR"] = "/home/grease/ego_dataset/picoset_20261002_sit/smpl"
sys.path.insert(0, "/home/grease/gam/gear_sonic/scripts")
from visualize_smpl_vs_robot import load_robot
rows = list(csv.DictReader(open("/home/grease/ego_dataset/picoset_20261002_sit/envs/chair_params.csv")))
sh = np.array([float(r["seat_h"]) for r in rows])
print("seat_h bands: <0.40:", int((sh < 0.40).sum()), " 0.40-0.45:", int(((sh >= 0.40) & (sh < 0.45)).sum()), " 0.45-0.50:", int(((sh >= 0.45) & (sh < 0.50)).sum()), " 0.50-0.56:", int((sh >= 0.50).sum()),
      " | percentiles 0/25/50/75/100:", np.round(np.percentile(sh, [0, 25, 50, 75, 100]), 3))
dang = []
for r in rows:
    R, nm, fps = load_robot(r["clip"])
    feet = [nm.index(k) for k in ("left_toe_link", "right_toe_link", "left_ankle_roll_link", "right_ankle_roll_link")]
    pel = R[:, nm.index("pelvis"), 2]
    seated = pel < pel.min() + 0.06
    dang.append(np.median(R[seated][:, feet, 2].min(1)))
dang = np.array(dang)
print("robot lowest foot-link height above the floor while seated (m): median %.3f  min %.3f  max %.3f  (clips > 0.05: %d/76)" % (np.median(dang), dang.min(), dang.max(), int((dang > 0.05).sum())))
yaw = np.array([float(r["yaw"]) for r in rows]); print("yaw deg range", np.degrees(yaw).min().round(0), np.degrees(yaw).max().round(0))
