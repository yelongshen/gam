import os, sys, glob, json
import numpy as np
os.environ["VIS_ROBOT_DIR"] = "/home/grease/GMR/picoset_20261002_sit_retargeted_g1_fps30"
os.environ["VIS_SMPL_DIR"] = "/home/grease/ego_dataset/picoset_20261002_sit/smpl"
sys.path.insert(0, "/home/grease/gam/gear_sonic/scripts")
from visualize_smpl_vs_robot import load_robot
D = "/home/grease/ego_dataset/picoset_20261002_sit/envs"
js = [json.load(open(f)) for f in sorted(glob.glob(D + "/pico1002sit_clip_*.json"))]
sh = np.array([j["seat_center"][2] for j in js])
print("seat top height after contact fit: mean %.3f median %.3f  range [%.3f, %.3f]" % (sh.mean(), np.median(sh), sh.min(), sh.max()))
print("bands: <0.30:", int((sh < 0.30).sum()), "0.30-0.35:", int(((sh >= 0.30) & (sh < 0.35)).sum()), "0.35-0.40:", int(((sh >= 0.35) & (sh < 0.40)).sum()),
      "0.40-0.45:", int(((sh >= 0.40) & (sh < 0.45)).sum()), ">=0.45:", int((sh >= 0.45).sum()),
      "| sources:", sorted(set(j.get("seat_h_source", "?") for j in js)))
dang, lowest = [], []
for j in js:
    R, nm, fps = load_robot(j["clip"])
    feet = [nm.index(k) for k in ("left_toe_link", "right_toe_link", "left_ankle_roll_link", "right_ankle_roll_link")]
    pel = R[:, nm.index("pelvis"), 2]
    seated = pel < pel.min() + 0.06
    dang.append(np.median(R[seated][:, feet, 2].min(1)))
dang = np.array(dang)
print("robot feet while seated: lowest foot-link height above the floor, median %.3f  p90 %.3f  max %.3f ; > 0.05 m on %d/76, > 0.10 m on %d/76" %
      (np.median(dang), np.percentile(dang, 90), dang.max(), int((dang > 0.05).sum()), int((dang > 0.10).sum())))
print("example json:", {k: js[0][k] for k in ("clip", "seat_center", "seat_yaw", "seat_h_source", "seat_h_inferred")})
