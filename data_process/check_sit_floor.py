import os, sys, glob
os.environ["VIS_SMPL_DIR"] = "/home/grease/ego_dataset/picoset_20261002_sit/smpl"
os.environ["VIS_ROBOT_DIR"] = "/home/grease/GMR/picoset_20261002_sit_retargeted_g1_fps30"
sys.path.insert(0, "/home/grease/gam/gear_sonic/scripts")
import numpy as np
from visualize_smpl_vs_robot import load_robot

off, pel_stand, pel_seat = [], [], []
for f in sorted(glob.glob(os.environ["VIS_ROBOT_DIR"] + "/*.pkl")):
    n = os.path.splitext(os.path.basename(f))[0]
    R, nm, fps = load_robot(n)
    toe = [nm.index("left_toe_link"), nm.index("right_toe_link"), nm.index("left_ankle_roll_link"), nm.index("right_ankle_roll_link")]
    low = R[:, toe, 2].min(1)
    pel = R[:, nm.index("pelvis"), 2]
    off.append((np.percentile(low, 5), low.min(), np.median(low[pel > 0.8]) if (pel > 0.8).any() else np.nan))
    pel_stand.append(np.median(pel[pel > 0.8])); pel_seat.append(np.median(pel[pel < 0.7]))
off = np.array(off)
print("raw robot frame (as stored by GMR), lowest foot-link z per clip:")
print("  p5 of lowest foot z while any: median %.3f range [%.3f, %.3f]" % (np.median(off[:, 0]), off[:, 0].min(), off[:, 0].max()))
print("  median lowest foot z while standing (pelvis>0.8): median %.3f range [%.3f, %.3f]" % (np.nanmedian(off[:, 2]), np.nanmin(off[:, 2]), np.nanmax(off[:, 2])))
print("  standing pelvis z: median %.3f   seated pelvis z: median %.3f (range %.3f-%.3f)" % (np.median(pel_stand), np.median(pel_seat), min(pel_seat), max(pel_seat)))
