import os, sys, glob
import numpy as np
sys.path.insert(0, "/home/grease/gam/gear_sonic/scripts")
import visualize_smpl_vs_robot as V
def metrics(d, clip):
    V.ROBOT_DIR = d
    R, nm, fps = V.load_robot(clip)
    ft = [nm.index("left_ankle_roll_link"), nm.index("right_ankle_roll_link")]
    low = R[:, ft, 2].min(1); pel = R[:, nm.index("pelvis"), 2]
    n = int(3.0 * fps)
    return float(np.median(low[:n])), float(pel[:n].max()), float(low[:n].max()), float(R[:n, ft[1], 2].max())
old = "/home/grease/GMR/picoset_20261002_sit_retargeted_g1_fps30"
new = sys.argv[1]
print("first 3 s of the clip: median lowest ankle z | max pelvis z | max lowest-foot z | max RIGHT foot z")
for f in sorted(glob.glob(new + "/*.pkl")):
    c = os.path.basename(f)[:-4]
    a, b = metrics(old, c), metrics(new, c)
    print(f"{c[-8:]}  old: {a[0]:.3f} {a[1]:.3f} {a[2]:.3f} {a[3]:.3f}   new: {b[0]:.3f} {b[1]:.3f} {b[2]:.3f} {b[3]:.3f}")
