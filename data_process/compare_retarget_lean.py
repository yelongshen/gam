import os, sys, glob
os.environ["VIS_SMPL_DIR"] = "/home/grease/ego_dataset/picoset_20261002_sit/smpl"
sys.path.insert(0, "/home/grease/gam/gear_sonic/scripts")
import numpy as np
import visualize_smpl_vs_robot as V
new = sys.argv[1]; old = "/home/grease/GMR/picoset_20261002_sit_retargeted_g1_fps30"
def m(d, c):
    V.ROBOT_DIR = d
    R, nm, fps = V.load_robot(c)
    v = R[:60, nm.index("torso_link")] - R[:60, nm.index("pelvis")]
    lean = np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))).mean()
    ft = [nm.index("left_ankle_roll_link"), nm.index("right_ankle_roll_link")]
    low = R[:, ft, 2].min(1); pel = R[:, nm.index("pelvis"), 2]
    # lowest foot over the standing/walking frames and pelvis there
    st = pel > 0.74
    return lean, float(np.median(low[st])) if st.any() else np.nan, float(np.median(pel[st])) if st.any() else np.nan
print("clip      torso lean first 2 s [deg] | median lowest ankle z while standing | median standing pelvis z   (old -> new)")
for f in sorted(glob.glob(new + "/*.pkl")):
    c = os.path.basename(f)[:-4]; a, b = m(old, c), m(new, c)
    print(f"{c[-8:]}  {a[0]:5.1f} -> {b[0]:5.1f} | {a[1]:.3f} -> {b[1]:.3f} | {a[2]:.3f} -> {b[2]:.3f}")
