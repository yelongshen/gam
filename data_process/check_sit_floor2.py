import os, sys, glob
import numpy as np
sys.path.insert(0, "/home/grease/gam/gear_sonic/scripts")
import joblib
import visualize_smpl_vs_robot as V

def stats(files, loader):
    out = []
    for f in files:
        R, nm, fps = loader(f)
        feet = [nm.index(k) for k in ("left_toe_link", "right_toe_link", "left_ankle_roll_link", "right_ankle_roll_link")]
        low = R[:, feet, 2].min(1)
        pel = R[:, nm.index("pelvis"), 2]
        std = pel > 0.75
        out.append((np.percentile(low, 5), np.median(low[std]) if std.any() else np.nan, np.median(pel[std]) if std.any() else np.nan,
                    np.percentile(low, 50)))
    a = np.array(out)
    return a

def gmr_loader_factory(dir_):
    def ld(f):
        V.ROBOT_DIR = dir_
        return V.load_robot(os.path.splitext(os.path.basename(f))[0])
    return ld

sets = {
    "0930 (non-sit, GMR raw)": "/home/grease/GMR/picoset_20260930_retargeted_g1_fps30",
    "0929 (non-sit, GMR raw)": "/home/grease/GMR/picoset_20260929_retargeted_g1_fps30",
    "1002 sit (GMR raw)": "/home/grease/GMR/picoset_20261002_sit_retargeted_g1_fps30",
}
for name, d in sets.items():
    fs = sorted(glob.glob(d + "/*.pkl"))[:60]
    if not fs:
        print(name, "no files"); continue
    a = stats(fs, gmr_loader_factory(d))
    print(f"{name:26s} n={len(fs)}  lowest-foot z: p5 med {np.nanmedian(a[:,0]):.3f} | while pelvis>0.75 med {np.nanmedian(a[:,1]):.3f} [{np.nanmin(a[:,1]):.3f},{np.nanmax(a[:,1]):.3f}] | standing pelvis med {np.nanmedian(a[:,2]):.3f}")
