import os, sys, glob
import numpy as np
os.environ["VIS_SMPL_DIR"] = "/home/grease/gam/logs_pkl/x"
sys.path.insert(0, "/home/grease/gam/gear_sonic/scripts")
import joblib
PELVIS_OFFSET = np.array([0.003, -0.351, 0.012])
for name, d in (("0929", "/home/grease/ego_dataset/picoset_20260929/smpl"), ("0930", "/home/grease/ego_dataset/picoset_20260930/smpl"),
                ("1002 sit", "/home/grease/ego_dataset/picoset_20261002_sit/smpl"),
                ("0928", "/home/grease/ego_dataset/picoset_20260928/smpl")):
    fs = sorted(glob.glob(d + "/*.pkl"))[:60]
    if not fs:
        print(name, "no files"); continue
    st, hd, ft = [], [], []
    for f in fs:
        x = joblib.load(f)
        x = x[next(iter(x))] if not ("smpl_joints" in x) else x
        J = np.asarray(x["smpl_joints"], float) - PELVIS_OFFSET + np.asarray(x["transl"], float)[:, None, :]
        J[..., 2] -= np.percentile(J[:, [10, 11], 2].min(1), 5)
        z = J[:, 0, 2]
        std = z > 0.85
        if std.any():
            st.append(np.median(z[std])); hd.append(np.median(J[std, 15, 2]))   # head joint z while standing
    print(f"{name:9s} n={len(fs)}  standing SMPL pelvis z median {np.median(st):.3f}  head(joint15) z {np.median(hd):.3f}  (n standing clips {len(st)})")
    if "betas" in x:
        print("          betas[:3] of last clip:", np.round(np.asarray(x["betas"]).reshape(-1)[:3], 2))
