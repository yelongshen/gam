import numpy as np
base = "/home/grease/GMR/picoset_20261002_sit_smplx_npz_fps30/pico1002sit_clip_%s.npz"
pkl30 = "/home/grease/gam/logs_pkl/picoset_20261002_sit_FPS30/pico1002sit_clip_%s.pkl"
import joblib
for cid in ("024", "032", "005", "010"):
    d = np.load(base % cid)
    t = d["trans"]
    p = joblib.load(pkl30 % cid); p = p[next(iter(p))] if "transl" not in p else p
    tz = np.asarray(p["transl"])[:, 2]
    i = np.arange(0, 100, 10)
    print(cid, "npz trans z   :", np.round(t[i, 2], 3))
    print(cid, "FPS30 pkl z   :", np.round(tz[i], 3), " frames", len(tz), len(t))
    print(cid, "root_orient aa:", np.round(d["root_orient"][0], 2), "pose_body[0,:6]", np.round(d["pose_body"][0, :6], 2))
