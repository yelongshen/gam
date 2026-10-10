import sys
from pathlib import Path
import joblib, numpy as np
PELV = np.array([0.003, -0.351, 0.012])
for ds, k in (("picoset_20260928", "pico0928_clip_018"), ("picoset_20260930", "pico0930_145524_clip_034"), ("picoset_20260930", "pico0930_145524_clip_021")):
    s = joblib.load(f"/home/grease/ego_dataset/{ds}/smpl/{k}.pkl"); s = s[next(iter(s))] if "smpl_joints" not in s else s
    J = np.asarray(s["smpl_joints"], float) - PELV + np.asarray(s["transl"], float)[:, None, :]
    toe = J[:, [10, 11], 2].min(1); pel = J[:, 0, 2]; fps = float(s["fps"])
    i = (np.arange(0, 9, 0.5) * fps).astype(int); i = i[i < len(toe)]
    print(k, "frames", len(toe), "fps", fps)
    print("  t   ", " ".join(f"{x/fps:5.1f}" for x in i))
    print("  toe ", " ".join(f"{toe[x]:5.2f}" for x in i))
    print("  pel ", " ".join(f"{pel[x]:5.2f}" for x in i))
E = None
