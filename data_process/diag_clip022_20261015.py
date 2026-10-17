"""Why does pico1015_161659_clip_022 retarget with an 80 deg robot lean while the human is upright?  Print time series of human lean (skeleton),
SMPL-X root orientation tilt (npz), and robot root tilt (retarget) at a few instants."""
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

clip = sys.argv[1] if len(sys.argv) > 1 else "pico1015_161659_clip_022"
npz = np.load(f"/home/grease/GMR/picoset_20261015_smplx_npz_fps30/{clip}.npz", allow_pickle=True)
print("npz keys:", {k: npz[k].shape for k in npz.files})
go = npz["global_orient"] if "global_orient" in npz.files else npz["pose_body"][:, :3]
pose = npz["poses"] if "poses" in npz.files else None
root_aa = pose[:, :3] if pose is not None else go
R = Rot.from_rotvec(root_aa)
up = R.apply([0, 0, 1])
tilt_npz = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
r = pickle.load(open(f"/home/grease/GMR/picoset_20261015_retargeted_g1_fps30/{clip}.pkl", "rb"))
q = np.asarray(r["root_rot"])
x, y, z, w = q.T
tilt_rob = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
sj = None
import joblib

s = joblib.load(f"/home/grease/ego_dataset/picoset_20261015/smpl/{clip}.pkl")
s = s if "smpl_joints" in s else s[next(iter(s))]
J = np.asarray(s["smpl_joints"], float)
v = J[:, 9] - J[:, 0]
lean_h = np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1)))
print(f"frames: npz {len(root_aa)} @30, robot {len(q)} @30, smpl {len(J)} @50")
print("t(s)   human lean(skeleton)  smplx-root tilt   robot root tilt   robot dof speed max")
for ts in (0, 0.5, 1, 2, 3, 5, 8, 12, 15):
    i30 = min(int(ts * 30), len(q) - 1)
    i50 = min(int(ts * 50), len(J) - 1)
    print(f"{ts:4.1f}   {lean_h[i50]:8.1f}             {tilt_npz[min(i30, len(tilt_npz) - 1)]:8.1f}          {tilt_rob[i30]:8.1f}")
print("fraction of frames with robot tilt > 45:", float((tilt_rob > 45).mean()), " human lean > 45:", float((lean_h > 45).mean()))
print("root_pos z first/mid/last:", np.round(np.asarray(r["root_pos"])[[0, len(q) // 2, -1], 2], 2).tolist())
