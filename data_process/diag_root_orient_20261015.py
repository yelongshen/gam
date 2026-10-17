"""Compare the SMPL-X root orientation of a failing clip and good clips: which body axis points up in the first frames, and the yaw-normalisation heading."""
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

for clip in sys.argv[1:]:
    npz = np.load(f"/home/grease/GMR/picoset_20261015_smplx_npz_fps30/{clip}.npz", allow_pickle=True)
    R = Rot.from_rotvec(npz["root_orient"])
    out = [f"{clip:26s}"]
    for ax, nm in (([1, 0, 0], "x"), ([0, 1, 0], "y"), ([0, 0, 1], "z")):
        v = R.apply(ax)
        out.append(f"{nm}-axis angle to world-up: first {np.degrees(np.arccos(np.clip(v[0, 2], -1, 1))):5.0f}  median {np.median(np.degrees(np.arccos(np.clip(v[:, 2], -1, 1)))):5.0f}")
    fwd = R.apply([1, 0, 0])[0]
    out.append(f"fwd(x) first frame = ({fwd[0]:+.2f},{fwd[1]:+.2f},{fwd[2]:+.2f})  yaw0 {np.degrees(np.arctan2(fwd[1], fwd[0])):+.0f} deg")
    print(" | ".join(out))
