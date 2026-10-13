import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/gam/data_process"))
import convert_smpl_filtered_to_bvh as V  # noqa: E402

for p in (os.path.expanduser("~/soma-retargeter/assets/motions/bvh/Neutral_walk_forward_002__A057.bvh"),
          os.path.expanduser("~/soma-retargeter/soma_retargeter/configs/soma/soma_zero_frame0.bvh")):
    h, joints, ch, off, n = V.parse_bvh_template(p)
    print("\n", os.path.basename(p), len(joints), "joints")
    print(" joints:", joints)

    def d(a):
        v = off[a]
        return np.round(v / np.linalg.norm(v), 2), round(float(np.linalg.norm(v)), 1)

    for name in ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Head", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
                 "RightLeg", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm"]:
        if name in off:
            print(f"  {name:14s} offset dir {d(name)[0]}  len {d(name)[1]} cm")
