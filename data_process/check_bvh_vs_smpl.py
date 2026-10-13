"""Verify a converted SOMA BVH against its source smpl_joints: bone-direction error per bone + pelvis frame."""
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/gam/data_process"))
import convert_smpl_filtered_to_bvh as V  # noqa: E402
import soma_bvh as S  # noqa: E402

bvh_path, pkl_path = sys.argv[1], sys.argv[2]
f0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
b = S.parse_bvh(bvh_path)
d = joblib.load(pkl_path)
sj = np.asarray(d["smpl_joints"], dtype=np.float64)[f0:f0 + len(b["frames"])]
sj = np.stack([sj[..., 0], sj[..., 2], -sj[..., 1]], axis=-1)
names = V.SMPL_TO_SOMA
errs = {}
for t in range(0, len(b["frames"]), max(1, len(b["frames"]) // 30)):
    Rw, Pw = S.fk(b, b["frames"][t])
    for p, c in V.PRIMARY_CHILD.items():
        pn = names.get(p, "Hips" if p == 0 else None)
        cn = names.get(c)
        if pn is None or cn is None:
            continue
        v = Pw[cn] - Pw[pn]
        w = sj[t, c] - sj[t, p]
        ang = np.degrees(np.arccos(np.clip(v @ w / (np.linalg.norm(v) * np.linalg.norm(w)), -1, 1)))
        errs.setdefault(f"{pn}->{cn}", []).append(ang)
print("bone-direction error (deg), mean / max over sampled frames")
for k, v in errs.items():
    print(f"  {k:24s} {np.mean(v):6.2f} / {np.max(v):6.2f}")
t = 0
Rw, Pw = S.fk(b, b["frames"][t])
print("frame0 pelvis world rotation columns (BVH):\n", np.round(Rw["Hips"], 2))
