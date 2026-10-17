"""Compare retarget variants on the faulty clips: robot lean vs human lean (first 2 s) and first-frame reference speed.
Reads the raw retarget pkl files written by retarget_warm_start.py (/tmp/rt_variants/<variant>/<clip>.pkl)."""
import glob
import os

import pickle

import joblib
import numpy as np

DS = "/home/grease/ego_dataset/picoset_20261015"
SMPL = "/home/grease/GMR/picoset_20261015_smplx_npz_fps30"
ORIG = "/home/grease/GMR/picoset_20261015_retargeted_g1_fps30"


def unwrap(d):
    return d[next(iter(d))] if "dof" not in d else d


def lean(q):
    x, y, z, w = q.T
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


def metrics(pkl, clip):
    r = pickle.load(open(pkl, "rb"))  # raw retarget_warm_start.py output: fps, root_pos, root_rot (xyzw), dof_pos
    s = joblib.load(f"{DS}/smpl/{clip}.pkl")
    s = s if "smpl_joints" in s else s[next(iter(s))]
    fps = float(r.get("fps", 30.0))
    J = np.asarray(s["smpl_joints"], float)
    v = J[:100, 9] - J[:100, 0]
    hl = float(np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))).mean())
    rl = float(lean(np.asarray(r["root_rot"])[: int(2 * fps)]).mean())
    spd0 = float(np.abs(r["dof_pos"][1] - r["dof_pos"][0]).max() * fps)
    return rl - hl, spd0, float(np.asarray(r["root_pos"])[:, 2].min())


clips = ["pico1015_161659_clip_022", "pico1015_161659_clip_017", "pico1015_160754_clip_014", "pico1015_161659_clip_014"]
variants = ["current (in dataset)"] + sorted(d for d in os.listdir("/tmp/rt_variants") if os.path.isdir(f"/tmp/rt_variants/{d}"))
print(f"{'clip':28s}" + "".join(f"{v[:22]:>24s}" for v in variants))
print(f"{'':28s}" + "".join(f"{'lean diff | spd0 rad/s':>24s}" for _ in variants))
for c in clips:
    cells = []
    for v in variants:
        p = f"{ORIG}/{c}.pkl" if v.startswith("current") else f"/tmp/rt_variants/{v}/{c}.pkl"
        if os.path.exists(p):
            a, b, _ = metrics(p, c)
            cells.append(f"{a:+7.1f} | {b:6.2f}")
        else:
            cells.append("missing")
    print(f"{c:28s}" + "".join(f"{x:>24s}" for x in cells))
