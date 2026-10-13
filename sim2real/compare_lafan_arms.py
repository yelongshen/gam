"""Compare LAFAN robot data (old `robot/` vs regenerated `robot_fixed/`) against eval_subset:
frame-0 root pose / heading, arm + wrist joint statistics, and the arm-retarget detector features.
Run with .venv_sim (needs mujoco)."""
import glob
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_arm_retarget_errors import FEATS, G1, HW, qc_clip  # noqa: E402

R = os.path.expanduser("~/ego_dataset")
ARM = [HW.index(n) for n in HW if any(k in n for k in ("shoulder", "elbow", "wrist"))]
WRIST = [HW.index(n) for n in HW if "wrist" in n]


def first(d):
    return d[next(iter(d))] if isinstance(d, dict) and "dof" not in d else d


def quat_info(q):
    """q xyzw -> (tilt_deg, yaw_deg)."""
    x, y, z, w = q
    zb = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    tilt = np.degrees(np.arccos(np.clip(zb[2], -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return tilt, yaw


def stats(files, label, g, stride):
    rows = []
    for f in files:
        r = first(joblib.load(f))
        dof = np.asarray(r["dof"], dtype=np.float64)
        root = np.asarray(r["root_trans_offset"], dtype=np.float64)
        rot = np.asarray(r["root_rot"], dtype=np.float64)
        d = np.degrees(dof)
        t0, y0 = quat_info(rot[0])
        step = np.abs(np.diff(d[:, ARM], axis=0)).max(1)
        lo, hi = g.lim[:, 0], g.lim[:, 1]
        rng = hi - lo
        at = ((dof < lo + 0.02 * rng) | (dof > hi - 0.02 * rng))[:, ARM].mean()
        rows.append(dict(
            xy0=np.linalg.norm(root[0, :2]), z0=root[0, 2], tilt0=t0, yaw0=y0,
            arm0=np.abs(d[0, ARM]).max(), wrist_rom=(d[:, WRIST].max(0) - d[:, WRIST].min(0)).mean(),
            wrist_speed=np.abs(np.diff(d[:, WRIST], axis=0)).mean() * 30,
            arm_speed=np.abs(np.diff(d[:, ARM], axis=0)).mean() * 30,
            maxstep=step.max(), n90=int((step > 90).sum()), atlim=at))
    print(f"\n== {label}: {len(rows)} clips")
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        print(f"  {k:12s} min {v.min():8.2f}  med {np.median(v):8.2f}  p90 {np.percentile(v, 90):8.2f}  max {v.max():8.2f}")
    return rows


def main():
    g = G1()
    sets = [
        ("eval_subset (reference format)", sorted(glob.glob(f"{R}/eval_subset/robot/*.pkl"))[::4]),
        ("lafan eval  robot_fixed (SOMA)", sorted(glob.glob(f"{R}/lafan1_evalset/robot_fixed/*.pkl"))),
        ("lafan eval  robot_v2 (GMR)", sorted(glob.glob(f"{R}/lafan1_evalset/robot_v2/*.pkl"))),
        ("lafan train robot_v2 (GMR)", sorted(glob.glob(f"{R}/lafan1_trainset/robot_v2/*.pkl"))),
    ]
    for label, files in sets:
        stats(files, label, g, 1)
    print("\n=== joints pinned at a limit (mean fraction of frames over clips; only >5% shown) ===")
    for label, files in sets:
        acc = np.zeros(29)
        for f in files:
            dof = np.asarray(first(joblib.load(f))["dof"], dtype=np.float64)
            lo, hi = g.lim[:, 0], g.lim[:, 1]
            acc += ((dof < lo + 0.02 * (hi - lo)) | (dof > hi - 0.02 * (hi - lo))).mean(0)
        acc /= len(files)
        print(f"  {label:34s} " + str({HW[i]: round(float(acc[i]), 2) for i in range(29) if acc[i] > 0.05}))
    return
    print("\n=== arm-retarget detector features (stride 10) ===")
    for label, sdir, rdir in [("eval_subset", "eval_subset/smpl", "eval_subset/robot"),
                              ("lafan eval fixed", "lafan1_evalset/smpl", "lafan1_evalset/robot_fixed"),
                              ("lafan train fixed", "lafan1_trainset/smpl", "lafan1_trainset/robot_fixed")]:
        fs = sorted(glob.glob(f"{R}/{rdir}/*.pkl"))
        fs = fs[::4] if label == "eval_subset" else fs
        acc = {k: [] for k in FEATS}
        for f in fs:
            n = os.path.basename(f)[:-4]
            try:
                m = qc_clip(g, f"{R}/{sdir}/{n}.pkl", f, 10)
            except Exception as exc:
                print("  skip", n, exc)
                continue
            for k in FEATS:
                acc[k].append(m[k])
        print(f"-- {label} ({len(acc[FEATS[0]])} clips): median / p90")
        for k in FEATS:
            v = np.array(acc[k])
            print(f"   {k:20s} {np.median(v):8.3f} / {np.percentile(v, 90):8.3f}")


if __name__ == "__main__":
    main()
