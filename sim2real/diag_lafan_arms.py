import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_arm_retarget_errors import HW, G1, SMPL, angle  # noqa: E402

R = os.path.expanduser("~/ego_dataset")
g = G1()


def first(d):
    return d[next(iter(d))] if isinstance(d, dict) and "dof" not in d else d


# 1) which joints sit at a limit, LAFAN fixed vs eval_subset
for label, pat in [("lafan eval fixed", f"{R}/lafan1_evalset/robot_fixed"), ("eval_subset", f"{R}/eval_subset/robot")]:
    acc = np.zeros(29)
    n = 0
    for fn in sorted(os.listdir(pat))[:: (1 if "lafan" in label else 4)]:
        r = first(joblib.load(f"{pat}/{fn}"))
        dof = np.asarray(r["dof"], dtype=np.float64)
        lo, hi = g.lim[:, 0], g.lim[:, 1]
        acc += ((dof < lo + 0.02 * (hi - lo)) | (dof > hi - 0.02 * (hi - lo))).mean(0)
        n += 1
    acc /= n
    print(f"\n{label}: joints at a limit on >5% of frames (mean over clips):")
    print({HW[i]: round(float(acc[i]), 2) for i in range(29) if acc[i] > 0.05})

# 2) time series on one clip: SMPL elbow angle vs robot elbow angle, arm joint values
for name, split, rdir in [("dance2_subject3", "lafan1_evalset", "robot_fixed"),
                          ("walk_180_R_003__A332_M", "eval_subset", "robot")]:
    s = joblib.load(f"{R}/{split}/smpl/{name}.pkl")
    r = first(joblib.load(f"{R}/{split}/{rdir}/{name}.pkl"))
    sj = np.asarray(s["smpl_joints"], dtype=np.float64)
    sfps, rfps = float(s["fps"]), float(r["fps"])
    dof = np.asarray(r["dof"], dtype=np.float64)
    pts = g.arm_points(dof[::30])
    print(f"\n=== {name}: t | SMPL elbow L/R | robot elbow L/R | robot L: sh_p sh_r sh_y elbow wr_r | R: sh_p sh_r sh_y elbow wr_r")
    for i in range(0, len(pts), max(1, len(pts) // 14)):
        t = i * 30 / rfps
        k = min(int(t * sfps), len(sj) - 1)
        es = [angle(sj[k:k + 1, a], sj[k:k + 1, b], sj[k:k + 1, c])[0] for (a, b, c) in (SMPL["L"], SMPL["R"])]
        er = [angle(pts[i:i + 1, si, 0], pts[i:i + 1, si, 1], pts[i:i + 1, si, 2])[0] for si in (0, 1)]
        d = np.degrees(dof[i * 30])
        L = [d[HW.index(f"left_{j}")] for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")]
        Rr = [d[HW.index(f"right_{j}")] for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")]
        print(f"{t:6.1f}s | {es[0]:5.0f} {es[1]:5.0f} | {er[0]:5.0f} {er[1]:5.0f} | "
              + " ".join(f"{v:5.0f}" for v in L) + " | " + " ".join(f"{v:5.0f}" for v in Rr))
