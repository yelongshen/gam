import json
import os
import sys

import joblib
import mujoco
import numpy as np

sys.path.insert(0, os.path.expanduser("~/gam/sim2real"))
from detect_arm_retarget_errors import G1  # noqa: E402

R = os.path.expanduser("~/ego_dataset/ICL_hardset_aligned")
g = G1()
m, d = g.m, g.d
man = json.load(open(os.path.expanduser("~/ego_dataset/ICL_hardset/manifest.json")))["clips"]
H = os.path.expanduser("~/gam/sim2real/icl_hard_set/icl_failure_seconds.csv")
import csv  # noqa: E402

old = {r["clip"]: r for r in csv.DictReader(open(H))}
print("frame-0 pose: lowest body z (m, ground=0), lowest foot-link z, pelvis z, root tilt (deg), base100k old failure second\n")
rows = []
for c in man:
    n = c["name"]
    rd = joblib.load(f"{R}/robot/{n}.pkl")
    r = rd[next(iter(rd))]
    dof = np.asarray(r["dof"], dtype=np.float64)[0]
    root = np.asarray(r["root_trans_offset"], dtype=np.float64)[0]
    q = np.asarray(r["root_rot"], dtype=np.float64)[0]
    d.qpos[:] = 0
    d.qpos[0:3] = root
    d.qpos[3:7] = [q[3], q[0], q[1], q[2]]
    for i, a in enumerate(g.adr):
        d.qpos[a] = dof[i]
    mujoco.mj_kinematics(m, d)
    # lowest geom surface (collision geoms of feet) approximated by lowest body origin minus nothing; use all bodies
    zs = d.xpos[1:, 2]
    foot = [d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)][2] for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[0] ** 2 + q[1] ** 2), -1, 1)))
    rows.append((n, zs.min(), min(foot), root[2], tilt, old[n]["fail_s_base100k"], c["source"]))
rows.sort(key=lambda r: -r[2])
for n, lo, ft, pz, tl, of, src in rows:
    print(f"{n[:46]:46s} lowest body {lo:6.3f}  ankle link z {ft:6.3f}  pelvis z {pz:5.2f}  tilt {tl:5.1f}  old_fail {of:>5s}  {src}")
