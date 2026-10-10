#!/usr/bin/env python
"""Frame-0 foot-contact diagnosis for the ICL hold-failure clips.

For each clip, put the robot at frame 0 and report, using the real foot collision
geometry (mesh vertices / primitive extents, not ankle-link origins):
  * sole_L / sole_R : lowest point of each foot (m above ground; <=0.005 = touching)
  * drop            : min(sole_L, sole_R) = height the robot free-falls before first contact
  * fall_t          : sqrt(2*drop/g), time to first touch
  * foot pitch/roll : foot-link tilt vs. ground (deg); large => lands on toe/heel/edge
  * n_contact_pts   : sole vertices within --contact-tol of the lowest foot point (support size)
  * com_in_support  : CoM (x,y) distance to the convex hull of the support points (m, <0 inside)
  * root tilt, root/pelvis speed from frames 0..1
Also prints the first --window frames' lowest-sole z to show if the reference itself is airborne.

    .venv_sim/bin/python sim2real/diag_frame0_contact.py
"""
import argparse
import os
import sys

import joblib
import mujoco
import numpy as np

sys.path.insert(0, os.path.expanduser("~/gam/sim2real"))
from detect_arm_retarget_errors import G1  # noqa: E402

HOLD_CLIPS = [
    "dance1_subject2__w067s", "dance2_subject4__w196s", "fight1_subject2__w219s",
    "fight1_subject3__w148s", "fightAndSports1_subject1__w062s", "jumps1_subject1__w050s",
    "jumps1_subject2__w169s", "multipleActions1_subject1__w124s",
    "multipleActions1_subject3__w036s", "run1_subject5__w035s", "run2_subject4__w205s",
]
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
G = 9.81


def geom_points(m, d, gid):
    """World-frame points describing a geom's lowest extent."""
    R = d.geom_xmat[gid].reshape(3, 3)
    p = d.geom_xpos[gid]
    t = m.geom_type[gid]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[gid]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        return m.mesh_vert[v0:v0 + nv] @ R.T + p
    s = m.geom_size[gid]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s
        return c @ R.T + p
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return p[None] - np.array([[0, 0, s[0]]])
    if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        ends = np.array([[0, 0, -s[1]], [0, 0, s[1]]]) @ R.T + p
        return np.vstack([ends + [0, 0, -s[0]]])
    return p[None]


def hull_dist(pt, pts):
    """Signed distance (m) from pt to convex hull of 2D pts (<0 inside)."""
    from scipy.spatial import ConvexHull
    if len(pts) < 3:
        return float(np.min(np.linalg.norm(pts - pt, axis=1)))
    try:
        h = ConvexHull(pts)
    except Exception:  # collinear
        return float(np.min(np.linalg.norm(pts - pt, axis=1)))
    return float(np.max(h.equations[:, :2] @ pt + h.equations[:, 2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/ego_dataset/ICL_hardset_aligned")
    ap.add_argument("--clips", nargs="*", default=HOLD_CLIPS)
    ap.add_argument("--contact-tol", type=float, default=0.02)
    ap.add_argument("--window", type=int, default=15, help="frames of lowest-sole trace to print")
    a = ap.parse_args()
    src = os.path.expanduser(a.src)

    g = G1()
    m, d = g.m, g.d
    body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in FEET]
    foot_geoms = [[i for i in range(m.ngeom) if m.geom_bodyid[i] == b] for b in body_ids]
    print("foot geoms:", [[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or i for i in gs] for gs in foot_geoms])

    def pose(r, t):
        d.qpos[:] = 0
        d.qpos[0:3] = r["root"][t]
        q = r["quat"][t]
        d.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        for i, adr in enumerate(g.adr):
            d.qpos[adr] = r["dof"][t, i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    def soles():
        pts = [np.vstack([geom_points(m, d, gi) for gi in gs]) for gs in foot_geoms]
        return pts

    print(f"\n{'clip':34s} {'soleL':>6s} {'soleR':>6s} {'drop':>5s} {'fall_t':>6s} {'pitchL':>6s} {'rollL':>5s} "
          f"{'pitchR':>6s} {'rollR':>5s} {'nct':>4s} {'com_off':>7s} {'tilt':>5s} {'v01':>5s}")
    traces = {}
    for n in a.clips:
        rd = joblib.load(f"{src}/robot/{n}.pkl")
        r = rd[next(iter(rd))] if "dof" not in rd else rd
        fps = float(r.get("fps", 30.0))
        r = {"dof": np.asarray(r["dof"], dtype=np.float64),
             "root": np.asarray(r["root_trans_offset"], dtype=np.float64),
             "quat": np.asarray(r["root_rot"], dtype=np.float64)}

        pose(r, 0)
        P = soles()
        zL, zR = P[0][:, 2].min(), P[1][:, 2].min()
        drop = max(min(zL, zR), 0.0)
        # foot orientation: local z-axis of each foot link vs world up
        ang = []
        for b in body_ids:
            Rb = d.xmat[b].reshape(3, 3)
            fwd = Rb[:, 0]  # link x ~ forward
            lat = Rb[:, 1]
            pitch = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
            roll = np.degrees(np.arcsin(np.clip(lat[2], -1, 1)))
            ang += [pitch, roll]
        allp = np.vstack(P)
        lo = allp[:, 2].min()
        sup = allp[allp[:, 2] <= lo + a.contact_tol]
        com = d.subtree_com[1] if m.nbody > 1 else d.xpos[1]
        off = hull_dist(com[:2], sup[:, :2])
        q = r["quat"][0]
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[0] ** 2 + q[1] ** 2), -1, 1)))
        v01 = np.linalg.norm(r["root"][1] - r["root"][0]) * fps
        print(f"{n[:34]:34s} {zL:6.3f} {zR:6.3f} {drop:5.3f} {np.sqrt(2 * drop / G):6.2f} "
              f"{ang[0]:6.1f} {ang[1]:5.1f} {ang[2]:6.1f} {ang[3]:5.1f} {len(sup):4d} {off:7.3f} {tilt:5.1f} {v01:5.2f}")

        tr = []
        for t in range(min(a.window, len(r["dof"]))):
            pose(r, t)
            Pt = soles()
            tr.append((Pt[0][:, 2].min(), Pt[1][:, 2].min()))
        traces[n] = tr

    print(f"\nlowest-sole z (L/R, cm) for first {a.window} frames (reference itself):")
    for n, tr in traces.items():
        print(f"{n[:34]:34s} " + " ".join(f"{100 * l:3.0f}/{100 * r:<3.0f}" for l, r in tr))
    print("\ncom_off: CoM xy distance to support hull of lowest foot points (m); >0 = CoM outside support (will tip).")


if __name__ == "__main__":
    main()
