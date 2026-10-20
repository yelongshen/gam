"""analyze_policy_walk.py - does the terrain map agree with the feet while a policy walks?

Takes a walk recorded with the deploy runner up (WALKING_TEST.md; gantry runs 2026-10-20):

  terrain.npz    record_terrain.py: every policy-facing terrain message (grid, validity, torso pose)
  g1_debug.npz   record_zmq_raw.py --port 5560 --topic g1_debug: the runner's state stream (msgpack),
                 with the runner started with --zmq-out-port 5560

and puts each sole (forward kinematics of the measured joints, G1 29-DoF MJCF) on the map the policy
saw at that tick, in the torso-centred, yaw-aligned grid frame. The floor is flat here, so the
supporting (lower) sole should sit on the observed floor around it.

    python3 analyze_policy_walk.py walk_terrain.npz walk_g1_debug.npz \\
        [--window START,END] [--exclude START,END;START,END] [--match 0.06]

Times are seconds since the epoch, as in the recordings' recv_time. Reports:
  - map and pose age before and during control, and gaps in the terrain stream;
  - torso motion (odometry);
  - per sole: speed, height range, ticks with >= 5 observed cells within 0.4 m, and the distance to
    the nearest observed cell (the blind zone under the feet: 3-11 cm in runs 8-9);
  - supporting sole - median observed floor within 0.4 m, all ticks and single support only
    (other foot >= 3 cm higher). Run 9: -2.0 cm median (p10/p90 -4.4/-0.8).

Needs numpy, msgpack and mujoco (not on the Jetson's system Python: run it off the robot).
"""
import argparse
import re
from pathlib import Path

import msgpack
import mujoco
import numpy as np

MJCF = (Path(__file__).resolve().parents[2] / "decoupled_wbc/dexmg/gr00trobocasa/robocasa/models/assets/"
        "robots/unitree_g1/g1_29dof_rev_1_0.xml")
SOLE = np.array([0.035, 0.0, -0.033])      # middle of the sole, in the ankle roll link frame
NEAR = 0.4                                 # observed floor within this radius of the sole (m)


def kinematic_model(path):
    """The MJCF without meshes: only kinematics are needed, and the meshes may not be checked out."""
    s = Path(path).read_text()
    s = re.sub(r"<mesh [^>]*/>", "", s)
    s = re.sub(r'<geom [^>]*mesh="[^"]*"[^>]*/>', "", s)
    return mujoco.MjModel.from_xml_string(s)


def quat_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def spans(text):
    return [tuple(float(x) for x in s.split(",")) for s in filter(None, (text or "").split(";"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("terrain")
    ap.add_argument("g1_debug")
    ap.add_argument("--window", help="START,END: only control ticks in this window (s since the epoch)")
    ap.add_argument("--exclude", help="START,END;...: drop these spans (e.g. odometry glitches)")
    ap.add_argument("--match", type=float, default=0.03, help="max |terrain - control| time (s)")
    ap.add_argument("--mjcf", default=str(MJCF))
    a = ap.parse_args()

    T = dict(np.load(a.terrain))
    G = dict(np.load(a.g1_debug))
    data, off = G["data"].tobytes(), G["offsets"]
    st = [msgpack.unpackb(data[s:e][len(b"g1_debug"):], raw=False) for s, e in zip(off[:-1], off[1:])]
    q, t_ctl = np.array([m["body_q"] for m in st]), G["recv_time"]
    keep = np.ones(len(t_ctl), bool)
    for lo, hi in spans(a.window):
        keep &= (t_ctl >= lo) & (t_ctl <= hi)
    for lo, hi in spans(a.exclude):
        keep &= ~((t_ctl >= lo) & (t_ctl <= hi))
    q, t_ctl = q[keep], t_ctl[keep]
    if not len(t_ctl):
        raise SystemExit("no control ticks in the window")

    r_t = T["recv_time"]
    ctl = (r_t >= t_ctl[0]) & (r_t <= t_ctl[-1])
    age, page = T["timestamp"] - T["map_stamp"], T["timestamp"] - T["pose_stamp"]
    for name, m in (("before control", r_t < t_ctl[0] - 1), ("during control", ctl)):
        if m.any():
            print(f"{name:15s}: {m.sum():5d} msgs, map age median {np.median(age[m]):.3f} "
                  f"p95 {np.percentile(age[m], 95):.3f} max {age[m].max():.3f} s; pose age median "
                  f"{np.median(page[m]) * 1000:.0f} max {page[m].max() * 1000:.0f} ms")
    gap = np.diff(r_t[ctl])
    print(f"terrain message gaps during control: median {np.median(gap) * 1000:.1f} ms, max {gap.max() * 1000:.0f} ms")
    tp = T["torso_pos"][ctl]
    print(f"torso (odometry) during control: z range {np.ptp(tp[:, 2]) * 100:.1f} cm, "
          f"xy range {np.ptp(tp[:, 0]) * 100:.1f} / {np.ptp(tp[:, 1]) * 100:.1f} cm")

    mdl = kinematic_model(a.mjcf)
    d = mujoco.MjData(mdl)
    bid = {n: mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_BODY, n)
           for n in ("torso_link", "left_ankle_roll_link", "right_ankle_roll_link")}

    def soles_in_torso(qj):
        d.qpos[:] = 0
        d.qpos[3] = 1
        d.qpos[7:7 + 29] = qj
        mujoco.mj_kinematics(mdl, d)
        Rt, pt = d.xmat[bid["torso_link"]].reshape(3, 3), d.xpos[bid["torso_link"]]
        out = {}
        for s in ("left", "right"):
            b = bid[f"{s}_ankle_roll_link"]
            out[s] = Rt.T @ (d.xpos[b] + d.xmat[b].reshape(3, 3) @ SOLE - pt)
        return out

    res = {"left": [], "right": []}          # (tick, sole - floor, nearest observed cell distance)
    soles = {"left": [], "right": []}
    o, rs = T["grid_origin"], float(T["grid_resolution"])
    for k in range(len(t_ctl)):
        i = int(np.argmin(np.abs(r_t - t_ctl[k])))
        if abs(r_t[i] - t_ctl[k]) > a.match:
            continue
        p, R = T["torso_pos"][i], quat_R(*T["torso_quat"][i])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s_ = np.cos(-yaw), np.sin(-yaw)
        H, V = T["height_grid"][i], T["height_grid_valid"][i].astype(bool)
        rr, cc = np.nonzero(V)
        for side, v in soles_in_torso(q[k]).items():
            w = p + R @ v                                     # sole in the odometry frame
            lx = c * (w[0] - p[0]) - s_ * (w[1] - p[1])       # ... in the yaw-aligned grid frame
            ly = s_ * (w[0] - p[0]) + c * (w[1] - p[1])
            soles[side].append((k, w))
            dist = np.hypot(o[0] + cc * rs - lx, o[1] + rr * rs - ly)
            near = dist < NEAR
            if near.sum() >= 5:
                res[side].append((k, (w[2] - p[2]) - np.median(H[rr[near], cc[near]]), dist[near].min()))

    for side in ("left", "right"):
        S = np.array([w for _, w in soles[side]])
        if len(S) < 2:
            raise SystemExit(f"{len(S)} {side} sole samples matched a terrain message: widen --match")
        sp = np.linalg.norm(np.gradient(S, axis=0) * 50, axis=1)
        print(f"{side:5s} sole: speed percentiles 10/50/90 {np.round(np.percentile(sp, [10, 50, 90]), 2)} m/s, "
              f"height range {np.ptp(S[:, 2]) * 100:.1f} cm; >= 5 observed cells within {NEAR} m in "
              f"{len(res[side])}/{len(S)} ticks")
    for side in ("left", "right"):
        if res[side]:
            print(f"{side:5s}: nearest observed map cell {np.median([r[2] for r in res[side]]) * 100:.0f} cm "
                  f"from the sole (median)")

    zl = dict((k, w[2]) for k, w in soles["left"])
    zr = dict((k, w[2]) for k, w in soles["right"])
    gl = dict((r[0], r[1]) for r in res["left"])
    gr = dict((r[0], r[1]) for r in res["right"])
    both = sorted(set(gl) & set(gr))
    if not both:
        raise SystemExit("no tick with observed floor near both soles")
    lower = np.array([gl[k] if zl[k] < zr[k] else gr[k] for k in both]) * 100
    single = np.array([abs(zl[k] - zr[k]) > 0.03 for k in both])
    print(f"lower (supporting) sole - observed floor within {NEAR} m: {len(lower)} ticks, median "
          f"{np.median(lower):+.1f} cm, p10/p90 {np.percentile(lower, 10):+.1f}/{np.percentile(lower, 90):+.1f} cm")
    if single.any():
        x = lower[single]
        print(f"  only ticks where the other foot is >= 3 cm higher (single support, {single.sum()} ticks): "
              f"median {np.median(x):+.1f} cm, p10/p90 {np.percentile(x, 10):+.1f}/{np.percentile(x, 90):+.1f} cm")


if __name__ == "__main__":
    main()
