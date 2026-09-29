#!/usr/bin/env python3
"""Jointly calibrate the D435 extrinsic (torso <- optical) and a per-pixel range correction against the
MID-360, from a bag with the robot held still at several heights / tilts / headings.

For every static segment (robot still >= 4 s, from DLIO's twist):
  * LiDAR scans -> torso frame (g1_frames.T_TORSO_LIVOX, trusted) -> floor plane + surface normals.
  * depth points (optical frame; pixel coords recovered from the intrinsics) are the measurements.
Model: p_torso = R (k(u, v, r) * p_opt) + t, with k = 1 + c . [1, u, v, u^2, v^2, uv, r]
Residuals: floor pixels -> distance to that segment's LiDAR floor plane (this also constrains the
near field the LiDAR cannot see); other pixels -> point-to-plane distance to the LiDAR surface.
Evaluation: leave-one-segment-out; floor height error vs horizontal distance on the held-out pose.

usage (inside the container): calib_depth.py <bag> <out.npz>
"""
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from sensor_msgs_py import point_cloud2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import g1_frames as gf

# decimated (320x240) D435 depth intrinsics, from g1_depth_publisher headers
FX = FY = 194.7845916748047
PPX, PPY, W, H = 157.80921936035156, 119.76753234863281, 320, 240
N_DEPTH_PTS = 6000       # per segment
N_DEPTH_FRAMES = 12

bag, out = sys.argv[1], sys.argv[2]
reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
types = {t.name: t.type for t in reader.get_all_topics_and_types()}
reader.set_filter(rosbag2_py.StorageFilter(topics=["/dlio/odom", "/g1/lidar/points", "/g1/depth/points"]))
odom, lidar, depth = [], [], []
while reader.has_next():
    topic, data, _ = reader.read_next()
    m = deserialize_message(data, get_message(types[topic]))
    t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    if topic == "/dlio/odom":
        q, v, w = m.pose.pose.orientation, m.twist.twist.linear, m.twist.twist.angular
        odom.append([t, q.x, q.y, q.z, q.w, np.linalg.norm([v.x, v.y, v.z]), np.linalg.norm([w.x, w.y, w.z])])
    else:
        pts = point_cloud2.read_points_numpy(m, field_names=("x", "y", "z")).astype(np.float64)
        (lidar if topic == "/g1/lidar/points" else depth).append((t, pts))
odom = np.array(odom)
t0 = odom[0, 0]

# ---- static segments ----
k = 50
spd = np.convolve(odom[:, 5], np.ones(k) / k, "same")
ang = np.convolve(odom[:, 6], np.ones(k) / k, "same")
static = (spd < 0.02) & (ang < 0.04)
segs, s0 = [], None
for i, s in enumerate(static):
    if s and s0 is None:
        s0 = i
    if (not s or i == len(static) - 1) and s0 is not None:
        if odom[i, 0] - odom[s0, 0] >= 3.0:
            segs.append((odom[s0, 0] + 0.3, odom[i - 1, 0] - 0.3))
        s0 = None

rng = np.random.default_rng(0)
data = []
for a, b in segs:
    L = [p for t, p in lidar if a <= t <= b][:20]
    D = [p for t, p in depth if a <= t <= b]
    if len(L) < 5 or len(D) < 5:
        continue
    lid = gf.transform_points(gf.T_TORSO_LIVOX, gf.clean_lidar(np.vstack(L)))
    lid = lid[np.linalg.norm(lid[:, :2], axis=1) < 4.0]
    # floor plane in the torso frame: RANSAC on the lowest points
    i_ = odom[(odom[:, 0] >= a) & (odom[:, 0] <= b)]
    rot = R.from_quat(i_[:, 1:5]).mean()
    yaw = rot.as_euler("zyx")[0]
    R_lv = (R.from_euler("z", -yaw) * rot).as_matrix()          # torso -> levelled
    zl = (lid @ R_lv.T)[:, 2]
    fz = gf.estimate_floor_z(np.c_[zl, zl, zl])
    cand = lid[np.abs(zl - fz) < 0.05]
    best = None
    for _ in range(200):
        p3 = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p3[1] - p3[0], p3[2] - p3[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((cand - p3[0]) @ n) < 0.015
        if best is None or inl.sum() > best[1].sum():
            best = (n, inl)
    fl = cand[best[1]]
    c = fl.mean(0)
    n = np.linalg.svd(fl - c, full_matrices=False)[2][-1]
    if n @ (R_lv.T @ [0, 0, 1]) < 0:
        n = -n
    d_plane = n @ c
    # lidar surface normals for point-to-plane on non-floor structure
    tree = cKDTree(lid)
    _, idx = tree.query(lid, k=12)
    nb = lid[idx] - lid[idx].mean(1, keepdims=True)
    w_, v_ = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
    normals, planar = v_[:, :, 0], w_[:, 0] / (w_.sum(1) + 1e-12) < 0.03
    # depth samples
    Dp = np.vstack([D[j] for j in rng.choice(len(D), min(N_DEPTH_FRAMES, len(D)), replace=False)])
    Dp = Dp[rng.choice(len(Dp), min(N_DEPTH_PTS, len(Dp)), replace=False)]
    u = (FX * Dp[:, 0] / Dp[:, 2] + PPX - W / 2) / (W / 2)
    v = (FY * Dp[:, 1] / Dp[:, 2] + PPY - H / 2) / (H / 2)
    data.append(dict(t=(a - t0, b - t0), p=Dp, u=u, v=v, r=np.linalg.norm(Dp, axis=1), n=n, d=d_plane,
                     R_lv=R_lv, tree=tree, lid=lid, normals=normals, planar=planar,
                     height=-fz, pitch=np.degrees(rot.as_euler("zyx")[1]), yaw=np.degrees(yaw)))
print(f"{len(data)} static segments:")
for i, s in enumerate(data):
    print(f"  [{i}] t={s['t'][0]:6.1f}-{s['t'][1]:6.1f}s  torso {s['height']:.3f} m  pitch {s['pitch']:+5.1f}  yaw {s['yaw']:+7.1f}")

R0, t0_ = gf.T_TORSO_OPTICAL[:3, :3], gf.T_TORSO_OPTICAL[:3, 3]
URDF = gf.T_TORSO_D435_LINK_URDF.copy(); URDF[:3, :3] = URDF[:3, :3] @ gf.R_D435_LINK_OPTICAL
NC = 7
VARIANTS = {  # name: (free extrinsic indices [rx,ry,rz,tx,ty,tz], free range-model indices)
    "rot + z":               ([0, 1, 2, 5], []),
    "rot + xyz":             ([0, 1, 2, 3, 4, 5], []),
    "rot + z + scale":       ([0, 1, 2, 5], [0]),
    "rot + z + scale,r":     ([0, 1, 2, 5], [0, 6]),
    "rot + z + full model":  ([0, 1, 2, 5], list(range(NC))),
}


def feats(s):
    u, v, r = s["u"], s["v"], s["r"]
    return np.stack([np.ones_like(u), u, v, u * u, v * v, u * v, r], 1)


def apply(theta, s):
    Rm = R.from_rotvec(theta[:3]).as_matrix() @ R0
    t = t0_ + theta[3:6]
    kk = 1.0 + feats(s) @ theta[6:6 + NC] if len(theta) > 6 else 1.0
    return (s["p"] * np.atleast_1d(kk)[:, None]) @ Rm.T + t


def residuals(theta, segs_, assoc):
    res = []
    for s, (fl_mask, nf_idx, nf_ok) in zip(segs_, assoc):
        q = apply(theta, s)
        res.append(q[fl_mask] @ s["n"] - s["d"])
        qn = q[~fl_mask][nf_ok]
        j = nf_idx[nf_ok]
        res.append(0.5 * np.einsum("ij,ij->i", qn - s["lid"][j], s["normals"][j]))   # down-weight structure
    return np.concatenate(res)


def associate(theta, segs_):
    assoc = []
    for s in segs_:
        q = apply(theta, s)
        h = q @ s["n"] - s["d"]
        fl = np.abs(h) < 0.10
        dist, j = s["tree"].query(q[~fl])
        ok = (dist < 0.10) & s["planar"][j]
        assoc.append((fl, j, ok))
    return assoc


def fit(segs_, variant):
    free_e, free_m = VARIANTS[variant]
    free = np.array(free_e + [6 + j for j in free_m], dtype=int)
    theta = np.zeros(6 + NC)
    def f(x, assoc):
        th = theta.copy(); th[free] = x
        return residuals(th, segs_, assoc)
    for _ in range(4):
        assoc = associate(theta, segs_)
        theta[free] = least_squares(f, theta[free], args=(assoc,), loss="huber", f_scale=0.02).x
    return theta


def floor_err(T_or_theta, s, bins=((0.2, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 2.5))):
    if isinstance(T_or_theta, np.ndarray) and T_or_theta.shape == (4, 4):
        q = s["p"] @ T_or_theta[:3, :3].T + T_or_theta[:3, 3]
    else:
        q = apply(T_or_theta, s)
    h = q @ s["n"] - s["d"]
    fl = np.abs(h) < 0.15
    hd = np.linalg.norm((q @ s["R_lv"].T)[:, :2], axis=1)
    out = []
    for lo, hi in bins:
        m = fl & (hd >= lo) & (hd < hi)
        out.append(np.median(h[m]) * 1000 if m.sum() > 50 else np.nan)
    return np.array(out)


# ---- leave-one-out ----
print("\nheld-out floor error (median, mm) by horizontal distance 0.2-0.6 | 0.6-1.0 | 1.0-1.5 | 1.5-2.5 m")
names = ["URDF", "current g1_frames"] + list(VARIANTS)
rows = {n: [] for n in names}
for i, s in enumerate(data):
    train = [x for j, x in enumerate(data) if j != i]
    rows["URDF"].append(floor_err(URDF, s)); rows["current g1_frames"].append(floor_err(gf.T_TORSO_OPTICAL, s))
    for v in VARIANTS:
        rows[v].append(floor_err(fit(train, v), s))
    print(f"  [{i}] torso {s['height']:.3f} pitch {s['pitch']:+5.1f} yaw {s['yaw']:+6.1f}: "
          f"URDF {' '.join(f'{x:+4.0f}' for x in rows['URDF'][-1])} | g1_frames {' '.join(f'{x:+4.0f}' for x in rows['current g1_frames'][-1])} | "
          f"rot+z {' '.join(f'{x:+4.0f}' for x in rows['rot + z'][-1])} | +scale,r {' '.join(f'{x:+4.0f}' for x in rows['rot + z + scale,r'][-1])}")
print("\nsummary over held-out poses: mean |error| per bin (mm) | worst")
for n, r in rows.items():
    r = np.abs(np.array(r))
    print(f"  {n:22s} " + " ".join(f"{x:5.1f}" for x in np.nanmean(r, 0)) + f"  | {np.nanmax(r):5.1f}")

fits = {}
for v in VARIANTS:
    th = fit(data, v); fits[v] = th
    TT = np.eye(4); TT[:3, :3] = R.from_rotvec(th[:3]).as_matrix() @ R0; TT[:3, 3] = t0_ + th[3:6]
    Rl = TT[:3, :3] @ gf.R_D435_LINK_OPTICAL.T
    print(f"all-data [{v:20s}] d435_joint xyz={TT[:3,3].round(4)} rpy={R.from_matrix(Rl).as_euler('xyz').round(4)} "
          f"range coef={th[6:][np.abs(th[6:])>0].round(4)}")
np.savez(out, **{v.replace(" ", "_").replace("+", "p").replace(",", "_"): th for v, th in fits.items()},
         R0=R0, t0=t0_, feature_names=np.array(["1", "u", "v", "u2", "v2", "uv", "r"]))
print("saved", out)
