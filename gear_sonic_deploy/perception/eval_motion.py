#!/usr/bin/env python3
"""Evaluate DLIO on a recorded bag: find static segments, then compare DLIO's pose change between
the first and last static segment against an independent scan-to-scan ICP of the raw lidar.
usage: eval_motion.py <bag_dir> <out_prefix>   (run inside the emc container)"""
import sys
from pathlib import Path
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from sensor_msgs_py import point_cloud2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import g1_frames as gf

bag, out = sys.argv[1], sys.argv[2]
reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
types = {t.name: t.type for t in reader.get_all_topics_and_types()}
reader.set_filter(rosbag2_py.StorageFilter(topics=["/dlio/odom", "/g1/lidar/points", "/g1/imu"]))
odom, scans, imu = [], [], []
while reader.has_next():
    topic, data, _ = reader.read_next()
    m = deserialize_message(data, get_message(types[topic]))
    t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    if topic == "/dlio/odom":
        p, q, v, w = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear, m.twist.twist.angular
        odom.append([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w, np.hypot(v.x, v.y), abs(w.z)])
    elif topic == "/g1/lidar/points":
        scans.append((t, point_cloud2.read_points_numpy(m, field_names=("x", "y", "z")).astype(np.float64)))
    else:
        a = m.linear_acceleration
        imu.append([t, a.x, a.y, a.z])
odom, imu = np.array(odom), np.array(imu)
t0 = odom[0, 0]
np.save(out + "_odom.npy", odom)

# static = low speed and yaw rate for >= 3 s (smoothed)
k = 50
spd = np.convolve(odom[:, 8], np.ones(k) / k, "same")
yr = np.convolve(odom[:, 9], np.ones(k) / k, "same")
static = (spd < 0.015) & (yr < 0.02)
segs, start = [], None
for i, s in enumerate(static):
    if s and start is None:
        start = i
    if (not s or i == len(static) - 1) and start is not None:
        if odom[i, 0] - odom[start, 0] >= 3.0:
            segs.append((odom[start, 0], odom[i - 1, 0]))
        start = None
print("static segments (s from bag start):", [(round(a - t0, 1), round(b - t0, 1)) for a, b in segs])
yaw = np.degrees(R.from_quat(odom[:, 4:8]).as_euler("zyx")[:, 0])
for a, b in segs:
    i = (odom[:, 0] >= a) & (odom[:, 0] <= b)
    print(f"   {a-t0:6.1f}-{b-t0:6.1f}s: xyz {odom[i,1:4].mean(0).round(3)}  yaw {yaw[i].mean():+.2f} deg")


def T_of(row):
    T = np.eye(4); T[:3, :3] = R.from_quat(row[4:8]).as_matrix(); T[:3, 3] = row[1:4]; return T


def seg_cloud(a, b, n_max=15):
    sel = [p for t, p in scans if a + 0.5 <= t <= b - 0.5][:n_max]
    return np.vstack(sel)


def mean_pose(a, b):
    i = (odom[:, 0] >= a) & (odom[:, 0] <= b)
    q = R.from_quat(odom[i, 4:8]).mean()
    T = np.eye(4); T[:3, :3] = q.as_matrix(); T[:3, 3] = odom[i, 1:4].mean(0); return T


def icp(src, dst, T0, iters=40):
    """point-to-plane ICP, src -> dst (both in the lidar frame of their segment)."""
    tree = cKDTree(dst)
    _, idx = tree.query(dst, k=10)
    nb = dst[idx] - dst[idx].mean(1, keepdims=True)
    n = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))[1][:, :, 0]
    T = T0.copy()
    for it, dmax in enumerate(np.r_[np.linspace(1.0, 0.1, 20), np.full(iters - 20, 0.1)]):
        p = src @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(p)
        ok = d < dmax
        pp, qq, nn = p[ok], dst[j[ok]], n[j[ok]]
        A = np.hstack([np.cross(pp, nn), nn]); b = -np.einsum("ij,ij->i", pp - qq, nn)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        dT = np.eye(4); dT[:3, :3] = R.from_rotvec(x[:3]).as_matrix(); dT[:3, 3] = x[3:]
        T = dT @ T
    res = np.abs(np.einsum("ij,ij->i", pp - qq, nn))
    return T, ok.mean(), np.median(res)


baseline = sys.argv[3] if len(sys.argv) > 3 else None   # optional ZMQ capture npz taken at DLIO's origin
if len(segs) >= 2 or (baseline and segs):
    (a1, b1) = segs[-1]
    if baseline:
        C0 = gf.clean_lidar(np.vstack(list(np.load(baseline, allow_pickle=True)["clouds"])[:15]))
        T_start = np.eye(4)   # DLIO initialised at identity with the robot in the baseline pose
    else:
        (a0, b0) = segs[0]
        C0 = gf.clean_lidar(seg_cloud(a0, b0)); T_start = mean_pose(a0, b0)
    C1 = gf.clean_lidar(seg_cloud(a1, b1))
    C0 = C0[np.random.default_rng(0).choice(len(C0), min(60000, len(C0)), replace=False)]
    C1 = C1[np.random.default_rng(1).choice(len(C1), min(60000, len(C1)), replace=False)]
    # DLIO's relative lidar pose: T_l0_l1 = inv(T_o_t0 T_t_l) (T_o_t1 T_t_l)
    Tl = gf.T_TORSO_LIVOX
    T_dlio = np.linalg.inv(T_start @ Tl) @ (mean_pose(a1, b1) @ Tl)
    for name, init in [("init=DLIO", T_dlio), ("init=identity", np.eye(4))]:
        T_icp, frac, med = icp(C1, C0, init)
        dT = np.linalg.inv(T_icp) @ T_dlio
        print(f"\nICP ({name}): inlier frac {frac:.2f}, median residual {med*1000:.1f} mm")
        for nm, T in [("ICP ", T_icp), ("DLIO", T_dlio)]:
            print(f"   {nm} lidar motion first->last static: t={T[:3,3].round(3)} m ({np.linalg.norm(T[:3,3])*1000:.0f} mm), "
                  f"rot={np.degrees(R.from_matrix(T[:3,:3]).magnitude()):.2f} deg")
        print(f"   DLIO error vs ICP: {np.linalg.norm(dT[:3,3])*1000:.1f} mm, {np.degrees(R.from_matrix(dT[:3,:3]).magnitude()):.2f} deg")
