#!/usr/bin/env python3
"""Find structure that moves with the robot (gantry, self): voxels in the levelled torso frame that
stay occupied through a recording in which the robot moved. usage: find_attached.py <bag> (in container)"""
import sys
from pathlib import Path
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as R
from sensor_msgs_py import point_cloud2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import g1_frames as gf

VOX = 0.05
reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
types = {t.name: t.type for t in reader.get_all_topics_and_types()}
reader.set_filter(rosbag2_py.StorageFilter(topics=["/dlio/odom", "/g1/lidar/points"]))
odom_t, odom_q, odom_p, scans = [], [], [], []
while reader.has_next():
    topic, data, _ = reader.read_next()
    m = deserialize_message(data, get_message(types[topic]))
    t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    if topic == "/dlio/odom":
        q, p = m.pose.pose.orientation, m.pose.pose.position
        odom_t.append(t); odom_q.append([q.x, q.y, q.z, q.w]); odom_p.append([p.x, p.y, p.z])
    else:
        scans.append((t, point_cloud2.read_points_numpy(m, field_names=("x", "y", "z")).astype(np.float64)))
odom_t, odom_q, odom_p = np.array(odom_t), np.array(odom_q), np.array(odom_p)

# use only scans taken while the robot was moving (else static world also looks "attached")
counts, n_used = {}, 0
floor_zs = []
for t, pts in scans:
    i = min(np.searchsorted(odom_t, t), len(odom_t) - 1)
    if not (5 < t - odom_t[0] < 94):          # the motion part of motion1 (static after 94 s)
        continue
    rot = R.from_quat(odom_q[i])
    yaw = rot.as_euler("zyx")[0]
    R_level = (R.from_euler("z", -yaw) * rot).as_matrix()     # torso -> levelled torso (roll/pitch removed)
    p = gf.transform_points(gf.T_TORSO_LIVOX, gf.clean_lidar(pts)) @ R_level.T
    floor_zs.append(gf.estimate_floor_z(p))
    near = p[(np.abs(p[:, 0]) < 1.5) & (np.abs(p[:, 1]) < 1.5)]
    for k in set(map(tuple, np.floor(near / VOX).astype(int))):
        counts[k] = counts.get(k, 0) + 1
    n_used += 1
floor = float(np.median(floor_zs))
print(f"{n_used} moving scans; torso {-floor:.3f} m above floor")
keys = np.array([k for k, c in counts.items() if c >= 0.5 * n_used])
if len(keys) == 0:
    sys.exit("no persistent voxels")
cent = (keys + 0.5) * VOX
cent = cent[(cent[:, 2] - floor) > 0.08]            # ignore the floor itself
print(f"persistent voxels (occupied in >=50% of moving scans, above floor): {len(cent)}")
from scipy.cluster.hierarchy import fcluster, linkage
lab = fcluster(linkage(cent, "single"), 1.5 * VOX, "distance")
for k in np.unique(lab):
    c = cent[lab == k]
    z = c[:, 2] - floor
    print(f"  n={len(c):3d}  x[{c[:,0].min()-VOX/2:+.2f},{c[:,0].max()+VOX/2:+.2f}] "
          f"y[{c[:,1].min()-VOX/2:+.2f},{c[:,1].max()+VOX/2:+.2f}] z_above_floor[{z.min()-VOX/2:.2f},{z.max()+VOX/2:.2f}]")
