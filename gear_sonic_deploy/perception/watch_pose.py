"""Print torso height above floor (from the lidar) and roll/pitch/yaw (DLIO) every few seconds.
usage: watch_pose.py <secs>   (inside the container, workspace sourced)"""
import sys, time
from pathlib import Path
import numpy as np, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import g1_frames as gf
secs = float(sys.argv[1]); rclpy.init(); n = Node("watch_pose"); last = {}
n.create_subscription(Odometry, "/dlio/odom", lambda m: last.update(o=m), 10)
n.create_subscription(PointCloud2, "/g1/lidar/points", lambda m: last.update(c=m), 5)
t0 = time.time(); tp = 0
while time.time() - t0 < secs:
    rclpy.spin_once(n, timeout_sec=0.1)
    if "o" in last and "c" in last and time.time() - tp >= 3:
        tp = time.time(); q = last["o"].pose.pose.orientation; rot = R.from_quat([q.x, q.y, q.z, q.w])
        yaw, pitch, roll = np.degrees(rot.as_euler("zyx"))
        R_lv = (R.from_euler("z", -np.radians(yaw)) * rot).as_matrix()
        p = gf.transform_points(gf.T_TORSO_LIVOX, point_cloud2.read_points_numpy(last["c"], field_names=("x", "y", "z")).astype(float)) @ R_lv.T
        h = -gf.estimate_floor_z(p[np.hypot(p[:, 0], p[:, 1]) > 1.0])
        print(f"t={time.time()-t0:5.0f}s  torso {h:.3f} m above floor  roll {roll:+5.1f}  pitch {pitch:+5.1f}  yaw {yaw:+7.1f} deg", flush=True)
