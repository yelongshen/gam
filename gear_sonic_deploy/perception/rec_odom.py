"""Record /dlio/odom to npz: t, xyz, quat(xyzw). usage: rec_odom.py seconds out.npz"""
import sys, time, numpy as np, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
secs, out = float(sys.argv[1]), sys.argv[2]
rclpy.init(); n = Node("rec_odom"); rows = []
def cb(m):
    p, q = m.pose.pose.position, m.pose.pose.orientation
    rows.append([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
n.create_subscription(Odometry, "/dlio/odom", cb, 50)
end = time.time() + secs
while time.time() < end: rclpy.spin_once(n, timeout_sec=0.1)
a = np.array(rows); np.save(out, a); print(f"{len(a)} odom msgs over {a[-1,0]-a[0,0]:.1f}s -> {out}")
