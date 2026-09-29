import sys, time, numpy as np, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
secs = float(sys.argv[1]); rclpy.init(); n = Node("watch_odom"); last = {}
n.create_subscription(Odometry, "/dlio/odom", lambda m: last.update(m=m), 10)
t0 = time.time(); tp = 0
while time.time() - t0 < secs:
    rclpy.spin_once(n, timeout_sec=0.1)
    if "m" in last and time.time() - tp >= 5:
        tp = time.time(); m = last["m"]; p = m.pose.pose.position; q = m.pose.pose.orientation
        yaw = np.degrees(R.from_quat([q.x, q.y, q.z, q.w]).as_euler("zyx")[0]); v = m.twist.twist.linear
        print(f"t={time.time()-t0:5.0f}s  x={p.x:+.3f} y={p.y:+.3f} z={p.z:+.3f} m  yaw={yaw:+7.2f} deg  |v|={np.hypot(v.x,v.y):.2f} m/s", flush=True)
