#!/usr/bin/env python3
"""ZMQ -> ROS 2 bridge for the G1 head sensors (desktop side).

Subscribes to the robot's g1_depth_publisher.py (5557, `depth` image topic only;
run it with --no-points to save ~7 MB/s of Wi-Fi) and g1_lidar_publisher.py (5558)
and republishes:
  /g1/lidar/points  PointCloud2, frame mid360_link (raw livox_frame points; with a
                    per-point `time` field in seconds when the publisher runs --fastlio)
  /g1/imu           sensor_msgs/Imu, frame mid360_imu, linear_acceleration in m/s^2
                    (the MID-360 reports g)
  /g1/depth/points  PointCloud2, frame d435_optical (deprojected here, <= depth_max_range)
  TF  torso_link -> d435_optical  g1_frames.T_TORSO_OPTICAL (static)

gantry_filter:=true drops points within gantry_radius (0.8 m) of the torso that are more than
gantry_min_height (0.45 m) above the floor, from both clouds, before anything consumes them.
Use it for tests on the gantry only (see g1_frames.near_tall_mask).

odom_source:=static (default; robot on the gantry)
  TF  robot/odom -> torso_link   levelled from the MID-360 IMU, floor at z=0, torso height
                                 re-estimated from every lidar frame (the gantry can move)
  TF  torso_link -> mid360_link  g1_frames.T_TORSO_LIVOX (static)
odom_source:=dlio
  DLIO publishes robot/odom -> torso_link -> {mid360_link, mid360_imu} itself (see
  dlio_g1.yaml); run the lidar publisher with --fastlio so DLIO can de-skew.

Timestamps: lidar/IMU stamps come from the sensor clock (~18 min off the Jetson) and
depth stamps from the Jetson clock. Each is mapped onto this machine's clock with an
offset = min(receive time - sensor time) over a sliding window, i.e. the least-delayed
message defines the offset, so relative timing between lidar and IMU is preserved.
"""
import array
import collections
import json
import struct
import sys
import threading
import time

import lz4.frame as lz4frame
import numpy as np
import rclpy
import zmq
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

import g1_frames as gf

HEADER_SIZE = 1280
DTYPES = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64, "bool": np.bool_}
GRAVITY = 9.80665
XYZT_FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
               for i, n in enumerate(["x", "y", "z", "time"])]
XYZ_FIELDS = XYZT_FIELDS[:3]


def make_cloud(header, fields, pts):
    """PointCloud2 straight from an (N, len(fields)) float32 array. sensor_msgs_py's create_cloud
    packs point by point in Python on Humble (~3 cores at our rates on the Jetson; vectorised on
    Jazzy), which backed messages up and made DLIO diverge."""
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, pts.shape[0]
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 4 * len(fields)
    msg.row_step = msg.point_step * pts.shape[0]
    msg.is_dense = True
    msg.data = array.array("B", pts.tobytes())
    return msg


def parse_lidar(msg, topic):
    hdr = json.loads(msg[len(topic):len(topic) + HEADER_SIZE].rstrip(b"\x00"))
    off = len(topic) + HEADER_SIZE
    out = {}
    for f in hdr["fields"]:
        dt = np.dtype(DTYPES[f["dtype"]])
        n = int(np.prod(f["shape"])) * dt.itemsize
        out[f["name"]] = np.frombuffer(msg[off:off + n], dtype=dt).reshape(f["shape"])
        off += n
    return out


def parse_depth(part):
    n = struct.unpack("<I", part[:4])[0]
    hdr = json.loads(part[4:4 + n])
    body = part[4 + n:]
    if hdr["compress"] == "lz4":
        body = lz4frame.decompress(body)
    return hdr, np.frombuffer(body, dtype=hdr["dtype"]).reshape(hdr["shape"])


def to_tf(T, parent, child, stamp):
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, T[:3, 3])
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = map(float, q)
    return msg


def to_time_msg(t):
    sec = int(np.floor(t))
    return TimeMsg(sec=sec, nanosec=int((t - sec) * 1e9))


class ClockMap:
    """Map a remote clock onto the local one: local = remote + min(recv - remote) over a window."""

    def __init__(self, window=400):
        self.samples = collections.deque(maxlen=window)
        self.lock = threading.Lock()

    def __call__(self, remote_t):
        with self.lock:
            self.samples.append(time.time() - remote_t)
            return remote_t + min(self.samples)


class G1ZmqBridge(Node):
    def __init__(self):
        super().__init__("g1_zmq_bridge")
        self.declare_parameter("host", "192.168.8.227")
        self.declare_parameter("map_frame", "robot/odom")
        self.declare_parameter("depth_max_range", gf.DEPTH_MAX_RANGE)
        self.declare_parameter("odom_source", "static")
        self.declare_parameter("gantry_filter", False)
        self.declare_parameter("gantry_radius", 0.8)
        self.declare_parameter("gantry_min_height", 0.45)
        host = self.get_parameter("host").value
        self.map_frame = self.get_parameter("map_frame").value
        self.depth_max_range = float(self.get_parameter("depth_max_range").value)
        self.odom_source = self.get_parameter("odom_source").value
        assert self.odom_source in ("static", "dlio"), self.odom_source
        self.gantry = bool(self.get_parameter("gantry_filter").value)
        self.gantry_radius = float(self.get_parameter("gantry_radius").value)
        self.gantry_min_height = float(self.get_parameter("gantry_min_height").value)

        self.pub_lidar = self.create_publisher(PointCloud2, "/g1/lidar/points", 5)
        self.pub_depth = self.create_publisher(PointCloud2, "/g1/depth/points", 5)
        self.pub_imu = self.create_publisher(Imu, "/g1/imu", rclpy.qos.qos_profile_sensor_data)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        now = self.get_clock().now().to_msg()
        statics = [to_tf(gf.T_TORSO_OPTICAL, "torso_link", "d435_optical", now)]
        if self.odom_source == "static":
            statics.append(to_tf(gf.T_TORSO_LIVOX, "torso_link", "mid360_link", now))
        self.static_tf.sendTransform(statics)

        self.sensor_clock = ClockMap()   # MID-360 cloud + IMU stamps
        self.jetson_clock = ClockMap()   # g1_depth_publisher stamp_host
        self.accel = None       # EMA of MID-360 IMU specific force (livox frame, g)
        self.torso_height = None
        self.rays = None        # cached depth deprojection rays
        self.lock = threading.Lock()
        self.counts = {"lidar": 0, "depth": 0, "imu": 0, "timed": 0, "gantry_dropped": 0}

        ctx = zmq.Context.instance()
        self.sock_lidar = ctx.socket(zmq.SUB)
        self.sock_lidar.connect(f"tcp://{host}:5558")
        self.sock_lidar.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock_depth = ctx.socket(zmq.SUB)
        self.sock_depth.setsockopt(zmq.RCVHWM, 2)
        self.sock_depth.connect(f"tcp://{host}:5557")
        self.sock_depth.setsockopt(zmq.SUBSCRIBE, b"depth")
        threading.Thread(target=self.rx_loop, daemon=True).start()

        if self.odom_source == "static":
            self.create_timer(0.02, self.publish_static_odom)
        self.create_timer(5.0, self.log_stats)
        self.get_logger().info(f"bridging tcp://{host}:5557,5558 -> ROS 2 (odom_source={self.odom_source}, "
                               f"map_frame={self.map_frame})")

    def rx_loop(self):
        poller = zmq.Poller()
        poller.register(self.sock_lidar, zmq.POLLIN)
        poller.register(self.sock_depth, zmq.POLLIN)
        while rclpy.ok():
            for sock, _ in poller.poll(200):
                if sock is self.sock_lidar:
                    self.on_lidar(sock.recv())
                else:
                    self.on_depth(*sock.recv_multipart())

    def on_lidar(self, msg):
        if msg.startswith(b"lidar_imu"):
            d = parse_lidar(msg, b"lidar_imu")
            a = d["linear_acceleration"].astype(np.float64)
            with self.lock:
                self.accel = a if self.accel is None else 0.98 * self.accel + 0.02 * a
            imu = Imu()
            t = float(d["stamp_sec"][0]) + float(d["stamp_nanosec"][0]) * 1e-9
            imu.header = Header(stamp=to_time_msg(self.sensor_clock(t)), frame_id="mid360_imu")
            imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = map(float, a * GRAVITY)
            w = d["angular_velocity"].astype(np.float64)
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(float, w)
            imu.orientation_covariance[0] = -1.0  # no orientation estimate
            self.pub_imu.publish(imu)
            self.counts["imu"] += 1
            return
        if not msg.startswith(b"lidar_cloud"):
            return
        d = parse_lidar(msg, b"lidar_cloud")
        pts = d["points"]
        if "stamp_sec" in d:
            stamp = to_time_msg(self.sensor_clock(float(d["stamp_sec"][0]) + float(d["stamp_nanosec"][0]) * 1e-9))
        else:
            stamp = self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id="mid360_link")
        keep = np.linalg.norm(pts, axis=1) > gf.LIDAR_MIN_RANGE
        level = None
        if self.accel is not None:
            level = gf.transform_points(gf.T_TORSO_LIVOX, pts) @ gf.level_rotation(gf.up_from_livox_accel(self.accel)).T
        if self.gantry and level is not None and self.torso_height is not None:
            near = gf.near_tall_mask(level, -self.torso_height, self.gantry_radius, self.gantry_min_height)
            self.counts["gantry_dropped"] += int((keep & near).sum())
            keep &= ~near
        if "time" in d:
            # --fastlio: per-point offset from scan start in ns -> s (DLIO "Velodyne" convention)
            cloud = np.column_stack([pts[keep], d["time"][keep].astype(np.float64) * 1e-9]).astype(np.float32)
            self.pub_lidar.publish(make_cloud(header, XYZT_FIELDS, cloud))
            self.counts["timed"] += 1
        else:
            self.pub_lidar.publish(make_cloud(header, XYZ_FIELDS, pts[keep]))
        self.counts["lidar"] += 1
        if level is not None:
            # Track the torso height every frame (EMA, ~1 s at 6 Hz): the robot can be raised or
            # lowered on the gantry, and a stale value shifts the whole map up/down.
            h = -gf.estimate_floor_z(level[np.linalg.norm(pts, axis=1) > gf.LIDAR_MIN_RANGE])
            if np.isfinite(h):
                if self.torso_height is None:
                    self.get_logger().info(f"torso origin {h:.3f} m above floor (from lidar)")
                    self.torso_height = h
                else:
                    if abs(h - self.torso_height) > 0.03:
                        self.get_logger().warn(f"torso height changed: {self.torso_height:.3f} -> {h:.3f} m",
                                               throttle_duration_sec=2.0)
                    self.torso_height = 0.8 * self.torso_height + 0.2 * h

    def on_depth(self, topic, part):
        hdr, img = parse_depth(part)
        k = hdr["intrinsics"]
        key = (img.shape, k["fx"], k["fy"], k["ppx"], k["ppy"])
        if self.rays is None or self.rays[0] != key:
            v, u = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
            self.rays = (key, np.stack([(u - k["ppx"]) / k["fx"], (v - k["ppy"]) / k["fy"], np.ones_like(u)], -1))
        z = img.astype(np.float32) * float(hdr["depth_scale"])
        pts = gf.clean_depth(self.rays[1] * z[..., None], self.depth_max_range)
        if self.gantry and self.accel is not None and self.torso_height is not None:
            level = gf.transform_points(gf.T_TORSO_OPTICAL, pts) @ gf.level_rotation(gf.up_from_livox_accel(self.accel)).T
            near = gf.near_tall_mask(level, -self.torso_height, self.gantry_radius, self.gantry_min_height)
            self.counts["gantry_dropped"] += int(near.sum())
            pts = pts[~near]
        header = Header(stamp=to_time_msg(self.jetson_clock(float(hdr["stamp_host"]))), frame_id="d435_optical")
        self.pub_depth.publish(make_cloud(header, XYZ_FIELDS, pts))
        self.counts["depth"] += 1

    def publish_static_odom(self):
        with self.lock:
            accel = None if self.accel is None else self.accel.copy()
        if accel is None or self.torso_height is None:
            return
        T = np.eye(4)
        T[:3, :3] = gf.level_rotation(gf.up_from_livox_accel(accel))
        T[2, 3] = self.torso_height
        self.tf.sendTransform(to_tf(T, self.map_frame, "torso_link", self.get_clock().now().to_msg()))

    def log_stats(self):
        c = self.counts
        self.get_logger().info(f"last 5 s: lidar {c['lidar']/5:.1f} Hz (with per-point time: {c['timed']/5:.1f} Hz), "
                               f"depth {c['depth']/5:.1f} Hz, imu {c['imu']/5:.0f} Hz"
                               + (f", gantry filter dropped {c['gantry_dropped']/5:.0f} pts/s" if self.gantry else ""))
        self.counts = {k: 0 for k in c}


def main():
    rclpy.init(args=sys.argv)
    node = G1ZmqBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
