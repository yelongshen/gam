#!/usr/bin/env python3
"""zmq_to_ros_livox_bridge.py
=============================
Bridge the G1's ZMQ LiDAR+IMU streams into the ROS topics FAST-LIO expects,
so the full `elevation_mapping_humanoid` stack can run on the WORKSTATION
without ROS ever being installed on the robot ("Option 3").

RUNS ON THE WORKSTATION, inside the RoboStack ROS Noetic conda env.
Pairs with `sim2real/g1_lidar_publisher.py --fastlio` on the robot.

    [ROBOT]  g1_lidar_publisher.py --fastlio
                 |  ZMQ  "lidar_cloud" (xyz + per-point time/intensity/ring)
                 |       "lidar_imu"   (orientation / gyro / accel)
                 v
    [DESKTOP] this script
                 |  ROS   /livox/lidar  (livox_ros_driver2/CustomMsg)
                 |        /livox/imu    (sensor_msgs/Imu)
                 v
              FAST-LIO  ->  elevation_mapping  ->  rviz (grid_map_rviz_plugin)

Why CustomMsg and not PointCloud2: `fast_lio_mid360/config/mid360.yaml` sets
`lidar_type: 1` (Livox), whose `avia_handler` consumes
`livox_ros_driver2::CustomMsg`. Its `CustomPoint.offset_time` (ns relative
to `CustomMsg.timebase`) is what FAST-LIO uses to de-skew the Mid-360's
non-repetitive scan against the IMU. This G1's PointCloud2 exposes that as a
per-point `time` field (verified live: point_step=22,
fields = x/y/z/intensity/ring/time), which `--fastlio` forwards.

Field mapping (PointCloud2 -> CustomPoint):
    time      -> offset_time   (see --time-scale for units)
    intensity -> reflectivity
    ring      -> line

TIME UNITS CAVEAT: Livox/ROS drivers are inconsistent about whether the
per-point `time` field is SECONDS (float, relative to frame start) or
NANOSECONDS. `--time-scale auto` (default) inspects the first frame's max
value and guesses: a Mid-360 frame spans ~0.1 s, so a max around 0.1 means
seconds (scale 1e9), around 1e8 means nanoseconds (scale 1). Override with
`--time-scale 1e9` / `--time-scale 1` if the guess is wrong -- a wrong guess
silently wrecks de-skewing, so check the startup log line.

Usage:
    conda activate ros_noetic && source ~/ros_ws/devel/setup.bash
    roscore &
    python perception_heightmap/zmq_to_ros_livox_bridge.py --host 192.168.8.227

Then, in other terminals (same env + workspace sourced):
    roslaunch fast_lio mapping_mid360.launch
    roslaunch elevation_mapping_demos realsense_demo.launch
"""
import argparse
import os
import sys
import time

import numpy as np
import zmq

import rospy
from sensor_msgs.msg import Imu
from livox_ros_driver2.msg import CustomMsg, CustomPoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from view_lidar_client import parse_message  # noqa: E402


def guess_time_scale(t: np.ndarray) -> float:
    """Return the multiplier converting the per-point `time` field to
    NANOSECONDS (what CustomPoint.offset_time wants).

    A Mid-360 frame spans ~0.1 s, so:
        max(t) ~ 0.1      -> seconds      -> scale 1e9
        max(t) ~ 1e8      -> nanoseconds  -> scale 1
        max(t) ~ 1e5      -> microseconds -> scale 1e3
    """
    if t.size == 0:
        return 1e9
    tmax = float(np.nanmax(t))
    if tmax <= 0:
        return 1e9
    if tmax < 10.0:          # ~0.1 -> seconds
        return 1e9
    if tmax < 1e6:           # ~1e5 -> microseconds
        return 1e3
    return 1.0               # already nanoseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="robot IP running g1_lidar_publisher.py --fastlio")
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--cloud-topic-in", default="lidar_cloud")
    ap.add_argument("--imu-topic-in", default="lidar_imu")
    ap.add_argument("--cloud-topic-out", default="/livox/lidar",
                     help="must match fast_lio's mid360.yaml `lid_topic`")
    ap.add_argument("--imu-topic-out", default="/livox/imu",
                     help="must match fast_lio's mid360.yaml `imu_topic`")
    ap.add_argument("--frame-id", default="livox_frame",
                     help="frame_id stamped on both messages (matches the robot's own "
                          "PointCloud2 frame_id)")
    ap.add_argument("--time-scale", default="auto",
                     help="multiplier converting the per-point `time` field to nanoseconds; "
                          "'auto' guesses from the first frame (see module docstring)")
    ap.add_argument("--use-host-stamp", action="store_true",
                     help="stamp ROS messages with the publisher's host (Jetson) wall clock "
                          "instead of the sensor's own header stamp. Use this if the robot's "
                          "controller clock is offset from NTP (the publisher notes a ~17min "
                          "skew), since FAST-LIO needs cloud+IMU on a CONSISTENT timebase.")
    ap.add_argument("--stats-every", type=float, default=5.0)
    args = ap.parse_args()

    rospy.init_node("zmq_to_ros_livox_bridge", anonymous=True, disable_signals=True)
    pub_cloud = rospy.Publisher(args.cloud_topic_out, CustomMsg, queue_size=10)
    pub_imu = rospy.Publisher(args.imu_topic_out, Imu, queue_size=200)

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{args.port}")
    # Subscribe to BOTH topics off the robot's single PUB socket.
    sub.setsockopt(zmq.SUBSCRIBE, args.cloud_topic_in.encode())
    sub.setsockopt(zmq.SUBSCRIBE, args.imu_topic_in.encode())
    sub.setsockopt(zmq.RCVHWM, 20)
    rospy.loginfo(f"[bridge] ZMQ tcp://{args.host}:{args.port} "
                  f"({args.cloud_topic_in!r}, {args.imu_topic_in!r}) -> ROS "
                  f"{args.cloud_topic_out} + {args.imu_topic_out}")

    time_scale = None if args.time_scale == "auto" else float(args.time_scale)
    n_cloud = n_imu = 0
    n_pts_last = 0
    t_stat = time.time()

    cloud_prefix = args.cloud_topic_in.encode()
    imu_prefix = args.imu_topic_in.encode()

    try:
        while not rospy.is_shutdown():
            raw = sub.recv()

            # ---------------- IMU ----------------
            if raw.startswith(imu_prefix):
                data, err = parse_message(raw, args.imu_topic_in)
                if err is not None or data is None:
                    continue
                m = Imu()
                if args.use_host_stamp and "stamp_host" in data:
                    m.header.stamp = rospy.Time.from_sec(float(data["stamp_host"][0]))
                elif "stamp_sec" in data and "stamp_nanosec" in data:
                    m.header.stamp = rospy.Time(int(data["stamp_sec"][0]),
                                                int(data["stamp_nanosec"][0]))
                else:
                    m.header.stamp = rospy.Time.now()
                m.header.frame_id = args.frame_id

                q = data.get("orientation")
                if q is not None and len(q) == 4:
                    m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = \
                        (float(v) for v in q)
                av = data.get("angular_velocity")
                if av is not None:
                    m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = \
                        (float(v) for v in av)
                la = data.get("linear_acceleration")
                if la is not None:
                    m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = \
                        (float(v) for v in la)
                pub_imu.publish(m)
                n_imu += 1
                continue

            # ---------------- CLOUD ----------------
            if not raw.startswith(cloud_prefix):
                continue
            data, err = parse_message(raw, args.cloud_topic_in)
            if err is not None or data is None:
                continue

            pts = data.get("points")
            if pts is None or pts.shape[0] == 0:
                continue

            t_off = data.get("time")
            if t_off is None:
                rospy.logwarn_throttle(
                    10.0,
                    "[bridge] cloud has no per-point `time` field -- is the robot publisher "
                    "running with --fastlio? offset_time will be 0, which DISABLES de-skewing.")
                t_off = np.zeros(pts.shape[0], dtype=np.float32)

            if time_scale is None:
                time_scale = guess_time_scale(t_off)
                rospy.loginfo(f"[bridge] per-point time: max={float(np.nanmax(t_off)):.6g} "
                              f"-> using --time-scale {time_scale:g} (to nanoseconds)")

            offs = np.nan_to_num(t_off.astype(np.float64) * time_scale, nan=0.0)
            offs = np.clip(offs, 0, np.iinfo(np.uint32).max).astype(np.uint32)

            refl = data.get("intensity")
            line = data.get("ring")

            msg = CustomMsg()
            if args.use_host_stamp and "stamp_host" in data:
                stamp = rospy.Time.from_sec(float(data["stamp_host"][0]))
            elif "stamp_sec" in data and "stamp_nanosec" in data:
                stamp = rospy.Time(int(data["stamp_sec"][0]), int(data["stamp_nanosec"][0]))
            else:
                stamp = rospy.Time.now()
            msg.header.stamp = stamp
            msg.header.frame_id = args.frame_id
            msg.timebase = stamp.to_nsec()
            msg.lidar_id = 0
            msg.point_num = int(pts.shape[0])

            xs = pts[:, 0].astype(np.float32)
            ys = pts[:, 1].astype(np.float32)
            zs = pts[:, 2].astype(np.float32)
            refl_u8 = (np.clip(refl, 0, 255).astype(np.uint8) if refl is not None
                       else np.zeros(pts.shape[0], dtype=np.uint8))
            line_u8 = (np.clip(line, 0, 255).astype(np.uint8) if line is not None
                       else np.zeros(pts.shape[0], dtype=np.uint8))

            msg.points = [
                CustomPoint(offset_time=int(offs[i]), x=float(xs[i]), y=float(ys[i]),
                            z=float(zs[i]), reflectivity=int(refl_u8[i]), tag=0,
                            line=int(line_u8[i]))
                for i in range(pts.shape[0])
            ]
            pub_cloud.publish(msg)
            n_cloud += 1
            n_pts_last = pts.shape[0]

            now = time.time()
            if now - t_stat > args.stats_every:
                dt = now - t_stat
                rospy.loginfo(f"[bridge] cloud={n_cloud/dt:.1f} Hz ({n_pts_last} pts/frame), "
                              f"imu={n_imu/dt:.1f} Hz")
                n_cloud = n_imu = 0
                t_stat = now

    except KeyboardInterrupt:
        pass
    finally:
        rospy.loginfo("[bridge] stopping.")


if __name__ == "__main__":
    main()
