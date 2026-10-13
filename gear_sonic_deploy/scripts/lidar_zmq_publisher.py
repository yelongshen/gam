#!/usr/bin/env python3
"""lidar_zmq_publisher.py
=========================
Bridges the G1's onboard LiDAR (Mid-360) DDS topics to a ZMQ PUB socket so
that any client on the network (teleop PC, workstation, etc.) can monitor
LiDAR data in real time without needing direct DDS network access.

Run this ON the robot (or on a machine with DDS access to the robot's
network interface), then run `view_lidar_client.py` from anywhere that can
reach this machine over TCP.

Subscribes (DDS / ROS2 topics, verified live on this G1 via `ros2 topic
list` with RMW_IMPLEMENTATION=rmw_cyclonedds_cpp):
  - rt/utlidar/imu_livox_mid360   -> sensor_msgs.msg.dds_.Imu_          (IMU)
  - rt/utlidar/cloud_livox_mid360 -> sensor_msgs.msg.dds_.PointCloud2_  (raw points)

NOTE: `ros2 topic list` shows these WITHOUT the `rt/` prefix (e.g.
`utlidar/cloud_livox_mid360`) -- that's just how ROS2's CLI displays ROS
topic names. `rmw_cyclonedds` publishes the *raw DDS* topic name with an
`rt/` prefix, and this script's `ChannelSubscriber` talks directly to
CycloneDDS (bypassing ROS2's rmw layer), so the `rt/` prefix IS required
here. Verified live on this G1 (width=22080 points, frame_id=livox_frame).

Also NOTE: these are NOT the same as the generic `utlidar/cloud` / `rt/
utlidar/lidar_state` names used by some Unitree docs and by earlier
revisions of this script -- always confirm with `ros2 topic list` /
`ros2 topic info -v <topic>` on your own unit if this doesn't work.

Publishes (ZMQ PUB), using the same wire format as the rest of the
gear_sonic teleop stack ([topic][1280B JSON header][binary payload]; the
packer is inlined below rather than importing
gear_sonic.utils.teleop.zmq.zmq_planner_sender.pack_pose_message, since
that module uses `float | None` type-hint syntax requiring Python >= 3.10,
while this script must also run on the robot's Python 3.8):
  - topic "lidar_imu":       orientation/angular_velocity/linear_acceleration
  - topic "lidar_cloud":     Nx3 float32 XYZ points, packed as a pose message
  - topic "lidar_heightmap": grid_size x grid_size float32 height map (see
    height_map.py), built by binning the raw point cloud into a fixed-size
    grid around the sensor and taking the max Z per cell. Republished at
    the same rate as the point cloud (--rate).

Usage:
    # On the robot / DDS-connected machine:
    python3 gear_sonic_deploy/scripts/lidar_zmq_publisher.py eth0 --port 5558

    # If topic names differ on your firmware:
    python3 gear_sonic_deploy/scripts/lidar_zmq_publisher.py eth0 \
        --topic-imu rt/utlidar/imu_livox_mid360 --topic-cloud rt/utlidar/cloud_livox_mid360
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import zmq

# Ensure height_map.py (sibling script) is importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import Imu_, PointCloud2_

from height_map import HeightMapConfig, build_height_map

# Wire format: [topic bytes][1280-byte JSON header][concatenated binary
# field data]. Matches gear_sonic.utils.teleop.zmq.zmq_planner_sender's
# pack_pose_message (inlined here to avoid that module's Python>=3.10-only
# `float | None` type hints, since this script must run on Python 3.8 too).
_HEADER_SIZE = 1280
_DTYPE_TO_STR = {
    np.dtype("float32"): "f32",
    np.dtype("float64"): "f64",
    np.dtype("int32"): "i32",
    np.dtype("int64"): "i64",
    np.dtype("bool"): "bool",
}


def pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 4) -> bytes:
    fields = []
    binary_data = []
    for key, value in pose_data.items():
        value = np.asarray(value)
        dtype_str = _DTYPE_TO_STR.get(value.dtype, "f32")
        if dtype_str == "f32" and value.dtype != np.float32:
            value = value.astype(np.float32)
        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})
        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        binary_data.append(value.tobytes())

    header = {"v": version, "endian": "le", "count": 1, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > _HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {_HEADER_SIZE}")
    header_bytes = header_json.ljust(_HEADER_SIZE, b"\x00")

    return topic.encode("utf-8") + header_bytes + b"".join(binary_data)


def decode_xyz(msg: PointCloud2_) -> np.ndarray:
    """Decode an Nx3 float32 array of (x, y, z) points from a PointCloud2_."""
    field_by_name = {f.name: f for f in msg.fields}
    for name in ("x", "y", "z"):
        if name not in field_by_name:
            raise ValueError(f"PointCloud2_ has no '{name}' field; "
                              f"fields={[f.name for f in msg.fields]}")

    endian = ">" if msg.is_bigendian else "<"
    data = bytes(msg.data)
    n_points = (msg.width * msg.height) if msg.height else (len(data) // msg.point_step)

    xo, yo, zo = field_by_name["x"].offset, field_by_name["y"].offset, field_by_name["z"].offset
    fmt = endian + "f"
    out = np.empty((n_points, 3), dtype=np.float32)
    for i in range(n_points):
        base = i * msg.point_step
        out[i, 0] = struct.unpack_from(fmt, data, base + xo)[0]
        out[i, 1] = struct.unpack_from(fmt, data, base + yo)[0]
        out[i, 2] = struct.unpack_from(fmt, data, base + zo)[0]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("iface", nargs="?", default=None,
                         help="DDS network interface (e.g. eth0). Omit to use default DDS config.")
    parser.add_argument("--port", type=int, default=5558, help="ZMQ port to publish on")
    parser.add_argument("--topic-imu", default="rt/utlidar/imu_livox_mid360")
    parser.add_argument("--topic-cloud", default="rt/utlidar/cloud_livox_mid360")
    parser.add_argument("--zmq-topic-imu", default="lidar_imu")
    parser.add_argument("--zmq-topic-cloud", default="lidar_cloud")
    parser.add_argument("--zmq-topic-heightmap", default="lidar_heightmap")
    parser.add_argument("--rate", type=float, default=10.0, help="Max republish rate (Hz)")
    parser.add_argument("--heightmap-grid-size", type=int, default=11,
                         help="NxN height map grid size (default: 11)")
    parser.add_argument("--heightmap-half-extent", type=float, default=2.0,
                         help="Height map half-extent in meters; grid covers "
                              "[-half_extent, +half_extent] in X and Y (default: 2.0)")
    parser.add_argument("--heightmap-agg", default="max", choices=["max", "min", "mean"],
                         help="Per-cell aggregation of Z values (default: max)")
    parser.add_argument("--no-heightmap", action="store_true",
                         help="Disable height map computation/publishing")
    args = parser.parse_args()

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    imu_sub = ChannelSubscriber(args.topic_imu, Imu_)
    imu_sub.Init()
    cloud_sub = ChannelSubscriber(args.topic_cloud, PointCloud2_)
    cloud_sub.Init()

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.SNDBUF, 4096 * 1024)
    # Bind on all interfaces (not just loopback) so remote clients on the
    # network can connect -- binding to 127.0.0.1 always succeeds locally,
    # so a try/except fallback never actually reaches the "*" bind.
    socket.bind(f"tcp://*:{args.port}")

    print(f"DDS: subscribed to '{args.topic_imu}' + '{args.topic_cloud}'")
    print(f"ZMQ: publishing on port {args.port}, topics "
          f"'{args.zmq_topic_imu}' + '{args.zmq_topic_cloud}'"
          + (f" + '{args.zmq_topic_heightmap}'" if not args.no_heightmap else ""))
    if not args.no_heightmap:
        print(f"  height map: {args.heightmap_grid_size}x{args.heightmap_grid_size} grid, "
              f"half_extent={args.heightmap_half_extent}m, agg='{args.heightmap_agg}'")

    heightmap_config = HeightMapConfig(
        grid_size=args.heightmap_grid_size,
        half_extent=args.heightmap_half_extent,
        agg=args.heightmap_agg,
    )

    period = 1.0 / args.rate
    last_sent = 0.0
    n_sent = 0

    try:
        while True:
            now = time.time()

            imu_msg = imu_sub.Read()
            if imu_msg is not None:
                q = imu_msg.orientation
                av = imu_msg.angular_velocity
                la = imu_msg.linear_acceleration
                data = {
                    "stamp_sec": np.array([imu_msg.header.stamp.sec], dtype=np.int64),
                    "stamp_nanosec": np.array([imu_msg.header.stamp.nanosec], dtype=np.int64),
                    "orientation": np.array([q.x, q.y, q.z, q.w], dtype=np.float32),
                    "angular_velocity": np.array([av.x, av.y, av.z], dtype=np.float32),
                    "linear_acceleration": np.array([la.x, la.y, la.z], dtype=np.float32),
                }
                socket.send(pack_pose_message(data, topic=args.zmq_topic_imu, version=4))

            if now - last_sent >= period:
                cloud_msg = cloud_sub.Read()
                if cloud_msg is not None:
                    points = decode_xyz(cloud_msg)
                    data = {
                        "points": points,
                        "width": np.array([cloud_msg.width], dtype=np.int32),
                        "height": np.array([cloud_msg.height], dtype=np.int32),
                        "stamp": np.array([now], dtype=np.float64),
                    }
                    socket.send(pack_pose_message(data, topic=args.zmq_topic_cloud, version=4))

                    if not args.no_heightmap:
                        grid = build_height_map(points, heightmap_config)
                        hm_data = {
                            "grid": np.nan_to_num(grid, nan=-1e6).astype(np.float32),
                            "grid_size": np.array([heightmap_config.grid_size], dtype=np.int32),
                            "half_extent": np.array([heightmap_config.half_extent], dtype=np.float32),
                            "cell_size": np.array([heightmap_config.cell_size], dtype=np.float32),
                            "empty_sentinel": np.array([-1e6], dtype=np.float32),
                            "stamp": np.array([now], dtype=np.float64),
                        }
                        socket.send(pack_pose_message(hm_data, topic=args.zmq_topic_heightmap, version=4))

                    last_sent = now
                    n_sent += 1
                    if n_sent % 20 == 0:
                        print(f"  republished {n_sent} cloud frames "
                              f"({points.shape[0]} pts last frame)")

            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

