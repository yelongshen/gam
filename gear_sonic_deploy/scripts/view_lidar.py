#!/usr/bin/env python3
"""View LiDAR (Mid-360) data streamed from the G1 robot over DDS.

This subscribes to two DDS/ROS2 topics via the vendored `unitree_sdk2py`,
using topic names verified live on this G1 (via `ros2 topic list` /
`ros2 topic info -v` with RMW_IMPLEMENTATION=rmw_cyclonedds_cpp):

  1. `utlidar/imu_livox_mid360`   -> `sensor_msgs.msg.dds_.Imu_`
     LiDAR-mounted IMU: orientation, angular velocity, linear acceleration.
     No raw points.

  2. `utlidar/cloud_livox_mid360` -> `sensor_msgs.msg.dds_.PointCloud2_`
     Raw point cloud (packed binary `data` blob per the ROS2 PointCloud2
     wire format: fields describe the per-point layout, `point_step` is the
     stride in bytes, `width * height` is the point count for an unordered
     cloud). On this robot: ~20,000 points/frame @ ~10 Hz, point_step=22,
     frame_id="livox_frame".

NOTE: `ros2 topic list` shows these WITHOUT the `rt/` prefix (e.g.
`utlidar/cloud_livox_mid360`), because that's how ROS2's CLI displays ROS
topic names. But `rmw_cyclonedds` publishes the *raw DDS* topic name with
an `rt/` prefix, and this script's `ChannelSubscriber` talks directly to
CycloneDDS (bypassing ROS2's rmw layer) -- so the `rt/` prefix IS required
here. Verified live on this G1:
    rt/utlidar/cloud_livox_mid360 -> width=22080 points, frame_id=livox_frame

Also NOTE: these are NOT the same as the generic `utlidar/cloud` / `rt/
utlidar/lidar_state` names referenced in some Unitree docs/SDK examples
(those match a different LidarState_ diagnostics message that this robot
does not appear to publish) -- always double check with `ros2 topic list`
on your own unit if this doesn't work out of the box.

Usage (run on / networked with the robot, on the robot's DDS network
interface, e.g. `eth0`):

    python3 gear_sonic_deploy/scripts/view_lidar.py eth0

Options:

    --topic-imu   <name>   override the Imu_ topic (default utlidar/imu_livox_mid360)
    --topic-cloud <name>   override the PointCloud2_ topic (default utlidar/cloud_livox_mid360)
    --once                  print a single point-cloud snapshot then exit
    --save <path.npy>       save the decoded XYZ points from one snapshot to a .npy file
    --view                  open an Open3D window to visualize the snapshot (requires open3d)

If these topic names don't match what your G1 firmware actually publishes,
discover the live topic names with (requires ROS2 foxy + the robot's
cyclonedds workspace sourced -- see this repo's onboard
~/cyclonedds_ws/cyclonedds.xml for the exact NetworkInterface to use):

    source /opt/ros/foxy/setup.bash
    source ~/cyclonedds_ws/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=~/cyclonedds_ws/cyclonedds.xml
    ros2 topic list | grep -i lidar
    ros2 topic info -v <topic>   # confirms message type + publisher count
"""

import argparse
import struct
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import Imu_, PointCloud2_


def decode_xyz(msg: PointCloud2_):
    """Decode (x, y, z) float32 points out of a PointCloud2_ message.

    Assumes the common convention where 'x', 'y', 'z' fields are float32
    (datatype 7) at some offset within each point_step-sized record.
    """
    field_by_name = {f.name: f for f in msg.fields}
    for name in ("x", "y", "z"):
        if name not in field_by_name:
            raise ValueError(f"PointCloud2_ message has no '{name}' field; "
                              f"fields={[f.name for f in msg.fields]}")

    endian = ">" if msg.is_bigendian else "<"
    data = bytes(msg.data)
    n_points = (msg.width * msg.height) if msg.height else (len(data) // msg.point_step)

    xs, ys, zs = field_by_name["x"], field_by_name["y"], field_by_name["z"]
    points = []
    for i in range(n_points):
        base = i * msg.point_step
        x = struct.unpack_from(endian + "f", data, base + xs.offset)[0]
        y = struct.unpack_from(endian + "f", data, base + ys.offset)[0]
        z = struct.unpack_from(endian + "f", data, base + zs.offset)[0]
        points.append((x, y, z))
    return points


def print_imu(msg: Imu_):
    q = msg.orientation
    av = msg.angular_velocity
    la = msg.linear_acceleration
    print(
        f"[Imu] stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} "
        f"frame_id={msg.header.frame_id} "
        f"orientation(xyzw)=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f}) "
        f"angular_velocity=({av.x:.3f}, {av.y:.3f}, {av.z:.3f}) "
        f"linear_acceleration=({la.x:.3f}, {la.y:.3f}, {la.z:.3f})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("iface", nargs="?", default=None,
                         help="DDS network interface (e.g. eth0). Omit to use CYCLONEDDS_URI/default config.")
    parser.add_argument("--topic-imu", default="rt/utlidar/imu_livox_mid360")
    parser.add_argument("--topic-cloud", default="rt/utlidar/cloud_livox_mid360")
    parser.add_argument("--once", action="store_true", help="Print one cloud snapshot then exit")
    parser.add_argument("--save", default=None, help="Save decoded XYZ points to this .npy path")
    parser.add_argument("--view", action="store_true", help="Open an Open3D viewer window for one snapshot")
    args = parser.parse_args()

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    imu_sub = ChannelSubscriber(args.topic_imu, Imu_)
    imu_sub.Init()

    cloud_sub = ChannelSubscriber(args.topic_cloud, PointCloud2_)
    cloud_sub.Init()

    print(f"Subscribed to '{args.topic_imu}' (Imu_) and "
          f"'{args.topic_cloud}' (PointCloud2_). Waiting for data... "
          f"(Ctrl+C to stop)")

    got_cloud = False
    try:
        while True:
            imu_msg = imu_sub.Read()
            if imu_msg is not None:
                print_imu(imu_msg)

            cloud_msg = cloud_sub.Read()
            if cloud_msg is not None:
                points = decode_xyz(cloud_msg)
                print(f"[PointCloud2] width={cloud_msg.width} height={cloud_msg.height} "
                      f"point_step={cloud_msg.point_step} n_points={len(points)}")
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    zs = [p[2] for p in points]
                    print(f"  x range: [{min(xs):.2f}, {max(xs):.2f}]  "
                          f"y range: [{min(ys):.2f}, {max(ys):.2f}]  "
                          f"z range: [{min(zs):.2f}, {max(zs):.2f}]")

                if args.save:
                    import numpy as np
                    np.save(args.save, np.asarray(points, dtype="float32"))
                    print(f"  Saved {len(points)} points to {args.save}")

                if args.view:
                    try:
                        import numpy as np
                        import open3d as o3d
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype="float64"))
                        o3d.visualization.draw_geometries([pcd])
                    except ImportError:
                        print("  [--view requires open3d: pip install open3d]")

                got_cloud = True
                if args.once:
                    break

            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

    if args.once and not got_cloud:
        print("No PointCloud2_ message received. Check --topic-cloud / network interface.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
