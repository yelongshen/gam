#!/usr/bin/env python3
"""Publish live Mid-360 LiDAR point clouds (+ base orientation) from the G1's
onboard computer over ZMQ, using the SAME wire format `view_lidar_client.py`
already parses (`parse_message`, `HEADER_SIZE=1280`, matching
`gear_sonic.utils.teleop.zmq.zmq_planner_sender`).

RUNS ON THE ROBOT. Pair with `perception_heightmap/g1_lidar_subscriber_elevation.py`
on the workstation (analogous to how `g1_depth_publisher.py` pairs with
`g1_depth_subscriber.py` for the depth-camera path).

Sensor path (see `dev_notes/heightmap_architecture_analysis.md` and the URDF at
`gear_sonic/data/assets/robot_description/urdf/g1/main.urdf`):
    mid360_link is mounted on torso_link at xyz=(0.0002835, 0.00003, 0.41618),
    rpy=(0, 3.101, 3.1415) -- i.e. ~0.42m above the pelvis and mounted UPSIDE
    DOWN (pitch ~pi plus yaw pi is equivalent to a ~180 deg roll, with ~2.3 deg
    of residual pitch). The points published here are in the raw
    `livox_frame`, so the floor sits at z ~ +1.3 m; consumers must transform
    them (see `g1_frames.py`, verified against the MID-360 IMU and the depth
    camera) before building a height map. The `mid360_joint` in the repo's
    decoupled_wbc/**/g1*.urdf (rpy=(0, 0.04, 0)) does NOT match this data.

Unitree exposes the Mid-360 over the SAME DDS/CycloneDDS bus used by the
rest of the low-level SDK (see `external_dependencies/unitree_sdk2_python`),
publishing `sensor_msgs.msg.dds_.PointCloud2_` messages -- the standard ROS
PointCloud2 wire layout (packed binary `data` + `fields` describing offsets/
datatypes), NOT a ROS topic requiring `rospy`/a ROS master. This script
decodes that IDL message directly via `unitree_sdk2py`, with no ROS
dependency at all (unlike `height_map.py`, which bridges an *already
fused* elevation-mapping ROS topic).

NOTE on the exact DDS topic name: Unitree's own examples
(`external_dependencies/unitree_sdk2_python/example/go2/high_level/
go2_utlidar_switch.py`) only demonstrate the `rt/utlidar/switch` control
topic, not the cloud topic itself. The correct topic on THIS G1 has now been
verified live as `rt/utlidar/cloud_livox_mid360` (NOT the conventional
`rt/utlidar/cloud`, which does not exist here), via:

    source /opt/ros/foxy/setup.bash
    source ~/cyclonedds_ws/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=~/cyclonedds_ws/cyclonedds.xml
    ros2 topic list | grep -i lidar
    ros2 topic info -v /utlidar/cloud_livox_mid360   # Publisher count: 1

Important: `ros2 topic list` displays these WITHOUT the `rt/` prefix (e.g.
`utlidar/cloud_livox_mid360`) -- that's just how the ROS2 CLI renders ROS
topic names. `rmw_cyclonedds` publishes the *raw DDS* topic name with an
`rt/` prefix, and this script's `ChannelSubscriber` talks directly to
CycloneDDS (bypassing ROS2's rmw layer), so the `rt/` prefix IS required
here. Verified payload: ~20k points/frame @ ~10 Hz, point_step=22,
frame_id="livox_frame", fields = x/y/z/intensity/ring/time.
Pass `--lidar-topic` to override if your firmware differs.

First, ensure the LiDAR is switched on (see `go2_utlidar_switch.py`'s
pattern -- G1 uses the same `rt/utlidar/switch` topic):
    python3 -c "
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    ChannelFactoryInitialize(0)
    pub = ChannelPublisher('rt/utlidar/switch', String_)
    pub.Init()
    pub.Write(String_(data='ON'))
    "

Usage (on robot):
    python3 g1_lidar_publisher.py --iface eth0 --port 5558

Publishes two ZMQ topics from the same PUB socket (subscribers filter by
topic prefix, so a client asking only for one never receives the other):
    "lidar_cloud" -- (N, 3) float32 xyz points  (from rt/utlidar/cloud_livox_mid360)
    "lidar_imu"   -- orientation / angular_velocity / linear_acceleration
                     (from rt/utlidar/imu_livox_mid360); pass --no-imu to skip.

Both match the field names `view_lidar_client.py` already parses, so:
    python3 view_lidar_client.py --host <robot-ip> --port 5558
    python3 view_lidar_client.py --host <robot-ip> --port 5558 --stream imu
"""
import argparse
import json
import struct
import sys
import time

import numpy as np
import zmq

try:
    import lz4.frame as lz4frame
except ImportError:
    lz4frame = None

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import Imu_, PointCloud2_

# Must match gear_sonic.utils.teleop.zmq.zmq_planner_sender.HEADER_SIZE and
# perception_heightmap/view_lidar_client.py's HEADER_SIZE.
HEADER_SIZE = 1280

# PointField datatype codes, per the ROS sensor_msgs/PointField convention
# (also what unitree_sdk2py's PointField_ IDL uses).
_PF_DTYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}
_PF_DTYPE_NP = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}


def decode_pointcloud2_xyz(msg: PointCloud2_) -> np.ndarray:
    """Decode a `sensor_msgs/PointCloud2` DDS message into an (N, 3) float32
    xyz array, using the message's own `fields`/`point_step` (works whether
    or not extra fields like intensity/timestamp are interleaved), matching
    `height_map.py`'s `pointcloud2_to_xyz` (which instead uses `rospy`'s
    `sensor_msgs.point_cloud2.read_points` over an actual ROS topic -- here
    we do the equivalent decode ourselves since this is a raw DDS message,
    not a ROS message).
    """
    field_offsets = {}
    for f in msg.fields:
        field_offsets[f.name] = (int(f.offset), _PF_DTYPE_NP.get(int(f.datatype), np.float32))

    if not all(name in field_offsets for name in ("x", "y", "z")):
        return np.zeros((0, 3), dtype=np.float32)

    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    n_points = int(msg.width) * int(msg.height)
    point_step = int(msg.point_step)
    if n_points == 0 or point_step == 0:
        return np.zeros((0, 3), dtype=np.float32)

    data = data[: n_points * point_step].reshape(n_points, point_step)

    def _extract(name):
        off, dt = field_offsets[name]
        itemsize = np.dtype(dt).itemsize
        # Slicing a column out of the (n_points, point_step) byte matrix gives a
        # *strided* view (rows are point_step bytes apart, e.g. 22 for the
        # Mid-360's x/y/z/intensity/ring/time layout), and numpy refuses
        # `.view(dt)` on non-C-contiguous data when the dtype size changes
        # ("To change to a dtype of a different size, the array must be
        # C-contiguous"). Copy the column into contiguous memory first.
        col = np.ascontiguousarray(data[:, off: off + itemsize])
        return col.view(dt).reshape(-1)

    x = _extract("x").astype(np.float32)
    y = _extract("y").astype(np.float32)
    z = _extract("z").astype(np.float32)
    xyz = np.stack([x, y, z], axis=1)

    finite = np.all(np.isfinite(xyz), axis=1)
    return xyz[finite]


def decode_pointcloud2_full(msg: PointCloud2_) -> dict:
    """Decode a `sensor_msgs/PointCloud2` DDS message into a dict of ALL the
    per-point fields we care about, not just xyz.

    Needed for the FAST-LIO path (`--fastlio`): FAST-LIO's `lidar_type: 1`
    consumes Livox `CustomMsg`, whose `CustomPoint` carries
    {offset_time, x, y, z, reflectivity, tag, line}. The `offset_time` is
    what lets it de-skew the non-repetitive Mid-360 scan against the IMU --
    drop it and odometry degrades badly during motion (see
    `dev_notes/heightmap_architecture_analysis.md`).

    This G1's Mid-360 publishes point_step=22 with
    fields = x/y/z/intensity/ring/time (verified live), so:
        time      -> CustomPoint.offset_time
        intensity -> CustomPoint.reflectivity
        ring      -> CustomPoint.line

    Returns {"xyz": (N,3) f32, "time": (N,) f32 or None,
             "intensity": (N,) f32 or None, "ring": (N,) f32 or None},
    already filtered to finite xyz (with the same mask applied to every
    other field so they stay index-aligned).
    """
    field_meta = {}
    for f in msg.fields:
        field_meta[f.name] = (int(f.offset), _PF_DTYPE_NP.get(int(f.datatype), np.float32))

    empty = {"xyz": np.zeros((0, 3), dtype=np.float32),
             "time": None, "intensity": None, "ring": None}
    if not all(name in field_meta for name in ("x", "y", "z")):
        return empty

    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    n_points = int(msg.width) * int(msg.height)
    point_step = int(msg.point_step)
    if n_points == 0 or point_step == 0:
        return empty
    data = data[: n_points * point_step].reshape(n_points, point_step)

    def _extract(name):
        if name not in field_meta:
            return None
        off, dt = field_meta[name]
        itemsize = np.dtype(dt).itemsize
        # Same contiguity caveat as decode_pointcloud2_xyz -- see its comment.
        col = np.ascontiguousarray(data[:, off: off + itemsize])
        return col.view(dt).reshape(-1)

    x, y, z = (_extract(n).astype(np.float32) for n in ("x", "y", "z"))
    xyz = np.stack([x, y, z], axis=1)
    finite = np.all(np.isfinite(xyz), axis=1)

    out = {"xyz": xyz[finite]}
    for name, key in (("time", "time"), ("intensity", "intensity"), ("ring", "ring")):
        col = _extract(name)
        out[key] = col.astype(np.float32)[finite] if col is not None else None
    return out


def pack_message(topic: str, fields: dict) -> bytes:
    """[topic bytes][1280B JSON header][concatenated binary field data],
    matching `zmq_planner_sender._build_header` / `view_lidar_client.parse_message`.
    """
    header_fields = []
    payload = bytearray()
    dtype_name_map = {
        np.dtype(np.float32): "f32", np.dtype(np.float64): "f64",
        np.dtype(np.int32): "i32", np.dtype(np.int64): "i64",
        np.dtype(np.bool_): "bool",
    }
    for name, arr in fields.items():
        arr = np.ascontiguousarray(arr)
        dtype_name = dtype_name_map.get(arr.dtype, "f32")
        if dtype_name == "f32" and arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        header_fields.append({"name": name, "dtype": dtype_name, "shape": list(arr.shape)})
        payload += arr.tobytes()

    header = json.dumps({"fields": header_fields}).encode("utf-8")
    if len(header) > HEADER_SIZE:
        raise ValueError(f"header ({len(header)}B) exceeds HEADER_SIZE={HEADER_SIZE}B")
    header = header.ljust(HEADER_SIZE, b"\x00")
    return topic.encode("utf-8") + header + bytes(payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0", help="network interface for DDS (per unitree SDK examples)")
    ap.add_argument("--domain-id", type=int, default=0)
    ap.add_argument("--lidar-topic", default="rt/utlidar/cloud_livox_mid360",
                     help="DDS topic for the Mid-360 point cloud (verified live on this G1 "
                          "via `ros2 topic list`; note ros2 shows it without the 'rt/' prefix)")
    ap.add_argument("--imu-topic", default="rt/utlidar/imu_livox_mid360",
                     help="DDS topic for the Mid-360's built-in IMU (same 'rt/' prefix note)")
    ap.add_argument("--bind", default="tcp://0.0.0.0")
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--zmq-topic", default="lidar_cloud")
    ap.add_argument("--zmq-imu-topic", default="lidar_imu",
                     help="ZMQ topic for the IMU stream (matches view_lidar_client's --topic-state)")
    ap.add_argument("--no-imu", action="store_true",
                     help="publish only the point cloud, skip the IMU stream")
    ap.add_argument("--imu-send-hz", type=float, default=0.0,
                     help="throttle IMU publishing to this rate (0 = every message, ~200Hz native)")
    ap.add_argument("--send-hz", type=float, default=10.0,
                     help="throttle publishing to this rate (0 = every message)")
    ap.add_argument("--max-range", type=float, default=5.0,
                     help="drop points farther than this (metres, 3D) from the sensor before publishing")
    ap.add_argument("--min-range", type=float, default=0.35,
                     help="drop points closer than this (metres, 3D): removes the MID-360's (0,0,0) "
                          "no-return points and head/shoulder self-hits. Applied in --fastlio mode too.")
    ap.add_argument("--fastlio", action="store_true",
                     help="FAST-LIO mode: forward the FULL cloud with per-point time/"
                          "intensity/ring fields (needed to synthesize Livox CustomMsg on "
                          "the desktop for de-skewing), and default to NO throttling and NO "
                          "range cropping -- both of which are fine for visualization but "
                          "harmful for odometry. Explicit --send-hz/--max-range still win.")
    ap.add_argument("--stats-every", type=float, default=5.0)
    args = ap.parse_args()

    # In --fastlio mode, default to lossless streaming unless the user
    # explicitly asked otherwise on the command line.
    if args.fastlio:
        argv = sys.argv[1:]
        if not any(a.startswith("--send-hz") for a in argv):
            args.send_hz = 0.0
        if not any(a.startswith("--max-range") for a in argv):
            args.max_range = 0.0

    ChannelFactoryInitialize(args.domain_id, args.iface)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2)  # drop old frames rather than queue
    sock.bind(f"{args.bind}:{args.port}")
    print(f"[lidar_pub] bound {args.bind}:{args.port} zmq_topic={args.zmq_topic!r} "
          f"dds_topic={args.lidar_topic!r}", flush=True)
    if not args.no_imu:
        print(f"[lidar_pub] IMU enabled: zmq_topic={args.zmq_imu_topic!r} "
              f"dds_topic={args.imu_topic!r}", flush=True)

    state = {"n": 0, "bytes": 0, "t_stat": time.time(), "t_last_send": 0.0}
    min_dt = (1.0 / args.send_hz) if args.send_hz > 0 else 0.0
    imu_state = {"n": 0, "t_last_send": 0.0}
    imu_min_dt = (1.0 / args.imu_send_hz) if args.imu_send_hz > 0 else 0.0

    def callback(msg: PointCloud2_):
        now = time.time()
        if min_dt and (now - state["t_last_send"]) < min_dt:
            return
        state["t_last_send"] = now

        if args.fastlio:
            decoded = decode_pointcloud2_full(msg)
            xyz = decoded["xyz"]
        else:
            decoded = None
            xyz = decode_pointcloud2_xyz(msg)

        if xyz.shape[0] == 0:
            return
        # Always drop points inside --min-range: the MID-360 reports no-returns
        # as exact (0, 0, 0) (~30% of each frame here), which would otherwise
        # show up as a phantom obstacle at the sensor origin.
        dist = np.linalg.norm(xyz, axis=1)
        keep = dist > args.min_range
        if args.max_range > 0:
            keep &= dist <= args.max_range
        xyz = xyz[keep]
        if xyz.shape[0] == 0:
            return
        if decoded is not None:
            for k in ("time", "intensity", "ring"):
                if decoded[k] is not None:
                    decoded[k] = decoded[k][keep]

        payload = {
            "points": xyz,
            "width": np.array([xyz.shape[0]], dtype=np.int32),
            "height": np.array([1], dtype=np.int32),
            "stamp_host": np.array([now], dtype=np.float64),
        }
        if decoded is not None:
            # Sensor-side frame stamp: the desktop bridge uses this as
            # CustomMsg.timebase, with per-point `time` as offset_time.
            payload["stamp_sec"] = np.array([msg.header.stamp.sec], dtype=np.int64)
            payload["stamp_nanosec"] = np.array([msg.header.stamp.nanosec], dtype=np.int64)
            for k in ("time", "intensity", "ring"):
                if decoded[k] is not None:
                    payload[k] = decoded[k]

        packed = pack_message(args.zmq_topic, payload)
        sock.send(packed)

        state["n"] += 1
        state["bytes"] += len(packed)
        if now - state["t_stat"] > args.stats_every:
            rate = state["n"] / (now - state["t_stat"])
            mbps = state["bytes"] / (now - state["t_stat"]) / 1e6
            print(f"[lidar_pub] {rate:.1f} msg/s, {mbps:.2f} MB/s, "
                  f"last n_points={xyz.shape[0]}, imu_msgs={imu_state['n']}", flush=True)
            state["n"] = 0
            state["bytes"] = 0
            imu_state["n"] = 0
            state["t_stat"] = now

    def imu_callback(msg: Imu_):
        """Forward the Mid-360's built-in IMU on its own ZMQ topic.

        Field names/shapes match `view_lidar_client.print_state` (which reads
        orientation / angular_velocity / linear_acceleration / stamp_*), so the
        existing client parses this with no changes.
        """
        now = time.time()
        if imu_min_dt and (now - imu_state["t_last_send"]) < imu_min_dt:
            return
        imu_state["t_last_send"] = now

        q = msg.orientation
        av = msg.angular_velocity
        la = msg.linear_acceleration
        sock.send(pack_message(args.zmq_imu_topic, {
            # Sensor-side stamp comes from the robot's main controller clock,
            # which free-runs ~17min off this Jetson's NTP-synced clock -- keep
            # both so downstream can pick a consistent timebase.
            "stamp_sec": np.array([msg.header.stamp.sec], dtype=np.int64),
            "stamp_nanosec": np.array([msg.header.stamp.nanosec], dtype=np.int64),
            "stamp_host": np.array([now], dtype=np.float64),
            "orientation": np.array([q.x, q.y, q.z, q.w], dtype=np.float32),
            "angular_velocity": np.array([av.x, av.y, av.z], dtype=np.float32),
            "linear_acceleration": np.array([la.x, la.y, la.z], dtype=np.float32),
        }))
        imu_state["n"] += 1

    sub = ChannelSubscriber(args.lidar_topic, PointCloud2_)
    sub.Init(callback, 10)
    print(f"[lidar_pub] subscribed to DDS topic {args.lidar_topic!r}, publishing forever "
          f"(Ctrl-C to stop)...", flush=True)

    imu_sub = None
    if not args.no_imu:
        imu_sub = ChannelSubscriber(args.imu_topic, Imu_)
        imu_sub.Init(imu_callback, 10)
        print(f"[lidar_pub] subscribed to DDS topic {args.imu_topic!r} (IMU)", flush=True)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[lidar_pub] stopping.", file=sys.stderr)


if __name__ == "__main__":
    main()
