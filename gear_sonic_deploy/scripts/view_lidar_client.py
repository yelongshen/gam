#!/usr/bin/env python3
"""view_lidar_client.py
=======================
ZMQ client for real-time LiDAR monitoring. Connects to the publisher started
by `lidar_zmq_publisher.py` (running on/near the robot) and prints live
health stats + point-cloud summaries, with optional Open3D visualization.

Wire format (must match zmq_planner_sender.pack_pose_message):
    [topic bytes][1280-byte JSON header][concatenated binary field data]

Usage
-----
    # From a teleop PC / workstation, pointing at the robot's publisher:
    python3 gear_sonic_deploy/scripts/view_lidar_client.py --host 192.168.123.164

    # Live Open3D point-cloud viewer (updates each received frame):
    python3 gear_sonic_deploy/scripts/view_lidar_client.py --host 192.168.123.164 --view
"""

import argparse
import json
import time

import numpy as np
import zmq

# Must match gear_sonic.utils.teleop.zmq.zmq_planner_sender.HEADER_SIZE.
HEADER_SIZE = 1280

DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "bool": np.bool_,
}


def parse_message(raw: bytes, topic: str):
    """Parse the [topic][1280B JSON header][binary payload] message."""
    topic_bytes = topic.encode("utf-8")
    if not raw.startswith(topic_bytes):
        return None, f"topic prefix mismatch (expected '{topic}')"

    offset = len(topic_bytes)
    header_bytes = raw[offset: offset + HEADER_SIZE]
    offset += HEADER_SIZE

    try:
        header = json.loads(header_bytes.rstrip(b"\x00").decode("utf-8"))
    except Exception as e:
        return None, f"header decode failed: {e}"

    data = {}
    pos = offset
    for f in header.get("fields", []):
        dtype = DTYPE_MAP.get(f["dtype"], np.float32)
        shape = tuple(f["shape"])
        n_elem = int(np.prod(shape)) if shape else 1
        n_bytes = n_elem * np.dtype(dtype).itemsize
        buf = raw[pos: pos + n_bytes]
        if len(buf) < n_bytes:
            return None, f"truncated payload on field '{f['name']}'"
        data[f["name"]] = np.frombuffer(buf, dtype=dtype).reshape(shape)
        pos += n_bytes

    return data, None


def print_state(data: dict):
    ori = data["orientation"]
    av = data["angular_velocity"]
    la = data["linear_acceleration"]
    stamp = f"{int(data['stamp_sec'][0])}.{int(data['stamp_nanosec'][0]):09d}" \
        if "stamp_sec" in data else "?"
    print(
        f"[lidar_imu] stamp={stamp} "
        f"orientation(xyzw)={tuple(round(float(v), 3) for v in ori)} "
        f"angular_velocity={tuple(round(float(v), 3) for v in av)} "
        f"linear_acceleration={tuple(round(float(v), 3) for v in la)}"
    )


def print_cloud(data: dict):
    points = data["points"]
    width = int(data["width"][0]) if "width" in data else points.shape[0]
    height = int(data["height"][0]) if "height" in data else 1
    print(f"[lidar_cloud] n_points={points.shape[0]} width={width} height={height}")
    if points.shape[0] > 0:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        print(f"  x range: [{mins[0]:.2f}, {maxs[0]:.2f}]  "
              f"y range: [{mins[1]:.2f}, {maxs[1]:.2f}]  "
              f"z range: [{mins[2]:.2f}, {maxs[2]:.2f}]")


def print_heightmap(data: dict):
    grid = data["grid"]
    grid_size = int(data["grid_size"][0]) if "grid_size" in data else grid.shape[0]
    half_extent = float(data["half_extent"][0]) if "half_extent" in data else float("nan")
    cell_size = float(data["cell_size"][0]) if "cell_size" in data else float("nan")
    sentinel = float(data["empty_sentinel"][0]) if "empty_sentinel" in data else -1e6

    print(f"[lidar_heightmap] {grid_size}x{grid_size} grid, "
          f"half_extent={half_extent:.2f}m, cell_size={cell_size:.2f}m")

    # empty_sentinel marks cells with no LiDAR returns (see height_map.py).
    grid = np.where(grid <= sentinel / 2, np.nan, grid)
    for row in grid[::-1]:  # flip so +Y prints at the top
        cells = ["  nan " if np.isnan(v) else f"{v:6.2f}" for v in row]
        print("  " + " ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", type=str, default="localhost",
                     help="Publisher IP (the robot / DDS-connected machine running lidar_zmq_publisher.py)")
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--topic-state", default="lidar_imu")
    ap.add_argument("--topic-cloud", default="lidar_cloud")
    ap.add_argument("--topic-heightmap", default="lidar_heightmap")
    ap.add_argument("--stream", choices=["all", "imu", "cloud", "heightmap"], default="all",
                     help="Which stream(s) to subscribe to and display (default: all). "
                          "Use 'cloud' or 'heightmap' to view just one topic without the "
                          "other noise.")
    ap.add_argument("--seconds", type=float, default=0.0, help="Stop after N seconds (0 = run forever)")
    ap.add_argument("--view", action="store_true",
                     help="Live-update an Open3D window with the incoming point cloud (requires open3d)")
    args = ap.parse_args()

    want_imu = args.stream in ("all", "imu")
    want_cloud = args.stream in ("all", "cloud")
    want_heightmap = args.stream in ("all", "heightmap")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    url = f"tcp://{args.host}:{args.port}"
    socket.connect(url)
    if want_imu:
        socket.setsockopt_string(zmq.SUBSCRIBE, args.topic_state)
    if want_cloud:
        socket.setsockopt_string(zmq.SUBSCRIBE, args.topic_cloud)
    if want_heightmap:
        socket.setsockopt_string(zmq.SUBSCRIBE, args.topic_heightmap)
    socket.setsockopt(zmq.RCVTIMEO, 1000)


    print("=" * 70)
    print("LiDAR ZMQ client")
    print(f"  Connecting to: {url}")
    print(f"  Stream: {args.stream}")
    topics = []
    if want_imu:
        topics.append(args.topic_state)
    if want_cloud:
        topics.append(args.topic_cloud)
    if want_heightmap:
        topics.append(args.topic_heightmap)
    print(f"  Topics: {', '.join(repr(t) for t in topics)}")
    print("=" * 70)

    vis = None
    pcd = None
    if args.view and want_cloud:
        try:
            import open3d as o3d
            vis = o3d.visualization.Visualizer()
            vis.create_window("LiDAR live view")
            pcd = o3d.geometry.PointCloud()
            vis.add_geometry(pcd)
        except ImportError:
            print("[--view requires open3d: pip install open3d]")
            vis = None

    n_state = 0
    n_cloud = 0
    n_heightmap = 0
    n_timeouts = 0
    start_t = time.time()

    try:
        while args.seconds <= 0 or (time.time() - start_t) < args.seconds:
            try:
                raw = socket.recv()
            except zmq.Again:
                n_timeouts += 1
                print(f"  [timeout] no message in last 1s (total: {n_timeouts})")
                continue

            if want_imu and raw.startswith(args.topic_state.encode("utf-8")):
                data, err = parse_message(raw, args.topic_state)
                if err:
                    print(f"  [parse error/state] {err}")
                    continue
                n_state += 1
                print_state(data)

            elif want_heightmap and raw.startswith(args.topic_heightmap.encode("utf-8")):
                data, err = parse_message(raw, args.topic_heightmap)
                if err:
                    print(f"  [parse error/heightmap] {err}")
                    continue
                n_heightmap += 1
                print_heightmap(data)

            elif want_cloud and raw.startswith(args.topic_cloud.encode("utf-8")):
                data, err = parse_message(raw, args.topic_cloud)
                if err:
                    print(f"  [parse error/cloud] {err}")
                    continue
                n_cloud += 1
                print_cloud(data)

                if vis is not None:
                    try:
                        import open3d as o3d
                        pcd.points = o3d.utility.Vector3dVector(data["points"].astype(np.float64))
                        vis.update_geometry(pcd)
                        vis.poll_events()
                        vis.update_renderer()
                    except Exception as e:
                        print(f"  [view update failed] {e}")

    except KeyboardInterrupt:
        pass
    finally:
        if vis is not None:
            vis.destroy_window()

    print(f"\nReceived {n_state} state messages, {n_cloud} cloud messages, "
          f"{n_heightmap} height map messages, {n_timeouts} timeouts.")



if __name__ == "__main__":
    main()
