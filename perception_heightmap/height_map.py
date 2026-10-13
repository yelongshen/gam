#!/usr/bin/env python3
"""height_map.py
================
Convert the fused elevation point cloud published by `elevation_mapping_humanoid`
(https://github.com/ArthurAllshire/elevation_mapping_humanoid, a fork of
smoggy-P/elevation_mapping_humanoid, based on ANYbotics' `elevation_mapping`) into a
local, egocentric 2D heightmap grid around the robot, and republish it over ZMQ using
the same wire format that `view_lidar_client.py` already parses/visualizes
(topic + 1280B JSON header + concatenated binary field data, see `parse_message`
there and `HEADER_SIZE`).

This mirrors what VideoMimic's `videomimic_inference_real.cpp` does on-robot:
  - subscribes to `/elevation_map_fused_visualization/elevation_cloud`
    (a `sensor_msgs/PointCloud2`, already fused/denoised over time + odometry by
    the elevation_mapping node -- NOT the raw single-frame LiDAR scan)
  - builds an egocentric grid of shape (grid_size, grid_size), by default
    11x11 cells @ 0.1m resolution => 1.1m x 1.1m extent centered on the robot
    (i.e. "surrounding ~1m"), using a kNN + inverse-distance-weighted (IDW)
    height interpolation from the fused cloud's XY projection, with a
    default/sentinel height for cells too far from any return.

Unlike the C++ node, this version:
  - Runs standalone from Python via `rospy` (needs to run on the robot / a
    machine on the same ROS master, since it subscribes to a real ROS topic).
  - Uses a simple grid + IDW-over-kNN via a KD-tree (scipy) instead of PCL.
  - Publishes the resulting grid over ZMQ so it can be visualized with
    `view_lidar_client.py --view-heightmap` or consumed by any ZMQ subscriber,
    instead of feeding a TorchScript policy directly.

RUNS ON THE ROBOT (or wherever the elevation_mapping ROS node + ROS master are
reachable). Requires `rospy` + a sourced ROS environment, plus `scipy`.

Usage
-----
    # On the Jetson / robot, after `roslaunch elevation_mapping_demos ...`:
    python3 perception_heightmap/height_map.py --zmq-port 5558

    # From a workstation, view the result:
    .venv_sim/bin/python perception_heightmap/view_lidar_client.py \\
        --host <robot-ip> --port 5558 --view-heightmap --max-range 1.0
"""

import argparse
import json
import time

import numpy as np
import zmq

# Must match view_lidar_client.py's HEADER_SIZE / parse_message wire format.
HEADER_SIZE = 1280

DTYPE_NAME_MAP = {
    np.dtype(np.float32): "f32",
    np.dtype(np.float64): "f64",
    np.dtype(np.int32): "i32",
    np.dtype(np.int64): "i64",
    np.dtype(np.bool_): "bool",
}


def pack_message(topic: str, fields: dict) -> bytes:
    """Pack {name: np.ndarray} into [topic][1280B JSON header][binary payload].

    This is the inverse of `parse_message` in `view_lidar_client.py`.
    """
    header_fields = []
    payload = bytearray()
    for name, arr in fields.items():
        arr = np.ascontiguousarray(arr)
        dtype_name = DTYPE_NAME_MAP.get(arr.dtype, "f32")
        if dtype_name == "f32" and arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        header_fields.append({
            "name": name,
            "dtype": dtype_name,
            "shape": list(arr.shape),
        })
        payload += arr.tobytes()

    header = json.dumps({"fields": header_fields}).encode("utf-8")
    if len(header) > HEADER_SIZE:
        raise ValueError(
            f"heightmap header ({len(header)}B) exceeds HEADER_SIZE={HEADER_SIZE}B; "
            "reduce the number/size of field names or raise HEADER_SIZE in both "
            "this script and view_lidar_client.py")
    header = header.ljust(HEADER_SIZE, b"\x00")

    return topic.encode("utf-8") + header + bytes(payload)


def points_to_heightmap(
    points_xyz: np.ndarray,
    grid_size: int = 11,
    cell_size: float = 0.1,
    knn_k: int = 3,
    knn_max_distance: float = 0.15,
    empty_sentinel: float = -1e6,
    default_height: float = 0.0,
) -> np.ndarray:
    """Build an egocentric (grid_size x grid_size) height grid around the origin.

    points_xyz: (N, 3) array already expressed in the robot's local/base frame
        (i.e. the origin (0, 0) in XY is directly below/at the robot). If your
        elevation cloud is in odom/map frame, subtract the robot's current XY
        (and rotate by -yaw if you want a body-yaw-aligned grid) before calling
        this function.
    grid_size: number of cells per side (VideoMimic uses 11 -> ~1.1m extent
        at cell_size=0.1m).
    cell_size: metres per cell (VideoMimic uses 0.1m).
    knn_k: number of nearest neighbors (in XY) to inverse-distance-weight
        together for each cell's height (VideoMimic uses 3).
    knn_max_distance: if the single nearest neighbor is farther than this
        (metres) from a cell center, treat that cell as having no return, i.e.
        empty_sentinel is returned. Set to 0/None to disable this check.
    empty_sentinel: fill value for cells with no nearby returns (used by
        `view_lidar_client.py`'s `print_heightmap`/`heightmap_to_image` to
        detect "no data" cells; must stay <<< any real height in metres).
    default_height: used only to seed the "closest point too far away" default;
        the actual returned value in that case is `empty_sentinel`, matching
        the existing lidar_heightmap wire format's convention (any consumer
        that wants VideoMimic's own KNN_DEFAULT_HEIGHT_OFFSET behavior should
        replace empty_sentinel cells with that offset downstream).
    """
    from scipy.spatial import cKDTree

    half_extent = grid_size / 2.0 * cell_size
    grid = np.full((grid_size, grid_size), empty_sentinel, dtype=np.float32)

    if points_xyz.shape[0] == 0:
        return grid

    xy = points_xyz[:, :2].astype(np.float64)
    z = points_xyz[:, 2].astype(np.float64)
    tree = cKDTree(xy)

    # Cell centers: j -> Y index (rows), i -> X index (cols), matching the
    # convention used in videomimic_inference_real.cpp's ElevationCloudCallback.
    ys = -half_extent + (np.arange(grid_size) + 0.5) * cell_size
    xs = -half_extent + (np.arange(grid_size) + 0.5) * cell_size
    grid_x, grid_y = np.meshgrid(xs, ys)  # both (grid_size, grid_size)
    query_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    k = min(knn_k, xy.shape[0])
    dists, idxs = tree.query(query_xy, k=k)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    nearest_dist = dists[:, 0]
    valid = np.ones_like(nearest_dist, dtype=bool)
    if knn_max_distance and knn_max_distance > 0:
        valid = nearest_dist <= knn_max_distance

    # Inverse-distance weighting over the k neighbors (guard div-by-zero for
    # an exact coincidence with a sample point).
    eps = 1e-9
    weights = 1.0 / np.maximum(dists, eps)
    weights /= weights.sum(axis=1, keepdims=True)
    heights = (weights * z[idxs]).sum(axis=1)

    flat = grid.reshape(-1)
    flat[valid] = heights[valid].astype(np.float32)
    return grid


def pointcloud2_to_xyz(msg) -> np.ndarray:
    """Convert a `sensor_msgs/PointCloud2` into an (N, 3) float32 xyz array.

    Avoids a hard dependency on `ros_numpy` / `sensor_msgs_py` by decoding the
    raw buffer directly using the message's own field offsets/datatype, which
    works for the common case of packed float32 x, y, z (optionally with
    additional fields such as intensity/rgb that we simply ignore).
    """
    import sensor_msgs.point_cloud2 as pc2

    pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def run_ros_node(args):
    import rospy
    from sensor_msgs.msg import PointCloud2

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://0.0.0.0:{args.zmq_port}")
    print(f"[height_map] ZMQ PUB bound on tcp://0.0.0.0:{args.zmq_port} "
          f"topic={args.zmq_topic!r}")

    state = {"n": 0, "t_stat": time.time()}

    def callback(msg: PointCloud2):
        t0 = time.time()
        xyz = pointcloud2_to_xyz(msg)

        if args.center_xy is not None:
            xyz = xyz.copy()
            xyz[:, 0] -= args.center_xy[0]
            xyz[:, 1] -= args.center_xy[1]

        grid = points_to_heightmap(
            xyz,
            grid_size=args.grid_size,
            cell_size=args.cell_size,
            knn_k=args.knn_k,
            knn_max_distance=args.knn_max_distance,
            empty_sentinel=args.empty_sentinel,
        )

        half_extent = args.grid_size / 2.0 * args.cell_size
        payload = pack_message(args.zmq_topic, {
            "grid": grid,
            "grid_size": np.asarray([args.grid_size], dtype=np.float32),
            "half_extent": np.asarray([half_extent], dtype=np.float32),
            "cell_size": np.asarray([args.cell_size], dtype=np.float32),
            "empty_sentinel": np.asarray([args.empty_sentinel], dtype=np.float32),
        })
        sock.send(payload)

        state["n"] += 1
        now = time.time()
        if now - state["t_stat"] >= 2.0:
            dt_ms = (now - t0) * 1e3
            valid = float((grid > (args.empty_sentinel / 2)).mean()) * 100.0
            print(f"[height_map] {state['n']} msgs so far, last cb took {dt_ms:.1f}ms, "
                  f"{valid:.1f}% cells valid, n_points={xyz.shape[0]}")
            state["t_stat"] = now

    rospy.init_node("height_map_zmq_bridge", anonymous=True)
    rospy.Subscriber(args.ros_topic, PointCloud2, callback, queue_size=1)
    print(f"[height_map] subscribed to {args.ros_topic!r}, publishing '{args.zmq_topic}' "
          f"grid={args.grid_size}x{args.grid_size} cell_size={args.cell_size}m "
          f"(extent={args.grid_size * args.cell_size:.2f}m)")
    rospy.spin()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ros-topic", default="/elevation_map_fused_visualization/elevation_cloud",
                     help="sensor_msgs/PointCloud2 topic published by elevation_mapping_humanoid")
    ap.add_argument("--zmq-port", type=int, default=5558)
    ap.add_argument("--zmq-topic", default="lidar_heightmap")
    ap.add_argument("--grid-size", type=int, default=11,
                     help="cells per side (VideoMimic default: 11 -> ~1.1m extent @ 0.1m cells)")
    ap.add_argument("--cell-size", type=float, default=0.1, help="metres per cell")
    ap.add_argument("--knn-k", type=int, default=3)
    ap.add_argument("--knn-max-distance", type=float, default=0.15,
                     help="reject a cell (mark empty) if nearest return is farther than this (m)")
    ap.add_argument("--empty-sentinel", type=float, default=-1e6)
    ap.add_argument("--center-x", type=float, default=None,
                     help="if the elevation cloud is in odom/map frame (not already "
                          "robot-relative), subtract this X (m) before gridding")
    ap.add_argument("--center-y", type=float, default=None)
    args = ap.parse_args()

    args.center_xy = None
    if args.center_x is not None and args.center_y is not None:
        args.center_xy = (args.center_x, args.center_y)

    run_ros_node(args)


if __name__ == "__main__":
    main()
