#!/usr/bin/env python3
"""check_elevation_map.py
========================
Sanity-check / inspect the `grid_map_msgs/GridMap` produced by
`elevation_mapping` in the ZMQ LiDAR pipeline, without needing rviz.

Where this sits in the stack (see `zmq_to_ros_livox_bridge.py` for the
upstream half):

    [ROBOT]   g1_lidar_publisher.py --fastlio
                 |  ZMQ
    [DESKTOP] zmq_to_ros_livox_bridge.py   -> /livox/lidar + /livox/imu
                 |
              fast_lio (laserMapping)      -> /cloud_registered  (frame "odom")
                 |
              pc_filter.py                 -> /cloud_registered/filtered ("torso_link")
                 |
              elevation_mapping            -> /elevation_mapping/elevation_map_raw
                                              (grid_map_msgs/GridMap, frame "odom_torso")

A GridMap stores each layer as a `std_msgs/Float32MultiArray` in COLUMN-MAJOR
order with the origin at the map CENTRE and +x pointing "up"/"back" along the
first index -- i.e. `data[i]` is NOT simply row-major image order, which is the
usual reason a hand-rolled decode looks transposed/mirrored. We reproduce
grid_map's own convention here (see grid_map_core's `GridMap::at`):

    layer.layout.dim[0] -> "column_index", size = n_cols
    layer.layout.dim[1] -> "row_index",    size = n_rows

Unobserved cells are NaN (that is normal and expected around the map edges);
a map that is 100% NaN means no points ever landed in it -- usually a TF or
`sensor_processor/ignore_points_above` / `pc_filter` threshold problem rather
than a LiDAR problem.

Usage:
    conda activate ros_noetic && source ~/ros_ws/devel/setup.bash
    python perception_heightmap/check_elevation_map.py                 # text stats
    python perception_heightmap/check_elevation_map.py --ascii         # + ASCII map
    python perception_heightmap/check_elevation_map.py --save map.png  # + heatmap
    python perception_heightmap/check_elevation_map.py \
        --topic /elevation_mapping/elevation_map --layer elevation
"""
import argparse
import sys

import numpy as np

import rospy
from grid_map_msgs.msg import GridMap


def gridmap_layer_to_array(msg: GridMap, layer: str) -> np.ndarray:
    """Return the named layer as a (n_rows, n_cols) float32 array.

    grid_map serializes column-major with both dims reversed relative to the
    natural image layout, so decoding is: reshape to (n_cols, n_rows) then
    transpose. Row 0 / col 0 is the map's +x / +y corner.
    """
    if layer not in msg.layers:
        raise KeyError(f"layer {layer!r} not in {list(msg.layers)}")
    arr_msg = msg.data[msg.layers.index(layer)]
    n_cols = arr_msg.layout.dim[0].size
    n_rows = arr_msg.layout.dim[1].size
    return np.asarray(arr_msg.data, dtype=np.float32).reshape(n_cols, n_rows).T


def to_point_cloud(grid: np.ndarray, msg: GridMap) -> np.ndarray:
    """Lift the 2.5-D height grid into an (N, 3) xyz cloud in the map frame.

    This is the "3D elevation map" form: each finite cell becomes a point at
    its own cell centre. Cell (0, 0) sits at the +x/+y corner, and x/y DECREASE
    with increasing row/column index (grid_map's convention).
    """
    n_rows, n_cols = grid.shape
    res = msg.info.resolution
    cx, cy = msg.info.pose.position.x, msg.info.pose.position.y

    # Cell centres, measured from the map centre outward.
    xs = cx + (n_rows - 1) / 2.0 * res - np.arange(n_rows) * res
    ys = cy + (n_cols - 1) / 2.0 * res - np.arange(n_cols) * res
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    finite = np.isfinite(grid)
    return np.stack([xx[finite], yy[finite], grid[finite]], axis=1)


def render_ascii(grid: np.ndarray, width: int = 60) -> str:
    """Coarse ASCII heatmap, low -> high = ' .:-=+*#%@', NaN = ' '."""
    ramp = " .:-=+*#%@"
    n_rows, n_cols = grid.shape
    step = max(1, n_cols // width)
    sub = grid[::step, ::step]

    finite = np.isfinite(sub)
    if not finite.any():
        return "(all NaN)"
    lo, hi = np.nanmin(sub), np.nanmax(sub)
    span = max(hi - lo, 1e-6)

    lines = []
    for row in sub:
        chars = []
        for v in row:
            if not np.isfinite(v):
                chars.append(" ")
            else:
                idx = int((v - lo) / span * (len(ramp) - 1))
                chars.append(ramp[min(max(idx, 0), len(ramp) - 1)])
        lines.append("".join(chars))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/elevation_mapping/elevation_map_raw",
                    help="GridMap topic ('..._raw' updates fastest; "
                         "'/elevation_mapping/elevation_map' is the fused one)")
    ap.add_argument("--layer", default="elevation")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--ascii", action="store_true", help="print an ASCII heatmap")
    ap.add_argument("--save", default=None, help="write a matplotlib heatmap PNG here")
    ap.add_argument("--save-npz", default=None,
                    help="write the raw grid + the lifted (N,3) point cloud to this .npz")
    args = ap.parse_args()

    rospy.init_node("check_elevation_map", anonymous=True, disable_signals=True)
    print(f"[check] waiting up to {args.timeout:.0f}s for {args.topic} ...", flush=True)
    try:
        msg = rospy.wait_for_message(args.topic, GridMap, timeout=args.timeout)
    except rospy.ROSException:
        print(f"[check] TIMEOUT: nothing published on {args.topic}.\n"
              f"        Check, in order: /livox/lidar (bridge), /fastlio_odom "
              f"(FAST-LIO), /cloud_registered/filtered (pc_filter).", file=sys.stderr)
        return 1

    print(f"[check] frame_id      : {msg.info.header.frame_id}")
    print(f"[check] layers        : {list(msg.layers)}")
    print(f"[check] size          : {msg.info.length_x:.2f} x {msg.info.length_y:.2f} m "
          f"@ {msg.info.resolution:.3f} m/cell")
    print(f"[check] centre (x, y) : ({msg.info.pose.position.x:.3f}, "
          f"{msg.info.pose.position.y:.3f})")

    grid = gridmap_layer_to_array(msg, args.layer)
    finite = np.isfinite(grid)
    n_valid, n_total = int(finite.sum()), grid.size
    print(f"[check] grid shape    : {grid.shape}")
    print(f"[check] valid cells   : {n_valid}/{n_total} ({100.0 * n_valid / n_total:.1f}%)")

    if n_valid == 0:
        print("[check] *** map is entirely NaN -- no points were fused. ***\n"
              "        Most likely causes:\n"
              "          - TF odom_torso -> torso_link missing/stale (FAST-LIO not converged)\n"
              "          - pc_filter's ~distance_threshold rejecting every point\n"
              "          - elevation_mapping's sensor_processor/ignore_points_above too low",
              file=sys.stderr)
        return 1

    z = grid[finite]
    print(f"[check] elevation     : min={z.min():+.3f} max={z.max():+.3f} "
          f"mean={z.mean():+.3f} std={z.std():.3f} m")

    cloud = to_point_cloud(grid, msg)
    print(f"[check] 3-D points    : {cloud.shape[0]}")

    if args.ascii:
        print("\n[check] ASCII elevation map (low ' .:-=+*#%@' high, ' ' = unobserved):")
        print(render_ascii(grid))

    if args.save_npz:
        np.savez(args.save_npz, grid=grid, points=cloud,
                 resolution=msg.info.resolution,
                 center=np.array([msg.info.pose.position.x, msg.info.pose.position.y]),
                 frame_id=msg.info.header.frame_id)
        print(f"[check] wrote {args.save_npz}")

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        half_x, half_y = msg.info.length_x / 2.0, msg.info.length_y / 2.0
        cx, cy = msg.info.pose.position.x, msg.info.pose.position.y
        fig, ax = plt.subplots(figsize=(6, 5))
        # extent uses (left, right, bottom, top) in map coords; the grid's first
        # index runs along -x and its second along -y, so flip both axes.
        im = ax.imshow(grid, origin="upper", cmap="terrain",
                       extent=(cy + half_y, cy - half_y, cx - half_x, cx + half_x))
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
        ax.set_title(f"{args.layer} ({msg.info.header.frame_id})")
        fig.colorbar(im, ax=ax, label="height [m]")
        fig.tight_layout()
        fig.savefig(args.save, dpi=130)
        print(f"[check] wrote {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
