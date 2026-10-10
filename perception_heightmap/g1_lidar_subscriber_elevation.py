#!/usr/bin/env python3
"""Feed live Mid-360 LiDAR point clouds into `ElevationGridMap` (the
from-scratch, dependency-light reimplementation in `elevation_grid_map.py` --
NOT the real `elevation_mapping_cupy`/`elevation_mapping_humanoid` ROS nodes,
which require a full ROS+CUDA stack; see `g1_depth_subscriber_real_emc.py`
for that heavier alternative on the depth-camera path).

RUNS ON THE WORKSTATION. Pairs with `sim2real/g1_lidar_publisher.py` on the
robot (subscribes to its `lidar_cloud` ZMQ topic, wire format shared with
`view_lidar_client.py`'s `parse_message`).

Pipeline (mirrors VideoMimic's real elevation-mapping stack, see
`dev_notes/heightmap_architecture_analysis.md` and the paper-verified
LiDAR + FAST-LIO + `elevation_mapping` (ANYbotics) architecture):

    g1_lidar_publisher.py (on robot)
            |  raw Mid-360 point cloud, SENSOR frame, ZMQ topic "lidar_cloud"
            v
    [this script]
      1. transform points: sensor frame -> map/world frame
         (currently: FIXED extrinsic from the URDF's mid360_joint mount
         pose, ROTATED by the robot's live YAW-ONLY heading if `--imu-topic`
         data is available, translated by an integrated position estimate
         if `--pose-topic` data is available -- see "Known limitation"
         below if you have neither).
      2. ElevationGridMap.update(): persistent, Kalman-fused 2.5D grid in the
         map frame (this is the from-scratch stand-in for `elevation_mapping`
         the C++/ROS package VideoMimic actually uses).
      3. Resample: build the SAME 11x11, 0.1m-resolution, root-centered,
         heading-rotated egocentric query grid your sim policy trains on
         (`height_map_flat` in `gear_sonic.envs.manager_env.mdp.observations`
         / `commands.py`'s `_update_command`), by nearest-cell lookup into
         the fused map (falling back to a configurable default height for
         any cell that's still NaN/never-observed).
      4. Republish that 11x11 grid over ZMQ using the SAME wire format
         `height_map.py` already produces (so `view_lidar_client.py
         --view-heightmap` visualizes it, and it's a drop-in stand-in for
         the ROS-based `height_map.py` bridge wherever you don't have/want
         a full `elevation_mapping_humanoid` ROS install).

Known limitation (same caveat as every other real-robot script in this
project so far, see `g1_depth_subscriber_real_emc.py`'s docstring): without
a real pose source, the map only rotates with the robot's heading but does
NOT translate as the robot walks -- i.e. it behaves as if the robot were
spinning in place at a fixed spot, not really building a map that extends
beyond the current single scan's footprint. Wire a real translation
estimate (e.g. integrate `LowState_`'s IMU + leg odometry, or a proper
LiDAR-inertial odometry front end like FAST-LIO, which is what VideoMimic's
verified real pipeline actually uses -- see `dev_notes/
heightmap_architecture_analysis.md`) into `--pose-topic`/the `get_pose()`
stub below before trusting this for anything beyond a bench/stationary test.

Usage:
    .venv_sim/bin/python perception_heightmap/g1_lidar_subscriber_elevation.py \\
        --host 192.168.123.164 --port 5558 --zmq-out-port 5559 --visualize
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elevation_grid_map import ElevationGridMap  # noqa: E402
from view_lidar_client import HEADER_SIZE, parse_message  # noqa: E402

# --- mid360_link mount pose, from gear_sonic/data/assets/robot_description/
# urdf/g1/main.urdf: <origin xyz="0.0002835 0.00003 0.41618" rpy="0 3.101 3.1415" />
# relative to torso_link, which itself sits ~0.044m above the pelvis via the
# waist_roll joint (waist_yaw/waist_pitch have zero origin offset). Net: the
# sensor is ~0.46m above the pelvis, centered in X/Y.
MID360_HEIGHT_ABOVE_PELVIS_M = 0.044 + 0.41618  # ~0.460m
# rpy pitch/yaw both near pi is the "flip Z to point up" convention for a
# dome-mounted 360-degree LiDAR -- net effect, the sensor's own +Z points
# along the robot's actual up direction. We do NOT reproduce the full
# rotation here (unnecessary once the driver already reports points in a
# sensible sensor-local ENU-ish frame); if your raw points come out
# upside-down/mirrored, that's the first place to look.


def _jet_colormap(t: np.ndarray) -> np.ndarray:
    """Standard "jet" colormap (dark blue -> cyan -> green -> yellow -> red
    -> dark red), same palette convention used for the sim's height-map
    debug visualizer (see `_jet_colormap` in `gear_sonic.envs.manager_env.
    mdp.commands`), so this 3D view reads the same way. `t` in [0, 1],
    any shape; returns an (..., 3) RGB array in [0, 1].
    """
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def build_elevation_mesh(emap: ElevationGridMap, stride: int = 1,
                          vmin=None, vmax=None):
    """Build an Open3D `TriangleMesh` "terrain surface" from the accumulated
    `ElevationGridMap`, colored by height with a jet colormap -- the 3D-look
    equivalent of `heightmap_to_image`'s flat top-down 2D view (e.g. the
    reference "Elevation Map" screenshots from `elevation_mapping`/
    `elevation_mapping_cupy`'s own RViz visualizers).

    Cells that are still NaN (never observed) are excluded entirely: any
    triangle touching a NaN vertex is dropped, so the mesh only covers the
    region actually seen so far (matching the "dashed red boundary" /
    ragged-edge look of the reference image, rather than interpolating over
    unknown area).

    stride: subsample the grid every `stride` cells before meshing (2 or 4
        speeds things up a lot for large/fine maps with negligible visual
        loss, since the underlying map resolution is usually much finer
        than needed for a live 3D view).
    vmin/vmax: fixed color-scale bounds (metres). If None, auto-computed
        from the currently-observed cells each call, which is convenient but
        means the color meaning shifts frame-to-frame -- pass fixed bounds
        (e.g. from your terrain's expected height range) for a stable scale.

    Returns (mesh, vmin, vmax) -- an `open3d.geometry.TriangleMesh` (or None
    if nothing has been observed yet / open3d isn't installed) plus the
    color-scale bounds actually used (handy to reuse next call for a stable
    scale without recomputing from the first frame).
    """
    import open3d as o3d

    elevation = emap.elevation[::stride, ::stride]
    xs = emap._xs[::stride]
    ys = emap._ys[::stride]
    n_rows, n_cols = elevation.shape

    valid = ~np.isnan(elevation)
    if not valid.any():
        return None, vmin, vmax

    if vmin is None:
        vmin = float(np.nanmin(elevation))
    if vmax is None:
        vmax = float(np.nanmax(elevation))
    vrange = max(vmax - vmin, 1e-6)

    grid_x, grid_y = np.meshgrid(xs, ys)  # both (n_rows, n_cols)
    z = np.where(valid, elevation, 0.0)
    vertices = np.stack([grid_x, grid_y, z], axis=-1).reshape(-1, 3)

    t = (elevation - vmin) / vrange
    colors = _jet_colormap(t)
    colors = np.where(valid[..., None], colors, 0.35)  # gray-ish for masked verts
    colors = colors.reshape(-1, 3)

    # Two triangles per cell quad; only keep triangles where ALL 3 corner
    # vertices are actually observed (drops the "never seen" border cleanly
    # instead of interpolating a false surface over it).
    idx = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    v00 = idx[:-1, :-1].ravel()
    v01 = idx[:-1, 1:].ravel()
    v10 = idx[1:, :-1].ravel()
    v11 = idx[1:, 1:].ravel()
    valid_flat = valid.ravel()

    tri_a = np.stack([v00, v10, v01], axis=1)
    tri_b = np.stack([v10, v11, v01], axis=1)
    keep_a = valid_flat[v00] & valid_flat[v10] & valid_flat[v01]
    keep_b = valid_flat[v10] & valid_flat[v11] & valid_flat[v01]
    triangles = np.concatenate([tri_a[keep_a], tri_b[keep_b]], axis=0)

    if triangles.shape[0] == 0:
        return None, vmin, vmax

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh, vmin, vmax


def pack_message(topic: str, fields: dict) -> bytes:
    """Matches `height_map.py`'s `pack_message` / `view_lidar_client.py`'s
    `parse_message` wire format exactly."""
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


def build_query_grid(grid_size: int, resolution: float, heading_rad: float,
                      center_xy: np.ndarray) -> np.ndarray:
    """The SAME egocentric query grid `commands.py`'s `_update_command` uses
    for `height_map_flat`: local (x, y) offsets rotated by heading-only yaw,
    translated to the robot's current map-frame XY. Returns (grid_size**2, 2)
    world/map-frame query points, in the same row-major flatten order as sim.
    """
    half = grid_size / 2.0 * resolution
    lin = -half + (np.arange(grid_size) + 0.5) * resolution
    grid_x, grid_y = np.meshgrid(lin, lin)
    local = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (N, 2)

    c, s = np.cos(heading_rad), np.sin(heading_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + center_xy[None, :]


def sample_grid_map(emap: ElevationGridMap, query_xy: np.ndarray,
                     default_height: float) -> np.ndarray:
    """Nearest-cell lookup of `emap.elevation` at each query point (map
    frame), falling back to `default_height` for any query point outside the
    map bounds or landing on a still-NaN (never-observed) cell -- matching
    the fallback semantics VideoMimic's `KNN_DEFAULT_HEIGHT_OFFSET` serves in
    `interpolateHeightIDW`, just via nearest-cell instead of IDW-over-kNN
    (this map's cells are already much finer, 0.05m by default, than the
    0.1m query grid, so nearest-cell is a reasonable approximation)."""
    half_x = emap.size_x / 2.0 * emap.resolution
    half_y = emap.size_y / 2.0 * emap.resolution
    rel_x = query_xy[:, 0] - (emap.position[0] - half_x)
    rel_y = query_xy[:, 1] - (emap.position[1] - half_y)

    i = np.round(rel_x / emap.resolution - 0.5).astype(np.int64)
    j = np.round(rel_y / emap.resolution - 0.5).astype(np.int64)
    in_bounds = (i >= 0) & (i < emap.size_x) & (j >= 0) & (j < emap.size_y)

    heights = np.full(query_xy.shape[0], default_height, dtype=np.float32)
    ii, jj = i[in_bounds], j[in_bounds]
    vals = emap.elevation[jj, ii]
    valid = ~np.isnan(vals)
    idx = np.where(in_bounds)[0]
    heights[idx[valid]] = vals[valid].astype(np.float32)
    return heights


def full_map_to_heightmap_dict(emap: ElevationGridMap, empty_sentinel: float = -1e6) -> dict:
    """Package the FULL accumulated `ElevationGridMap` (not the 11x11
    egocentric query grid) into the same {"grid", "grid_size", "half_extent",
    "cell_size", "empty_sentinel"} shape `heightmap_to_image` /
    `print_heightmap` expect, so the whole persistent map can be visualized
    directly -- NaN ("never observed") cells become `empty_sentinel`, same
    convention as everywhere else in this project.

    Unlike the 11x11 query grid (egocentric, moves/rotates with the robot),
    this is the MAP-FRAME-FIXED grid itself: what has been accumulated so
    far, in its own fixed frame, regardless of where the robot currently is
    or is facing.
    """
    grid = np.where(np.isnan(emap.elevation), empty_sentinel, emap.elevation).astype(np.float32)
    return {
        "grid": grid,
        "grid_size": np.array([emap.size_x], dtype=np.int32),  # assumes size_x == size_y
        "half_extent": np.array([emap.size_x / 2.0 * emap.resolution], dtype=np.float32),
        "cell_size": np.array([emap.resolution], dtype=np.float32),
        "empty_sentinel": np.array([empty_sentinel], dtype=np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="robot IP running g1_lidar_publisher.py")
    ap.add_argument("--port", type=int, default=5558, help="ZMQ port of g1_lidar_publisher.py")
    ap.add_argument("--in-topic", default="lidar_cloud")
    ap.add_argument("--zmq-out-port", type=int, default=5559,
                     help="port to republish the resampled heightmap on")
    ap.add_argument("--out-topic", default="lidar_heightmap")
    ap.add_argument("--full-map-out-topic", default="lidar_elevation_map",
                     help="ZMQ topic the FULL accumulated map is republished on "
                          "(separate from --out-topic's egocentric 11x11 grid)")

    # ElevationGridMap params (see elevation_grid_map.py's docstring).
    ap.add_argument("--map-resolution", type=float, default=0.05)
    ap.add_argument("--map-length", type=float, default=4.0,
                     help="fixed map extent in metres (must be large enough to always "
                          "cover wherever the egocentric query grid ends up)")

    # Egocentric query-grid params -- MUST match commands.py's use_height_map
    # cfg (height_map_size=1.0, height_map_resolution=0.1 -> 11x11) for the
    # output to be a valid drop-in for a deployed ONNX policy's height_map_flat.
    ap.add_argument("--grid-size", type=int, default=11)
    ap.add_argument("--grid-resolution", type=float, default=0.1)
    ap.add_argument("--default-height-offset", type=float, default=0.0,
                     help="fallback height (map-frame Z) for never-observed cells")

    ap.add_argument("--heading-rad", type=float, default=0.0,
                     help="STATIC placeholder robot heading (yaw, radians) used to "
                          "rotate both the incoming points and the query grid. "
                          "Replace with a live IMU/odometry feed for real use -- see "
                          "the module docstring's 'Known limitation' section.")
    ap.add_argument("--visualize", action="store_true",
                     help="show the 11x11 egocentric query grid (what the policy sees)")
    ap.add_argument("--visualize-full-map", action="store_true",
                     help="show the FULL accumulated, map-frame-fixed elevation map "
                          "(what has been observed so far, in its own fixed frame)")
    ap.add_argument("--full-map-scale", type=int, default=4,
                     help="pixel scale for the full-map window (it has many more cells "
                          "than the 11x11 query grid, so a smaller per-cell scale usually "
                          "looks better -- see heightmap_to_image's `scale` arg)")
    ap.add_argument("--visualize-3d", action="store_true",
                     help="show a live Open3D 3D terrain-surface view of the accumulated "
                          "elevation map (jet-colored mesh, ragged edge at the observed "
                          "boundary -- similar look to elevation_mapping/elevation_mapping_"
                          "cupy's own RViz visualizer). Requires `pip install open3d`.")
    ap.add_argument("--mesh-stride", type=int, default=1,
                     help="subsample the map every N cells before building the 3D mesh "
                          "(speeds up --visualize-3d for large/fine maps)")
    ap.add_argument("--mesh-vmin", type=float, default=None,
                     help="fixed lower bound (metres) for the 3D mesh's color scale; "
                          "default auto-computes from the current frame each update")
    ap.add_argument("--mesh-vmax", type=float, default=None,
                     help="fixed upper bound (metres) for the 3D mesh's color scale")
    ap.add_argument("--stats-every", type=float, default=5.0)
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{args.port}")
    sub.setsockopt(zmq.SUBSCRIBE, args.in_topic.encode("utf-8"))
    sub.setsockopt(zmq.RCVHWM, 2)
    print(f"[lidar_elev] connected to tcp://{args.host}:{args.port}, "
          f"topic={args.in_topic!r}")

    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://0.0.0.0:{args.zmq_out_port}")
    print(f"[lidar_elev] republishing heightmap on tcp://0.0.0.0:{args.zmq_out_port} "
          f"topic={args.out_topic!r}")

    emap = ElevationGridMap(
        resolution=args.map_resolution,
        length_x=args.map_length,
        length_y=args.map_length,
    )

    # STATIC robot position in the map frame (see "Known limitation" above --
    # without real translation odometry, the map is centered at the origin
    # for the whole run; only heading rotates).
    robot_xy = np.array([0.0, 0.0], dtype=np.float64)

    c, s = np.cos(args.heading_rad), np.sin(args.heading_rad)
    rot_world_from_sensor = np.array([[c, -s], [s, c]], dtype=np.float64)

    # --- Open3D 3D-mesh viewer setup (see build_elevation_mesh) ---
    vis3d = None
    mesh3d = None
    axes3d = None
    mesh_vmin, mesh_vmax = args.mesh_vmin, args.mesh_vmax
    first_mesh = True
    if args.visualize_3d:
        try:
            import open3d as o3d
            vis3d = o3d.visualization.Visualizer()
            vis3d.create_window("lidar elevation map (3D terrain surface)", width=1024, height=768)
            opt = vis3d.get_render_option()
            opt.background_color = np.asarray([0.05, 0.05, 0.05])
            opt.mesh_show_back_face = True
            axes3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            vis3d.add_geometry(axes3d)  # marks the robot's current position/frame
        except ImportError:
            print("[lidar_elev] --visualize-3d requires open3d: pip install open3d",
                  file=sys.stderr)
            args.visualize_3d = False

    n_msgs = 0
    t_stat = time.time()

    try:
        while True:
            raw = sub.recv()
            data, err = parse_message(raw, args.in_topic)
            if err is not None or data is None:
                if err is not None:
                    print(f"[lidar_elev] parse error: {err}", file=sys.stderr)
                continue

            points = data.get("points")
            if points is None or points.shape[0] == 0:
                continue

            # Sensor frame -> map frame: yaw-rotate (heading placeholder),
            # then add sensor's known static height offset above the pelvis
            # (see MID360_HEIGHT_ABOVE_PELVIS_M) and the robot's XY position.
            xy = points[:, :2].astype(np.float64) @ rot_world_from_sensor.T
            xy += robot_xy[None, :]
            z = points[:, 2].astype(np.float64)
            points_map = np.concatenate([xy, z[:, None]], axis=1)

            emap.update(points_map, stamp=time.time())

            query_xy = build_query_grid(
                args.grid_size, args.grid_resolution, args.heading_rad, robot_xy)
            heights_flat = sample_grid_map(
                emap, query_xy, default_height=args.default_height_offset)
            grid = heights_flat.reshape(args.grid_size, args.grid_size)

            packed = pack_message(args.out_topic, {
                "grid": grid.astype(np.float32),
                "grid_size": np.array([args.grid_size], dtype=np.int32),
                "cell_size": np.array([args.grid_resolution], dtype=np.float32),
                "half_extent": np.array(
                    [args.grid_size / 2.0 * args.grid_resolution], dtype=np.float32),
                "empty_sentinel": np.array([-1e6], dtype=np.float32),
            })
            pub.send(packed)

            # Republish the FULL accumulated map too (separate topic), so it
            # can be viewed live with `view_lidar_client.py --view-heightmap
            # --port <zmq-out-port> ` (pass a distinct --topic override there
            # if you want both at once) or captured for later analysis.
            full_map_dict = full_map_to_heightmap_dict(emap)
            pub.send(pack_message(args.full_map_out_topic, full_map_dict))

            n_msgs += 1
            now = time.time()
            if now - t_stat > args.stats_every:
                n_obs_cells = int((~np.isnan(emap.elevation)).sum())
                print(f"[lidar_elev] {n_msgs / (now - t_stat):.1f} msg/s, "
                      f"map cells observed={n_obs_cells}/{emap.elevation.size}, "
                      f"last cloud n_points={points.shape[0]}", flush=True)
                n_msgs = 0
                t_stat = now

            if args.visualize or args.visualize_full_map:
                try:
                    from view_lidar_client import heightmap_to_image
                    import cv2

                    if args.visualize:
                        img = heightmap_to_image({
                            "grid": grid, "grid_size": np.array([args.grid_size]),
                            "cell_size": np.array([args.grid_resolution]),
                            "empty_sentinel": np.array([-1e6]),
                        })
                        cv2.imshow("lidar elevation (egocentric 11x11 query grid)", img)

                    if args.visualize_full_map:
                        full_img = heightmap_to_image(
                            full_map_dict, scale=args.full_map_scale)
                        cv2.imshow(
                            "lidar elevation (FULL accumulated map, fixed frame)", full_img)

                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                except ImportError:
                    print("[lidar_elev] --visualize/--visualize-full-map require "
                          "opencv-python; disabling.", file=sys.stderr)
                    args.visualize = False
                    args.visualize_full_map = False

            if args.visualize_3d and vis3d is not None:
                new_mesh, mesh_vmin, mesh_vmax = build_elevation_mesh(
                    emap, stride=args.mesh_stride, vmin=mesh_vmin, vmax=mesh_vmax)
                if new_mesh is not None:
                    if mesh3d is None:
                        mesh3d = new_mesh
                        vis3d.add_geometry(mesh3d)
                    else:
                        mesh3d.vertices = new_mesh.vertices
                        mesh3d.triangles = new_mesh.triangles
                        mesh3d.vertex_colors = new_mesh.vertex_colors
                        mesh3d.vertex_normals = new_mesh.vertex_normals
                        vis3d.update_geometry(mesh3d)
                    if first_mesh:
                        vis3d.reset_view_point(True)
                        first_mesh = False
                vis3d.poll_events()
                vis3d.update_renderer()

    except KeyboardInterrupt:
        print("[lidar_elev] stopping.", file=sys.stderr)


if __name__ == "__main__":
    main()
