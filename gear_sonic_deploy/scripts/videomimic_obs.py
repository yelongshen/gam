#!/usr/bin/env python3
"""videomimic_obs.py
===================
Build VideoMimic's `terrain_height_noisy` policy input from G1 sensor points.

Contract (from hongsukchoi/VideoMimic, verified against both the training
sensor and the real-robot deployment):
  * simulation: HeightfieldCfg(name="terrain_height_noisy", body_name="torso_link",
    size=(1.0, 1.0), resolution=0.1, only_heading=True) -> grid_pattern() with
    ordering="xy" -> depth_map.view(grid_height, grid_width), i.e. obs[y][x];
    value = downward ray distance from the torso = torso_z - terrain_z.
    Training noise: 2 cm white + 2 cm per-episode offset, 0.04 rad roll/pitch,
    0.08 rad yaw, up to 3 steps delay.
  * deployment (sim2real/videomimic_real/videomimic_inference_real.cpp):
    11x11 query points at -0.5..+0.5 m (0.1 m) in the torso's heading frame,
    each value = torso_z - IDW(z of 3 nearest map points in xy), falling back
    to 0.85 when the nearest point is > 0.15 m away; heights_(j=y, i=x), and
    the Eigen column-major -> torch transpose yields obs[y][x] again.

Inputs here are points in a gravity-aligned frame (e.g. g1_frames.lidar_to_level
/ depth_to_level output, or elevation-map cells in the map frame) plus the
torso position and yaw in that same frame. For the levelled torso frame the
torso is at the origin with yaw 0, which is the default.

Gantry filter: while the robot hangs on the gantry, its uprights sit inside
or next to the 1 m window and read as walls ~0.3 m above the torso, which the
policy never saw in training. The robot yaws within the gantry, so the live
mode's --gantry uses g1_frames.near_tall_mask (tall points near the torso);
`remove_boxes()` / GANTRY_BOXES (fixed boxes measured once) remain for offline use.

Usage (live, needs g1_lidar_publisher.py and g1_depth_publisher.py on the robot):
    python3 videomimic_obs.py --host 192.168.8.227 --gantry
"""

import argparse
import json
import time

import numpy as np

import g1_frames

try:
    from scipy.spatial import cKDTree
except ImportError:  # the IDW path needs scipy; the robot has scipy 1.3
    cKDTree = None

GRID_N = 11
GRID_RES = 0.1
DEFAULT_HEIGHT = 0.85   # KNN_DEFAULT_HEIGHT_OFFSET in the deployment code
KNN_K = 3
KNN_MAX_DISTANCE = 0.15

# Levelled torso frame, z relative to the floor. Measured on the gantry
# 2026-09-28 from two captures (upright behind-left, bar behind-right),
# padded by ~5 cm.
GANTRY_BOXES = [
    # (xmin, xmax, ymin, ymax, zmin, zmax)
    (-0.55, -0.15, 0.43, 0.75, 0.50, 1.50),
    (-1.00, -0.55, -1.05, -0.70, 0.60, 0.80),
]


def query_points(torso_xy=(0.0, 0.0), torso_yaw=0.0):
    """(GRID_N*GRID_N, 2) xy query points, row-major over [y][x] like the policy."""
    c = (np.arange(GRID_N) - (GRID_N - 1) / 2.0) * GRID_RES  # -0.5 .. +0.5
    gx, gy = np.meshgrid(c, c, indexing="xy")                # gx[y][x] = c[x]
    local = np.stack([gx.ravel(), gy.ravel()], axis=1)
    cy, sy = np.cos(torso_yaw), np.sin(torso_yaw)
    rot = np.array([[cy, -sy], [sy, cy]])
    return local @ rot.T + np.asarray(torso_xy, dtype=np.float64)


def terrain_obs(points, torso_pos=(0.0, 0.0, 0.0), torso_yaw=0.0,
                default=DEFAULT_HEIGHT, return_mask=False, tree=None):
    """VideoMimic terrain_height observation, shape (11, 11) as obs[y][x].

    points: (N, 3) terrain points in a gravity-aligned frame.
    torso_pos, torso_yaw: torso position / heading in that frame.
    Cells whose nearest point is > 0.15 m away get `default` (as deployed).
    tree: optional prebuilt cKDTree over points[:, :2] (points must then be all-finite), to reuse
    one tree across many calls on the same map (e.g. sampling at 50 Hz from a 5 Hz map).
    """
    if cKDTree is None:
        raise ImportError("terrain_obs needs scipy (cKDTree)")
    torso_pos = np.asarray(torso_pos, dtype=np.float64)
    q = query_points(torso_pos[:2], torso_yaw)
    obs = np.full(len(q), float(default))
    seen = np.zeros(len(q), dtype=bool)
    points = np.asarray(points, dtype=np.float64)
    if tree is None:
        points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) >= KNN_K:
        dist, idx = (tree if tree is not None else cKDTree(points[:, :2])).query(q, k=KNN_K)
        seen = dist[:, 0] <= KNN_MAX_DISTANCE
        w = 1.0 / (dist + 1e-6)
        z = (points[idx, 2] * w).sum(1) / w.sum(1)
        obs[seen] = torso_pos[2] - z[seen]
    obs = obs.reshape(GRID_N, GRID_N)
    return (obs, seen.reshape(GRID_N, GRID_N)) if return_mask else obs


def remove_boxes(points_level, boxes=GANTRY_BOXES, floor_z=None):
    """Drop points inside any (xmin,xmax,ymin,ymax,zmin,zmax) box.

    points_level: levelled torso frame. Box z is relative to the floor; pass
    floor_z (torso-frame z of the floor, negative) or it is estimated."""
    p = np.asarray(points_level)
    if floor_z is None:
        floor_z = g1_frames.estimate_floor_z(p)
    z = p[:, 2] - floor_z
    keep = np.ones(len(p), dtype=bool)
    for x0, x1, y0, y1, z0, z1 in boxes:
        keep &= ~((p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1) & (z >= z0) & (z <= z1))
    return p[keep]


def gridmap_to_points(elevation, resolution, center_xy=(0.0, 0.0)):
    """elevation_mapping_cupy / grid_map layer -> (N, 3) points in the map frame.
    grid_map convention: row index runs along -x, column index along -y."""
    n_rows, n_cols = elevation.shape
    xs = center_xy[0] + (n_rows / 2.0 - (np.arange(n_rows) + 0.5)) * resolution
    ys = center_xy[1] + (n_cols / 2.0 - (np.arange(n_cols) + 0.5)) * resolution
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ok = np.isfinite(elevation)
    return np.stack([X[ok], Y[ok], elevation[ok]], axis=1)


def format_obs(obs, seen=None):
    """Printable grid: +y at the top, +x (forward) to the right; '*' = default-filled."""
    lines = []
    for yi in range(GRID_N - 1, -1, -1):
        cells = []
        for xi in range(GRID_N):
            s = f"{obs[yi, xi]:5.2f}"
            cells.append(s + ("*" if seen is not None and not seen[yi, xi] else " "))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def _live(args):
    import struct
    import zmq
    try:
        import lz4.frame as lz4frame
    except ImportError:
        lz4frame = None

    ctx = zmq.Context.instance()
    lid = ctx.socket(zmq.SUB)
    lid.connect(f"tcp://{args.host}:{args.lidar_port}")
    lid.setsockopt(zmq.SUBSCRIBE, b"")
    dep = ctx.socket(zmq.SUB)
    dep.setsockopt(zmq.RCVHWM, 2)
    dep.connect(f"tcp://{args.host}:{args.depth_port}")
    dep.setsockopt(zmq.SUBSCRIBE, b"depth")    # depth image (works with the publisher's --no-points)
    poller = zmq.Poller()
    poller.register(lid, zmq.POLLIN)
    poller.register(dep, zmq.POLLIN)

    def lidar_fields(raw, topic):
        hdr = json.loads(raw[len(topic):len(topic) + 1280].rstrip(b"\x00"))
        pos, out = len(topic) + 1280, {}
        dts = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64, "bool": np.bool_}
        for f in hdr["fields"]:
            dt = np.dtype(dts[f["dtype"]])
            n = int(np.prod(f["shape"])) * dt.itemsize
            out[f["name"]] = np.frombuffer(raw[pos:pos + n], dtype=dt).reshape(f["shape"])
            pos += n
        return out

    accel, lidar_pts, depth_pts, t_print = None, None, None, 0.0
    while True:
        for s, _ in poller.poll(200):
            if s is lid:
                raw = lid.recv()
                if raw.startswith(b"lidar_imu"):
                    a = lidar_fields(raw, b"lidar_imu")["linear_acceleration"].astype(np.float64)
                    accel = a if accel is None else 0.95 * accel + 0.05 * a
                elif raw.startswith(b"lidar_cloud") and accel is not None:
                    lidar_pts = g1_frames.lidar_to_level(lidar_fields(raw, b"lidar_cloud")["points"], accel)
            else:
                _, part = dep.recv_multipart()
                n = struct.unpack("<I", part[:4])[0]
                hdr = json.loads(part[4:4 + n])
                body = part[4 + n:]
                if hdr["compress"] == "lz4":
                    body = lz4frame.decompress(body)
                if accel is not None:
                    z = np.frombuffer(body, dtype=hdr["dtype"]).reshape(hdr["shape"]).astype(np.float32) * hdr["depth_scale"]
                    k = hdr["intrinsics"]
                    v, u = np.mgrid[0:z.shape[0], 0:z.shape[1]].astype(np.float32)
                    vtx = np.stack([(u - k["ppx"]) / k["fx"] * z, (v - k["ppy"]) / k["fy"] * z, z], axis=-1)
                    depth_pts = g1_frames.depth_to_level(vtx, accel)
        if time.time() - t_print < args.period or lidar_pts is None:
            continue
        t_print = time.time()
        pts = lidar_pts if depth_pts is None else np.vstack([lidar_pts, depth_pts])
        floor_z = g1_frames.estimate_floor_z(lidar_pts)
        if args.gantry:
            pts = pts[~g1_frames.near_tall_mask(pts, floor_z)]
        obs, seen = terrain_obs(pts, return_mask=True)
        print(f"[videomimic] torso {-floor_z:.3f} m above floor; seen {seen.mean()*100:.0f}% "
              f"(ideal flat floor = {-floor_z:.2f}; '*' = filled with {DEFAULT_HEIGHT})")
        print(format_obs(obs, seen), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.8.227")
    ap.add_argument("--lidar-port", type=int, default=5558)
    ap.add_argument("--depth-port", type=int, default=5557)
    ap.add_argument("--gantry", action="store_true",
                    help="drop points > 0.45 m above the floor within 0.8 m of the torso (g1_frames.near_tall_mask)")
    ap.add_argument("--period", type=float, default=1.0, help="seconds between printouts")
    _live(ap.parse_args())
