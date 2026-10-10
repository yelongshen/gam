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
    python3 perception_heightmap/view_lidar_client.py --host 192.168.123.164

    # Live Open3D point-cloud viewer (updates each received frame):
    python3 perception_heightmap/view_lidar_client.py --host 192.168.123.164 --view

    # Live OpenCV heightmap viewer (colorized 2D top-down grid):
    python3 perception_heightmap/view_lidar_client.py --host 192.168.123.164 --view-heightmap
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


class LidarRecorder:
    """Accumulates raw `lidar_cloud` frames + `lidar_imu` samples in memory and
    writes them to a single compressed `.npz` on close (including on Ctrl+C).

    Point clouds are *ragged* -- the Mid-360's non-repetitive scan pattern
    gives a different point count every frame -- so they are stored the
    standard way for variable-length sequences: all points concatenated into
    one `(total_points, 3)` array plus a `cloud_offsets` index array, rather
    than an object/pickled array (which `np.savez` can only store with
    `allow_pickle`, making the file unsafe/awkward to load elsewhere).
    Use `load_recording()` below, or slice manually:

        d = np.load("rec.npz")
        frame_i = d["cloud_points"][d["cloud_offsets"][i]:d["cloud_offsets"][i+1]]

    Memory note: ~20k points/frame at 10 Hz is ~2.4 MB/s of float32, so a
    few minutes is fine but an hour is not -- bound it with `--seconds`.
    """

    def __init__(self, path: str, record_filtered: bool = False):
        self.path = path
        self.record_filtered = record_filtered
        self._cloud_chunks = []      # list of (N_i, 3) float32
        self._cloud_counts = []      # N_i per frame
        self._cloud_recv_time = []   # host wall-clock at receive
        self._imu_orientation = []
        self._imu_angular_velocity = []
        self._imu_linear_acceleration = []
        self._imu_stamp = []         # sensor stamp, seconds (sec + nanosec/1e9)
        self._imu_recv_time = []

    def add_cloud(self, data: dict, recv_time: float):
        pts = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
        self._cloud_chunks.append(pts.copy())
        self._cloud_counts.append(pts.shape[0])
        self._cloud_recv_time.append(recv_time)

    def add_imu(self, data: dict, recv_time: float):
        self._imu_orientation.append(np.asarray(data["orientation"], dtype=np.float64).reshape(4))
        self._imu_angular_velocity.append(
            np.asarray(data["angular_velocity"], dtype=np.float64).reshape(3))
        self._imu_linear_acceleration.append(
            np.asarray(data["linear_acceleration"], dtype=np.float64).reshape(3))
        if "stamp_sec" in data and "stamp_nanosec" in data:
            stamp = float(data["stamp_sec"][0]) + float(data["stamp_nanosec"][0]) * 1e-9
        else:
            stamp = float("nan")
        self._imu_stamp.append(stamp)
        self._imu_recv_time.append(recv_time)

    @property
    def n_clouds(self) -> int:
        return len(self._cloud_counts)

    @property
    def n_imu(self) -> int:
        return len(self._imu_stamp)

    def save(self):
        if self.n_clouds == 0 and self.n_imu == 0:
            print(f"[record] nothing captured -- not writing {self.path}")
            return

        if self._cloud_chunks:
            cloud_points = np.concatenate(self._cloud_chunks, axis=0)
        else:
            cloud_points = np.zeros((0, 3), dtype=np.float32)
        # offsets[i]:offsets[i+1] slices frame i out of cloud_points
        cloud_offsets = np.zeros(len(self._cloud_counts) + 1, dtype=np.int64)
        if self._cloud_counts:
            cloud_offsets[1:] = np.cumsum(self._cloud_counts)

        def _stack(rows, width):
            return (np.stack(rows, axis=0) if rows
                    else np.zeros((0, width), dtype=np.float64))

        meta = {
            "cloud_frame": "lidar/mid360 sensor frame, as published (no transform applied)",
            "cloud_filtered": bool(self.record_filtered),
            "imu_orientation_convention": "xyzw",
            "n_clouds": self.n_clouds,
            "n_imu": self.n_imu,
        }

        np.savez_compressed(
            self.path,
            cloud_points=cloud_points,
            cloud_offsets=cloud_offsets,
            cloud_recv_time=np.asarray(self._cloud_recv_time, dtype=np.float64),
            imu_orientation=_stack(self._imu_orientation, 4),
            imu_angular_velocity=_stack(self._imu_angular_velocity, 3),
            imu_linear_acceleration=_stack(self._imu_linear_acceleration, 3),
            imu_stamp=np.asarray(self._imu_stamp, dtype=np.float64),
            imu_recv_time=np.asarray(self._imu_recv_time, dtype=np.float64),
            meta=json.dumps(meta),
        )
        print(f"[record] wrote {self.path}: {self.n_clouds} cloud frames "
              f"({cloud_points.shape[0]} points total), {self.n_imu} IMU samples"
              f"{' [range-filtered]' if self.record_filtered else ' [raw]'}")


def load_recording(path: str):
    """Load a `--record` .npz back into (clouds, imu) convenience structures.

    Returns:
        clouds: list of (N_i, 3) float32 arrays, one per frame
        imu: dict of stacked IMU arrays (orientation/angular_velocity/
             linear_acceleration/stamp/recv_time)
    """
    d = np.load(path)
    offsets = d["cloud_offsets"]
    pts = d["cloud_points"]
    clouds = [pts[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    imu = {
        "orientation": d["imu_orientation"],
        "angular_velocity": d["imu_angular_velocity"],
        "linear_acceleration": d["imu_linear_acceleration"],
        "stamp": d["imu_stamp"],
        "recv_time": d["imu_recv_time"],
    }
    return clouds, imu


def filter_points_by_range(points: np.ndarray, max_range: float, planar: bool = True) -> np.ndarray:
    """Keep only points within `max_range` metres of the origin (the sensor).

    planar=True measures distance in the XY plane only (a cylinder around the
    sensor), which is usually what you want for "surrounding Nm" ground-level
    filtering; planar=False uses full 3D Euclidean distance (a sphere).
    """
    if points.shape[0] == 0 or max_range <= 0:
        return points
    if planar:
        dist = np.linalg.norm(points[:, :2], axis=1)
    else:
        dist = np.linalg.norm(points, axis=1)
    return points[dist <= max_range]


def points_to_heightmap(
    points_xyz: np.ndarray,
    grid_size: int = 21,
    cell_size: float = 0.1,
    agg: str = "max",
    empty_sentinel: float = -1e6,
) -> dict:
    """Bin a raw LiDAR point cloud (already sensor/robot-centered, i.e. origin
    (0, 0) in XY is the sensor) into a local egocentric 2D height grid, in the
    same {"grid", "grid_size", "half_extent", "cell_size", "empty_sentinel"}
    shape produced by `parse_message` for the `lidar_heightmap` topic -- so it
    can be fed straight into `print_heightmap` / `heightmap_to_image`.

    This is a simple, dependency-free alternative to the kNN/IDW interpolation
    VideoMimic's on-robot elevation_mapping pipeline uses: each grid cell just
    takes the max (or mean) Z of whichever raw LiDAR points happen to land in
    it this frame. It's noisier frame-to-frame (no temporal fusion / denoising
    like `elevation_mapping_humanoid` does), but requires nothing beyond the
    raw `lidar_cloud` ZMQ messages you already receive here.

    grid_size: number of cells per side (e.g. 21 cells @ 0.1m = 2.1m extent,
        i.e. ~1m radius around the robot in every direction).
    cell_size: metres per cell.
    agg: "max" (highest point in each cell -> good for obstacle/step
        detection) or "mean" (average height -> smoother, less step-sensitive).
    empty_sentinel: fill value for cells with zero points landing in them.
    """
    half_extent = grid_size / 2.0 * cell_size
    grid = np.full((grid_size, grid_size), empty_sentinel, dtype=np.float32)

    if points_xyz.shape[0] == 0:
        return {
            "grid": grid,
            "grid_size": np.asarray([grid_size], dtype=np.float32),
            "half_extent": np.asarray([half_extent], dtype=np.float32),
            "cell_size": np.asarray([cell_size], dtype=np.float32),
            "empty_sentinel": np.asarray([empty_sentinel], dtype=np.float32),
        }

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    # Keep only points that actually fall within the grid's footprint.
    in_bounds = (np.abs(x) < half_extent) & (np.abs(y) < half_extent)
    x, y, z = x[in_bounds], y[in_bounds], z[in_bounds]

    if x.shape[0] == 0:
        return {
            "grid": grid,
            "grid_size": np.asarray([grid_size], dtype=np.float32),
            "half_extent": np.asarray([half_extent], dtype=np.float32),
            "cell_size": np.asarray([cell_size], dtype=np.float32),
            "empty_sentinel": np.asarray([empty_sentinel], dtype=np.float32),
        }

    # i = column (X index), j = row (Y index), matching print_heightmap's
    # "flip so +Y is at the top" convention.
    i = np.clip(((x + half_extent) / cell_size).astype(np.int64), 0, grid_size - 1)
    j = np.clip(((y + half_extent) / cell_size).astype(np.int64), 0, grid_size - 1)
    flat_idx = j * grid_size + i

    flat_grid = grid.reshape(-1)
    if agg == "mean":
        sums = np.zeros(grid_size * grid_size, dtype=np.float64)
        counts = np.zeros(grid_size * grid_size, dtype=np.int64)
        np.add.at(sums, flat_idx, z)
        np.add.at(counts, flat_idx, 1)
        has_data = counts > 0
        flat_grid[has_data] = (sums[has_data] / counts[has_data]).astype(np.float32)
    else:  # "max"
        # np.maximum.at handles repeated indices correctly (unlike plain
        # fancy-index assignment), starting from -inf so any real Z wins.
        heights = np.full(grid_size * grid_size, -np.inf, dtype=np.float64)
        np.maximum.at(heights, flat_idx, z)
        has_data = np.isfinite(heights)
        flat_grid[has_data] = heights[has_data].astype(np.float32)

    return {
        "grid": grid,
        "grid_size": np.asarray([grid_size], dtype=np.float32),
        "half_extent": np.asarray([half_extent], dtype=np.float32),
        "cell_size": np.asarray([cell_size], dtype=np.float32),
        "empty_sentinel": np.asarray([empty_sentinel], dtype=np.float32),
    }


class ElevationMapFuser:
    """A from-scratch, simplified re-implementation of what
    `elevation_mapping_humanoid` (https://github.com/ArthurAllshire/elevation_mapping_humanoid,
    a fork of smoggy-P/elevation_mapping_humanoid, itself based on ANYbotics'
    `elevation_mapping`) does to produce `/elevation_map_fused_visualization/
    elevation_cloud`, plus the kNN/IDW query step VideoMimic's
    `videomimic_inference_real.cpp` (`ElevationCloudCallback` /
    `interpolateHeightIDW`) does on top of it. Runs entirely client-side here,
    fed by the raw `lidar_cloud` ZMQ messages -- no publisher/ROS changes.

    CORRECTION vs. earlier assumptions in this file: in VideoMimic's shipped
    deployment (`sim2real/scripts/start_robot.sh`), the elevation cloud is fed
    by a **RealSense depth camera** via `roslaunch elevation_mapping_demos
    realsense_demo.launch` -- not LiDAR/Livox (the `livox_ros_driver_DIR` env
    var in that script is set but unused by that particular launch). LiDAR
    could feed the same `elevation_map_fused_visualization/elevation_cloud`
    topic in principle, but that's not what actually ships. This grid (21 or
    41 cells here vs. the real 11x11 @ 0.1m = 1.1m extent, matching
    `HeightfieldCfg(size=(1.0, 1.0), resolution=0.1)` in
    `g1_deepmimic_config.py`) is still fed by our own `lidar_cloud` topic here
    as a stand-in, since that's what our robot actually publishes.

    Also, the real C++ node's grid is placed in the **torso's body frame with
    yaw-only rotation** (pitch/roll ignored) via TF lookups (`odom_corrected`
    -> `torso_link`), and each cell's returned value is **`torso_z -
    ground_z`** (distance straight down from the torso), not an absolute world
    height -- which is why its fallback constant is 0.85 (nominal torso height
    above flat ground), not 0. This class has no TF/torso-pose input, so it
    reports raw sensor-frame Z instead; if you want the same "distance below
    torso" convention, subtract the torso height from `default_height` and
    from the fused grid downstream.

    Pipeline (matches the two real stages):

    1) Per-cell Kalman-filtered fusion over time ("elevation_mapping" stage):
       each grid cell keeps a running (height estimate, variance) pair. Every
       incoming LiDAR frame's points are binned into cells; each cell with new
       points this frame gets a scalar Kalman measurement update using the
       batch mean of those points (with measurement variance scaled down by
       how many points landed in the cell, since averaging N iid measurements
       reduces effective noise by 1/N). A small process-noise term is added to
       every cell's variance every frame (the "predict" step), so old
       estimates slowly relax if a cell stops getting hits (e.g. an obstacle
       moves away) instead of being frozen forever.

    2) kNN + inverse-distance-weighted interpolation ("inference" stage):
       exactly as `interpolateHeightIDW`/`ElevationCloudCallback` do (2D XY
       nearest-neighbor search, Z-blind -- see limitation note below), at
       query time we run a k-NN search (default k=3, matching `KNN_K`) over
       the fused cells that currently have a confident height estimate, and
       fill any grid cell (including still-empty ones) via inverse-distance
       weighting of those neighbors, rejecting a query point if its nearest
       neighbor is farther than `knn_max_distance` (matching `KNN_MAX_DISTANCE
       = 0.15`), falling back to `default_height` in that case (matching
       `KNN_DEFAULT_HEIGHT_OFFSET`).

    Two limitations carried over faithfully from the real system (per the
    VideoMimic C++ source):
      - **2D-only interpolation is Z-blind and can't represent overhangs.**
        A query column blends whichever points are nearest in XY regardless
        of height, so e.g. a chair seat and the floor beneath it collapse into
        one blurred value -- the same limitation the sim raycaster has, so
        sim/real are at least consistently unable to represent "sitting"
        geometry.
      - **The fallback height is an *optimistic* default, not "unknown".**
        Any cell farther than `knn_max_distance` from a real measurement is
        asserted to be flat ground (`default_height`/`KNN_DEFAULT_HEIGHT_
        OFFSET`), not flagged as unobserved. Occlusion therefore reads as
        "safe flat floor" -- a reasonable bias for walking on open ground, but
        a dangerous one right next to an obstacle whose far side is occluded.
      - Additionally, the real node **silently keeps the previous heights**
        (does nothing) if the elevation cloud is empty or a TF lookup fails
        that frame, rather than erroring -- so a policy consuming it can be
        acting on stale data without any explicit signal that fusion stalled.

    Important limitation vs. the real ROS node: `elevation_mapping_humanoid`
    keeps its map fixed in the world/odom frame and re-projects/shifts it each
    frame using the robot's live pose (TF), so cells stay geometrically
    correct as the robot walks around. This class has no odometry input (the
    `lidar_imu` topic here doesn't carry position, only orientation/IMU rates)
    so it assumes the sensor is effectively stationary (or only slowly
    drifting) between frames -- fine for "check the ground around me before
    stepping" while standing/slow-walking, but it will smear/corrupt the map
    if the robot moves quickly without a matching pose feed. Feed a
    `dx, dy` (metres, in the current sensor frame) into `shift(dx, dy)` before
    `update()` if you do have odometry, to (approximately, via nearest-cell
    re-indexing) keep the map robot-centered.
    """

    def __init__(
        self,
        grid_size: int = 21,
        cell_size: float = 0.1,
        process_noise: float = 2e-4,
        measurement_noise: float = 1e-3,
        init_variance: float = 1.0,
        confident_variance: float = 5e-2,
        knn_k: int = 3,
        knn_max_distance: float = 0.15,
        default_height: float = 0.0,
        empty_sentinel: float = -1e6,
        max_effective_count: int = 8,
        mode: str = "decay_max",
        decay: float = 0.9,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.confident_variance = confident_variance
        self.knn_k = knn_k
        self.knn_max_distance = knn_max_distance
        self.default_height = default_height
        self.empty_sentinel = empty_sentinel
        # Cap how much a single frame's point count can shrink the effective
        # measurement variance (meas_var = measurement_noise / min(count,
        # max_effective_count)). Without this cap, a LiDAR frame with dozens
        # of points per cell drives the Kalman gain to ~1 every frame (i.e.
        # near-total overwrite by the newest single frame -- no real temporal
        # smoothing at all, just very noisy raw per-frame data wearing a
        # "fused" label). Capping it keeps a genuine multi-frame blend, at the
        # cost of a few frames' worth of extra latency to fully adopt a
        # persistent change (e.g. someone standing still nearby).
        self.max_effective_count = max_effective_count
        # mode="kalman": the original continuous elevation-surface fusion
        # (per-cell height + variance, kNN/IDW query stage). Faithful to
        # elevation_mapping_humanoid/VideoMimic, but the resulting continuous
        # heights are genuinely hard to eyeball for "what's approaching" --
        # small, physically meaningless variations (grazing/edge points mixing
        # floor+obstacle returns in the same XY cell) are just as visually
        # loud as a real obstacle.
        # mode="decay_max" (default): a much more legible alternative for
        # obstacle/approach detection. Each cell just tracks a decaying max
        # height: every frame the stored height relaxes a bit toward
        # `default_height` (so stale detections fade out over ~10-20 frames
        # if not re-observed), then gets pulled back up to at least this
        # frame's max Z if anything new/taller was seen. An approaching
        # object's cells light up almost immediately (no slow Kalman
        # convergence) and fade out again once it leaves, rather than getting
        # smeared into a continuous "surface".
        self.mode = mode
        self.decay = decay
        self.half_extent = grid_size / 2.0 * cell_size

        self.height = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.variance = np.full((grid_size, grid_size), init_variance, dtype=np.float64)
        self.seen = np.zeros((grid_size, grid_size), dtype=bool)

        ys = -self.half_extent + (np.arange(grid_size) + 0.5) * cell_size
        xs = -self.half_extent + (np.arange(grid_size) + 0.5) * cell_size
        grid_x, grid_y = np.meshgrid(xs, ys)  # (grid_size, grid_size), j=row(Y), i=col(X)
        self._cell_centers_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    def reset(self):
        self.height.fill(0.0)
        self.variance.fill(self.variance[0, 0])  # keep init_variance
        self.seen[:] = False

    def shift(self, dx: float, dy: float):
        """Approximate the robot having moved (dx, dy) metres (in the current
        sensor/base frame) since the last update, by rolling the grid arrays
        by the nearest integer number of cells and marking the newly-exposed
        border as unseen again. This is a coarse stand-in for the real TF-based
        re-registration `elevation_mapping` does; sub-cell motion is ignored.
        """
        shift_cells_x = int(round(dx / self.cell_size))
        shift_cells_y = int(round(dy / self.cell_size))
        if shift_cells_x == 0 and shift_cells_y == 0:
            return
        self.height = np.roll(self.height, (-shift_cells_y, -shift_cells_x), axis=(0, 1))
        self.variance = np.roll(self.variance, (-shift_cells_y, -shift_cells_x), axis=(0, 1))
        self.seen = np.roll(self.seen, (-shift_cells_y, -shift_cells_x), axis=(0, 1))

        init_var = self.variance.max()  # cheap heuristic for "reset" variance
        if shift_cells_y > 0:
            self.seen[-shift_cells_y:, :] = False
            self.variance[-shift_cells_y:, :] = init_var
        elif shift_cells_y < 0:
            self.seen[:-shift_cells_y, :] = False
            self.variance[:-shift_cells_y, :] = init_var
        if shift_cells_x > 0:
            self.seen[:, -shift_cells_x:] = False
            self.variance[:, -shift_cells_x:] = init_var
        elif shift_cells_x < 0:
            self.seen[:, :-shift_cells_x] = False
            self.variance[:, :-shift_cells_x] = init_var

    def update(self, points_xyz: np.ndarray):
        """Fuse one new LiDAR frame's points into the persistent grid (the
        "elevation_mapping" stage)."""
        if self.mode == "decay_max":
            self._update_decay_max(points_xyz)
        else:
            self._update_kalman(points_xyz)

    def _update_decay_max(self, points_xyz: np.ndarray):
        # Relax every cell a bit back toward the default/floor height each
        # frame, so a stale detection fades out over ~10-20 frames if it's
        # not re-observed (decay=0.9 => ~90% gone after ~22 frames).
        self.height = self.default_height + (self.height - self.default_height) * self.decay

        if points_xyz.shape[0] == 0:
            return

        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        z = points_xyz[:, 2].astype(np.float64)

        in_bounds = (np.abs(x) < self.half_extent) & (np.abs(y) < self.half_extent)
        x, y, z = x[in_bounds], y[in_bounds], z[in_bounds]
        if x.shape[0] == 0:
            return

        gs = self.grid_size
        i = np.clip(((x + self.half_extent) / self.cell_size).astype(np.int64), 0, gs - 1)
        j = np.clip(((y + self.half_extent) / self.cell_size).astype(np.int64), 0, gs - 1)
        flat_idx = j * gs + i

        batch_max = np.full(gs * gs, -np.inf, dtype=np.float64)
        np.maximum.at(batch_max, flat_idx, z)
        touched = np.isfinite(batch_max)
        if not touched.any():
            return

        flat_height = self.height.reshape(-1)
        flat_seen = self.seen.reshape(-1)
        # Pull the (already-decayed) estimate back up to at least this
        # frame's observed max -- an obstacle appears almost immediately
        # instead of slowly converging like a Kalman blend would.
        flat_height[touched] = np.maximum(flat_height[touched], batch_max[touched])
        flat_seen[touched] = True

    def _update_kalman(self, points_xyz: np.ndarray):
        # Predict step: every cell's uncertainty grows a little every frame,
        # so stale estimates can eventually be overwritten/relaxed.
        self.variance += self.process_noise

        if points_xyz.shape[0] == 0:
            return

        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        z = points_xyz[:, 2].astype(np.float64)

        in_bounds = (np.abs(x) < self.half_extent) & (np.abs(y) < self.half_extent)
        x, y, z = x[in_bounds], y[in_bounds], z[in_bounds]
        if x.shape[0] == 0:
            return

        gs = self.grid_size
        i = np.clip(((x + self.half_extent) / self.cell_size).astype(np.int64), 0, gs - 1)
        j = np.clip(((y + self.half_extent) / self.cell_size).astype(np.int64), 0, gs - 1)
        flat_idx = j * gs + i

        # Batch mean + count per touched cell this frame.
        sums = np.zeros(gs * gs, dtype=np.float64)
        counts = np.zeros(gs * gs, dtype=np.int64)
        np.add.at(sums, flat_idx, z)
        np.add.at(counts, flat_idx, 1)

        touched = counts > 0
        if not touched.any():
            return
        batch_mean = sums[touched] / counts[touched]
        # Averaging N iid noisy measurements shrinks the effective measurement
        # variance by 1/N (standard error of the mean) -- but cap N so a
        # single dense frame can never fully overwrite prior history (see
        # `max_effective_count` in __init__).
        effective_counts = np.minimum(counts[touched], self.max_effective_count)
        batch_meas_var = self.measurement_noise / effective_counts

        flat_height = self.height.reshape(-1)
        flat_variance = self.variance.reshape(-1)
        flat_seen = self.seen.reshape(-1)

        prior_h = flat_height[touched]
        prior_v = flat_variance[touched]

        # Scalar Kalman measurement update.
        kalman_gain = prior_v / (prior_v + batch_meas_var)
        new_h = prior_h + kalman_gain * (batch_mean - prior_h)
        new_v = (1.0 - kalman_gain) * prior_v

        flat_height[touched] = new_h
        flat_variance[touched] = new_v
        flat_seen[touched] = True

    def query_grid(self) -> dict:
        """Produce the final egocentric height grid (the "inference" stage):
        kNN + inverse-distance-weighted interpolation over confidently-fused
        cells, matching `interpolateHeightKNN` in
        `videomimic_inference_real.cpp`.
        """
        gs = self.grid_size
        if self.mode == "decay_max":
            # No variance concept in this mode -- any cell we've ever touched
            # (and hasn't fully decayed away) is directly usable.
            confident = self.seen
        else:
            confident = self.seen & (self.variance <= self.confident_variance)
        flat_confident = confident.reshape(-1)

        out = np.full(gs * gs, self.empty_sentinel, dtype=np.float32)

        n_confident = int(flat_confident.sum())
        if n_confident == 0:
            return {
                "grid": out.reshape(gs, gs),
                "grid_size": np.asarray([gs], dtype=np.float32),
                "half_extent": np.asarray([self.half_extent], dtype=np.float32),
                "cell_size": np.asarray([self.cell_size], dtype=np.float32),
                "empty_sentinel": np.asarray([self.empty_sentinel], dtype=np.float32),
            }

        try:
            from scipy.spatial import cKDTree
            src_xy = self._cell_centers_xy[flat_confident]
            src_h = self.height.reshape(-1)[flat_confident]
            tree = cKDTree(src_xy)
            k = min(self.knn_k, src_xy.shape[0])
            dists, idxs = tree.query(self._cell_centers_xy, k=k)
            if k == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]

            valid_query = (
                (dists[:, 0] <= self.knn_max_distance) if self.knn_max_distance > 0 else
                np.ones(dists.shape[0], dtype=bool)
            )
            eps = 1e-9
            weights = 1.0 / np.maximum(dists, eps)
            weights /= weights.sum(axis=1, keepdims=True)
            interp = (weights * src_h[idxs]).sum(axis=1)

            out[valid_query] = interp[valid_query].astype(np.float32)
            out[~valid_query] = np.float32(self.default_height)
        except ImportError:
            # scipy unavailable: fall back to just publishing the confident
            # cells directly (no interpolation/fill of gaps).
            out[flat_confident] = self.height.reshape(-1)[flat_confident].astype(np.float32)

        return {
            "grid": out.reshape(gs, gs),
            "grid_size": np.asarray([gs], dtype=np.float32),
            "half_extent": np.asarray([self.half_extent], dtype=np.float32),
            "cell_size": np.asarray([self.cell_size], dtype=np.float32),
            "empty_sentinel": np.asarray([self.empty_sentinel], dtype=np.float32),
        }


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


def heightmap_to_image(data: dict, scale: int = 8, max_range: float = 0.0,
                        fixed_vmin=None, fixed_vmax=None, obstacle_thresh=None):
    """Colorize a heightmap grid into a BGR uint8 image for cv2.imshow.

    If max_range > 0, crop the grid to just the square window of cells within
    that many metres of the center (the sensor), instead of the full grid.

    By default the color scale auto-normalizes to the current frame's
    min/max height -- convenient for a quick look, but it means real
    frame-to-frame changes in absolute height can be invisible (they just
    get rescaled back into the same 0-255 range each frame). Pass
    `fixed_vmin`/`fixed_vmax` (e.g. from `--heightmap-vmin`/`--heightmap-vmax`)
    to use an absolute scale instead, so an approaching object's true height
    change is visually obvious across frames.

    If `obstacle_thresh` is set (metres above 0), skip the continuous
    colormap entirely and instead render a much easier-to-read binary mask:
    bright red = "something is here" (height above threshold), dark = free
    space, gray = never observed. This directly answers "what's around the
    robot right now" instead of asking you to read subtle color gradients.
    """
    import cv2

    grid = data["grid"].astype(np.float32)
    half_extent = float(data["half_extent"][0]) if "half_extent" in data else float("nan")
    cell_size = float(data["cell_size"][0]) if "cell_size" in data else float("nan")

    if max_range > 0 and cell_size > 0 and not np.isnan(half_extent):
        gsize = grid.shape[0]
        center = gsize / 2.0
        half_cells = int(np.ceil(max_range / cell_size))
        lo = max(0, int(center - half_cells))
        hi = min(gsize, int(center + half_cells))
        grid = grid[lo:hi, lo:hi]

    sentinel = float(data["empty_sentinel"][0]) if "empty_sentinel" in data else -1e6
    valid = grid > (sentinel / 2)

    if obstacle_thresh is not None:
        obstacle = valid & (grid >= obstacle_thresh)
        img = np.zeros((*grid.shape, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)          # unobserved -> dark gray
        img[valid & ~obstacle] = (40, 90, 40)   # observed, free space -> dim green
        img[obstacle] = (0, 0, 255)             # obstacle -> solid red (BGR)
        img = np.flipud(img)
        if scale > 1:
            img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                              interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"{grid.shape[0]}x{grid.shape[1]} half_extent={half_extent:.2f}m "
                          f"cell={cell_size:.2f}m obstacle_thresh={obstacle_thresh:.2f}m "
                          f"(red=obstacle, green=free, gray=unseen)",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    if fixed_vmin is not None and fixed_vmax is not None:
        vmin, vmax = fixed_vmin, fixed_vmax
    elif valid.any():
        vmin = grid[valid].min()
        vmax = grid[valid].max()
    else:
        vmin, vmax = 0.0, 1.0
    span = max(vmax - vmin, 1e-6)

    norm = np.zeros_like(grid, dtype=np.uint8)
    norm[valid] = np.clip(((grid[valid] - vmin) / span) * 255.0, 0, 255).astype(np.uint8)

    img = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    img[~valid] = (30, 30, 30)  # dark gray for cells with no returns

    img = np.flipud(img)  # +Y at the top, matching print_heightmap
    if scale > 1:
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                          interpolation=cv2.INTER_NEAREST)

    half_extent = float(data["half_extent"][0]) if "half_extent" in data else float("nan")
    cell_size = float(data["cell_size"][0]) if "cell_size" in data else float("nan")
    cv2.putText(img, f"{grid.shape[0]}x{grid.shape[1]} half_extent={half_extent:.2f}m "
                      f"cell={cell_size:.2f}m z=[{vmin:.2f},{vmax:.2f}]m",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", type=str, default="localhost",
                     help="Publisher IP (the robot / DDS-connected machine running lidar_zmq_publisher.py)")
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--topic-state", default="lidar_imu")
    ap.add_argument("--topic-cloud", default="lidar_cloud")
    ap.add_argument("--topic-heightmap", default="lidar_heightmap")
    ap.add_argument("--no-heightmap", action="store_true",
                     help="Don't subscribe to / print the height map topic")
    ap.add_argument("--seconds", type=float, default=0.0, help="Stop after N seconds (0 = run forever)")
    ap.add_argument("--record", type=str, default=None, metavar="PATH",
                    help="Record raw point-cloud frames AND IMU samples to this "
                         ".npz (written on exit, including Ctrl+C). Clouds are "
                         "stored ragged as cloud_points + cloud_offsets; see "
                         "LidarRecorder / load_recording(). Buffered in RAM "
                         "(~2.4 MB/s at 20k pts @ 10Hz) -- bound long captures "
                         "with --seconds.")
    ap.add_argument("--record-filtered", action="store_true",
                    help="Record the --max-range-filtered cloud instead of the "
                         "raw published cloud (default: record raw, so the "
                         "recording is independent of view settings)")
    ap.add_argument("--view", action="store_true",
                     help="Live-update an Open3D window with the incoming point cloud (requires open3d)")
    ap.add_argument("--view-heightmap", action="store_true",
                     help="Live-update an OpenCV window with a colorized top-down height map (requires opencv-python)")
    ap.add_argument("--heightmap-scale", type=int, default=8,
                     help="Pixel scale factor for the height map image (each grid cell -> NxN pixels)")
    ap.add_argument("--max-range", type=float, default=0.0,
                     help="Only show points/cells within this many metres of the sensor "
                          "(0 = no filtering, i.e. show everything)")
    ap.add_argument("--range-3d", action="store_true",
                     help="Use full 3D distance for --max-range instead of the default XY-planar distance")
    ap.add_argument("--heightmap-from-cloud", action="store_true",
                     help="Build the height map client-side from raw `lidar_cloud` points "
                          "instead of subscribing to the publisher's `lidar_heightmap` topic "
                          "(implies --no-heightmap; no publisher-side changes needed)")
    ap.add_argument("--heightmap-grid-size", type=int, default=21,
                     help="Cells per side when using --heightmap-from-cloud "
                          "(e.g. 21 cells @ 0.1m cell size = 2.1m extent, i.e. ~1m radius. "
                          "Note VideoMimic's actual policy grid is 11x11 @ 0.1m = 1.1m "
                          "extent -- use --heightmap-grid-size 11 to match that exactly, "
                          "e.g. for sanity-checking against the real GRID_X/Y_SIDE_LENGTH.)")
    ap.add_argument("--heightmap-cell-size", type=float, default=0.1,
                     help="Metres per cell when using --heightmap-from-cloud")
    ap.add_argument("--heightmap-agg", choices=["max", "mean"], default="max",
                     help="Per-cell height aggregation when using --heightmap-from-cloud "
                          "(ignored if --heightmap-fused is set)")
    ap.add_argument("--heightmap-vmin", type=float, default=None,
                     help="Fixed lower bound (metres) for the heightmap color scale. If "
                          "unset, the color scale auto-normalizes to each frame's own min/max, "
                          "which can visually hide real frame-to-frame height changes. Set "
                          "both --heightmap-vmin/--heightmap-vmax to make changes obvious.")
    ap.add_argument("--heightmap-vmax", type=float, default=None,
                     help="Fixed upper bound (metres) for the heightmap color scale")
    ap.add_argument("--obstacle-thresh", type=float, default=None,
                     help="Instead of a continuous color gradient (hard to read), render a "
                          "simple binary mask: red = obstacle (height >= this many metres "
                          "above the floor/default height), green = confirmed free space, "
                          "gray = not yet observed. Much easier to tell 'what's around the "
                          "robot' at a glance, e.g. --obstacle-thresh 0.15")
    ap.add_argument("--heightmap-fused", action="store_true",
                     help="Instead of naive single-frame binning, fuse LiDAR frames over "
                          "time with a per-cell Kalman filter + kNN/IDW query, mirroring "
                          "elevation_mapping_humanoid + VideoMimic's "
                          "videomimic_inference_real.cpp ElevationCloudCallback. Implies "
                          "--heightmap-from-cloud. Assumes the sensor is roughly stationary "
                          "between frames (no odometry input available here).")
    ap.add_argument("--fusion-mode", choices=["decay_max", "kalman"], default="decay_max",
                     help="'decay_max' (default): legible obstacle-tracking mode -- each cell "
                          "tracks a decaying max height, reacts almost immediately to new "
                          "obstacles and fades out if they leave. 'kalman': the original "
                          "continuous elevation-surface fusion (faithful to "
                          "elevation_mapping_humanoid/VideoMimic, but the smooth surface is "
                          "harder to read at a glance for 'what's approaching').")
    ap.add_argument("--fusion-decay", type=float, default=0.9,
                     help="(decay_max mode only) per-frame decay factor toward the default/"
                          "floor height; e.g. 0.9 => ~90%% relaxed after ~22 frames without "
                          "re-observation. Lower = faster fade-out, higher = more persistent.")
    ap.add_argument("--fusion-process-noise", type=float, default=2e-4,
                     help="Per-frame variance growth added to every cell (predict step)")
    ap.add_argument("--fusion-measurement-noise", type=float, default=1e-3,
                     help="Assumed per-point LiDAR height measurement noise variance")
    ap.add_argument("--fusion-init-variance", type=float, default=1.0,
                     help="Initial variance for never-yet-seen cells")
    ap.add_argument("--fusion-confident-variance", type=float, default=5e-2,
                     help="A cell must have variance <= this to be used as a kNN source "
                          "point in the interpolation stage")
    ap.add_argument("--fusion-knn-k", type=int, default=3,
                     help="k for the kNN/IDW interpolation stage (matches VideoMimic's KNN_K)")
    ap.add_argument("--fusion-knn-max-distance", type=float, default=0.15,
                     help="Reject (use default height) if nearest confident cell is farther "
                          "than this many metres (matches VideoMimic's KNN_MAX_DISTANCE)")
    ap.add_argument("--fusion-default-height", type=float, default=0.0,
                     help="Height used when no confident cell is within "
                          "--fusion-knn-max-distance (matches VideoMimic's "
                          "KNN_DEFAULT_HEIGHT_OFFSET=0.85 in intent -- an *optimistic* 'assume "
                          "flat ground' fallback for unobserved cells, NOT 'unknown'. Their "
                          "0.85 is in torso_z-minus-ground_z units (nominal torso height above "
                          "flat ground, via TF); we report raw sensor-frame Z here since we "
                          "have no torso pose, so the equivalent absolute default is usually "
                          "0.0, not 0.85 -- adjust to your sensor's actual mount height above "
                          "the floor if you want a literal match.)")
    ap.add_argument("--fusion-max-count", type=int, default=8,
                     help="Cap on the per-frame point count used to shrink a cell's effective "
                          "measurement variance. Without this cap, a dense LiDAR frame drives "
                          "the Kalman gain to ~1 every frame (i.e. near-total overwrite by the "
                          "newest frame alone -- no real temporal smoothing, just noisy raw "
                          "per-frame data). Lower = smoother/more stable but slower to react "
                          "to a real persistent change; higher = more reactive but noisier.")
    ap.add_argument("--debug-diff", action="store_true",
                     help="With --heightmap-fused/--heightmap-from-cloud, print each frame's "
                          "max |height change| + confident-cell count vs. the previous frame, "
                          "so you can confirm the map is actually reacting to new events "
                          "(e.g. someone walking closer) in real time")
    args = ap.parse_args()

    if args.heightmap_fused:
        args.heightmap_from_cloud = True
    if args.heightmap_from_cloud:
        args.no_heightmap = True

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    url = f"tcp://{args.host}:{args.port}"
    socket.connect(url)
    socket.setsockopt_string(zmq.SUBSCRIBE, args.topic_state)
    socket.setsockopt_string(zmq.SUBSCRIBE, args.topic_cloud)
    if not args.no_heightmap:
        socket.setsockopt_string(zmq.SUBSCRIBE, args.topic_heightmap)
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    print("=" * 70)
    print("LiDAR ZMQ client")
    print(f"  Connecting to: {url}")
    topics_str = f"'{args.topic_state}', '{args.topic_cloud}'"
    if not args.no_heightmap:
        topics_str += f", '{args.topic_heightmap}'"
    print(f"  Topics: {topics_str}")
    if args.heightmap_from_cloud:
        extent = args.heightmap_grid_size * args.heightmap_cell_size
        mode = ("fused (Kalman + kNN/IDW, like elevation_mapping_humanoid + VideoMimic)"
                if args.heightmap_fused else f"naive per-frame binning (agg={args.heightmap_agg})")
        print(f"  Height map: computed client-side from '{args.topic_cloud}' "
              f"({args.heightmap_grid_size}x{args.heightmap_grid_size} cells @ "
              f"{args.heightmap_cell_size:.2f}m => {extent:.2f}m extent), mode: {mode}")
    print("=" * 70)

    fuser = None
    if args.heightmap_fused:
        fuser = ElevationMapFuser(
            grid_size=args.heightmap_grid_size,
            cell_size=args.heightmap_cell_size,
            process_noise=args.fusion_process_noise,
            measurement_noise=args.fusion_measurement_noise,
            init_variance=args.fusion_init_variance,
            confident_variance=args.fusion_confident_variance,
            knn_k=args.fusion_knn_k,
            knn_max_distance=args.fusion_knn_max_distance,
            default_height=args.fusion_default_height,
            max_effective_count=args.fusion_max_count,
            mode=args.fusion_mode,
            decay=args.fusion_decay,
        )

    prev_confident_grid = None  # for --debug-diff frame-to-frame change tracking

    vis = None
    pcd = None
    first_cloud = True
    if args.view:
        try:
            import open3d as o3d
            vis = o3d.visualization.Visualizer()
            vis.create_window("LiDAR live view", width=1024, height=768)
            pcd = o3d.geometry.PointCloud()
            vis.add_geometry(pcd)
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.05, 0.05, 0.05])
            opt.point_size = 2.5
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            vis.add_geometry(axes)
        except ImportError:
            print("[--view requires open3d: pip install open3d]")
            vis = None

    heightmap_view = False
    if args.view_heightmap:
        try:
            import cv2  # noqa: F401  (import check only)
            heightmap_view = True
        except ImportError:
            print("[--view-heightmap requires opencv-python: pip install opencv-python]")
            heightmap_view = False

    n_state = 0
    n_cloud = 0
    n_heightmap = 0
    n_timeouts = 0
    start_t = time.time()

    recorder = None
    if args.record:
        recorder = LidarRecorder(args.record, record_filtered=args.record_filtered)
        print(f"[record] recording to {args.record} "
              f"({'range-filtered' if args.record_filtered else 'raw'} clouds + IMU); "
              f"Ctrl+C or --seconds to stop and write")

    try:
        while args.seconds <= 0 or (time.time() - start_t) < args.seconds:
            try:
                raw = socket.recv()
            except zmq.Again:
                n_timeouts += 1
                print(f"  [timeout] no message in last 1s (total: {n_timeouts})")
                continue

            if raw.startswith(args.topic_state.encode("utf-8")):
                data, err = parse_message(raw, args.topic_state)
                if err:
                    print(f"  [parse error/state] {err}")
                    continue
                n_state += 1
                if recorder is not None:
                    recorder.add_imu(data, time.time())
                print_state(data)

            elif (not args.no_heightmap) and raw.startswith(args.topic_heightmap.encode("utf-8")):
                data, err = parse_message(raw, args.topic_heightmap)
                if err:
                    print(f"  [parse error/heightmap] {err}")
                    continue
                n_heightmap += 1
                print_heightmap(data)

                if heightmap_view:
                    try:
                        import cv2
                        img = heightmap_to_image(
                            data, scale=args.heightmap_scale, max_range=args.max_range,
                            fixed_vmin=args.heightmap_vmin, fixed_vmax=args.heightmap_vmax,
                            obstacle_thresh=args.obstacle_thresh)
                        cv2.imshow("LiDAR height map", img)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                    except Exception as e:
                        print(f"  [heightmap view update failed] {e}")

            elif raw.startswith(args.topic_cloud.encode("utf-8")):
                data, err = parse_message(raw, args.topic_cloud)
                if err:
                    print(f"  [parse error/cloud] {err}")
                    continue
                n_cloud += 1
                # Record the RAW cloud by default, before any view-only range
                # filtering below, so the recording doesn't silently depend on
                # --max-range/--range-3d. --record-filtered opts into the
                # filtered cloud instead.
                if recorder is not None and not args.record_filtered:
                    recorder.add_cloud(data, time.time())
                if args.max_range > 0:
                    data = data.copy()
                    data["points"] = filter_points_by_range(
                        data["points"], args.max_range, planar=not args.range_3d)
                if recorder is not None and args.record_filtered:
                    recorder.add_cloud(data, time.time())
                print_cloud(data)

                if vis is not None:
                    try:
                        import open3d as o3d
                        pts = data["points"].astype(np.float64)
                        pcd.points = o3d.utility.Vector3dVector(pts)
                        if pts.shape[0] > 0:
                            # Color by height (z) so the cloud is visible against the dark bg.
                            z = pts[:, 2]
                            z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)
                            colors = np.stack(
                                [z_norm, 1.0 - z_norm, np.full_like(z_norm, 0.6)], axis=1)
                            pcd.colors = o3d.utility.Vector3dVector(colors)
                        vis.update_geometry(pcd)
                        if first_cloud and pts.shape[0] > 0:
                            vis.reset_view_point(True)
                            first_cloud = False
                        vis.poll_events()
                        vis.update_renderer()
                    except Exception as e:
                        print(f"  [view update failed] {e}")

                if args.heightmap_from_cloud:
                    if fuser is not None:
                        fuser.update(data["points"])
                        hm_data = fuser.query_grid()
                    else:
                        hm_data = points_to_heightmap(
                            data["points"],
                            grid_size=args.heightmap_grid_size,
                            cell_size=args.heightmap_cell_size,
                            agg=args.heightmap_agg,
                        )
                    n_heightmap += 1
                    print_heightmap(hm_data)

                    if args.debug_diff:
                        cur_grid = hm_data["grid"]
                        cur_sentinel = float(hm_data["empty_sentinel"][0])
                        cur_confident = cur_grid > (cur_sentinel / 2)
                        n_confident = int(cur_confident.sum())
                        if prev_confident_grid is not None:
                            both_confident = cur_confident & prev_confident_grid[1]
                            if both_confident.any():
                                max_diff = float(np.abs(
                                    cur_grid[both_confident] - prev_confident_grid[0][both_confident]
                                ).max())
                            else:
                                max_diff = float("nan")
                            print(f"  [debug-diff] confident_cells={n_confident} "
                                  f"max|Δheight vs prev frame|={max_diff:.3f}m")
                        prev_confident_grid = (cur_grid.copy(), cur_confident)

                    if heightmap_view:
                        try:
                            import cv2
                            win_name = ("LiDAR height map (fused)" if fuser is not None
                                        else "LiDAR height map (from cloud)")
                            img = heightmap_to_image(
                                hm_data, scale=args.heightmap_scale, max_range=args.max_range,
                                fixed_vmin=args.heightmap_vmin, fixed_vmax=args.heightmap_vmax,
                                obstacle_thresh=args.obstacle_thresh)
                            if fuser is not None:
                                cv2.putText(img, "[r] reset fused map", (6, img.shape[0] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                                            cv2.LINE_AA)
                            cv2.imshow(win_name, img)
                            key = cv2.waitKey(1) & 0xFF
                            if key in (27, ord("q")):
                                break
                            if key == ord("r") and fuser is not None:
                                fuser.reset()
                                prev_confident_grid = None
                                print("  [heightmap] fused map reset")
                        except Exception as e:
                            print(f"  [heightmap view update failed] {e}")

    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            recorder.save()
        if vis is not None:
            vis.destroy_window()
        if heightmap_view:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

    print(f"\nReceived {n_state} state messages, {n_cloud} cloud messages, "
          f"{n_heightmap} height map messages, {n_timeouts} timeouts.")


if __name__ == "__main__":
    main()
