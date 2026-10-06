#!/usr/bin/env python3
"""Receive live D435i depth frames from the G1 (see `g1_depth_publisher.py`).

RUNS ON THE WORKSTATION.

Examples:
    # live colourised 2D depth view + latency/FPS readout
    .venv_sim/bin/python perception_heightmap/g1_depth_subscriber.py --visualize

    # live 3D point-cloud view (consumes the "points" topic, requires open3d)
    .venv_sim/bin/python perception_heightmap/g1_depth_subscriber.py --visualize-points

    # headless stats only
    .venv_sim/bin/python perception_heightmap/g1_depth_subscriber.py --frames 100

    # record to an .npz for offline heightmap work
    .venv_sim/bin/python perception_heightmap/g1_depth_subscriber.py --frames 300 --save /tmp/g1_depth.npz

Keys in the 2D viewer window: q / ESC to quit.
Keys in the 3D point-cloud window: q / ESC to quit, r to reset the camera view.
"""
import argparse
import json
import shlex
import struct
import sys
import time

import numpy as np
import zmq

try:
    import lz4.frame as lz4frame
except ImportError:
    lz4frame = None


def decode(msg):
    """Split a payload into (header_dict, ndarray reshaped per header).

    Works for both the "depth" topic (uint16 raw depth image) and the
    "points" topic (organized float32 XYZ point cloud) from
    `g1_depth_publisher.py`, since both share the same
    [4-byte header length][json header][optionally-lz4 payload] framing.
    """
    (hdr_len,) = struct.unpack("<I", msg[:4])
    hdr = json.loads(msg[4:4 + hdr_len])
    blob = msg[4 + hdr_len:]
    if hdr["compress"] == "lz4":
        if lz4frame is None:
            raise RuntimeError("frame is lz4-compressed but lz4 is not installed")
        blob = lz4frame.decompress(blob)
    img = np.frombuffer(blob, dtype=hdr["dtype"]).reshape(hdr["shape"])
    return hdr, img


def deproject(img, hdr, stride=4):
    """Depth image -> Nx3 point cloud in the camera optical frame (metres).

    Only needed as a fallback if the "points" topic isn't available (e.g. an
    older publisher, or `g1_depth_publisher.py --no-points`); when the
    "points" topic is used, points are already deprojected by the RealSense
    SDK on the publisher (see `organized_points_to_xyz`).
    """
    k = hdr["intrinsics"]
    scale = hdr["depth_scale"]
    sub = img[::stride, ::stride]
    ys, xs = np.nonzero(sub)
    z = sub[ys, xs].astype(np.float32) * scale
    # undo the striding to get true pixel coords
    px = xs * stride
    py = ys * stride
    x = (px - k["ppx"]) / k["fx"] * z
    y = (py - k["ppy"]) / k["fy"] * z
    return np.stack([x, y, z], axis=1)


def organized_points_to_xyz(points_organized: np.ndarray) -> np.ndarray:
    """Organized (H, W, 3) float32 point cloud from the "points" topic ->
    Nx3 valid points (drops the (0, 0, 0) invalid/no-return sentinel pixels
    RealSense uses, per `g1_depth_publisher.build_points_header`)."""
    flat = points_organized.reshape(-1, 3)
    valid = np.any(flat != 0.0, axis=1)
    return flat[valid]


def optical_to_map_frame(pts_opt: np.ndarray, cam_pitch_deg: float = 0.0,
                          cam_offset=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera optical frame (+X right, +Y down, +Z forward) -> a simple
    fixed "map" frame (+X forward, +Y left, +Z up), for feeding into
    `ElevationGridMap`. This is the same kind of approximate, TF-free
    extrinsics transform used in `perception_heightmap/
    view_depth_heightmap.py`'s `optical_to_body_frame` -- see that module's
    docstring for the exact axis-remap/pitch-rotation math and its caveats.
    """
    if pts_opt.shape[0] == 0:
        return pts_opt
    x_opt, y_opt, z_opt = pts_opt[:, 0], pts_opt[:, 1], pts_opt[:, 2]
    x_body = z_opt
    y_body = -x_opt
    z_body = -y_opt
    if cam_pitch_deg != 0.0:
        theta = np.deg2rad(cam_pitch_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x_body * cos_t + z_body * sin_t
        z_rot = -x_body * sin_t + z_body * cos_t
        x_body, z_body = x_rot, z_rot
    x_body = x_body + cam_offset[0]
    y_body = y_body + cam_offset[1]
    z_body = z_body + cam_offset[2]
    return np.stack([x_body, y_body, z_body], axis=1).astype(np.float32)


def filter_points_footprint(pts_opt: np.ndarray, box_size: float) -> np.ndarray:
    """Keep only points within a `box_size` x `box_size` square footprint
    directly in front of the camera, in the camera OPTICAL frame (+X right,
    +Y down, +Z forward -- i.e. the raw, pre-`optical_to_map_frame` points):
    lateral |X| <= box_size/2, depth 0 <= Z <= box_size. Vertical (Y) extent
    is left unclipped, since "2m x 2m" naturally reads as a footprint area
    (width x depth) rather than a full 3D cube. `box_size <= 0` disables
    filtering (returns `pts_opt` unchanged)."""
    if box_size <= 0 or pts_opt.shape[0] == 0:
        return pts_opt
    half = box_size / 2.0
    mask = (np.abs(pts_opt[:, 0]) <= half) & (pts_opt[:, 2] >= 0.0) & (pts_opt[:, 2] <= box_size)
    return pts_opt[mask]


def set_open3d_view_to_camera(vis3d, intr: dict):
    """Point the Open3D virtual camera at the SAME pose as the physical
    camera: eye at the origin, looking along +Z, +X right, +Y down -- which
    is both the RealSense optical-frame convention our raw point cloud is
    already expressed in, AND Open3D/OpenCV's own pinhole-camera convention.
    So an IDENTITY extrinsic reproduces exactly what the camera itself sees,
    rather than the arbitrary fitted angle `reset_view_point()` picks based
    on the cloud's bounding box."""
    import open3d as o3d
    ctr = vis3d.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    w, h = int(intr["width"]), int(intr["height"])
    params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        w, h, intr["fx"], intr["fy"], intr["ppx"], intr["ppy"])
    params.extrinsic = np.eye(4)
    try:
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    except TypeError:
        # older open3d versions don't have the allow_arbitrary kwarg
        ctr.convert_from_pinhole_camera_parameters(params)


class ElevationGridMap:
    """Stage 2 of the real pipeline (and ONLY stage 2 -- see class-level
    scope note below): a persistent, **robot-centric scrolling** fused
    elevation map, mirroring `elevation_mapping`'s internal
    `grid_map_msgs/GridMap` (ANYbotics `elevation_mapping` /
    ArthurAllshire/elevation_mapping_humanoid). Structurally different from
    everything else in this script: it's NOT a point cloud, and its cells
    can be **NaN = "never observed"** -- a real epistemic "unknown".

    Stage-2 scope implemented here (all five items live entirely inside
    `update()`, before/at the moment the fused map is produced):
      1. Per-point sensor uncertainty: each incoming point's measurement
         variance is modeled as a function of its range from the sensor
         (`sigma(d) = noise_a + noise_b * d^2`, a simple quadratic stand-in
         for how RealSense-style depth noise grows with distance -- see
         `--noise-a`/`--noise-b`), instead of one constant for the whole
         cloud.
      2. Robot-centric scrolling grid: `update()` takes a `pose_delta=(dx,
         dy)` this frame and rolls the grid buffer by the corresponding
         whole number of cells *before* fusing new points, so the map stays
         centered near the robot as it moves (matching how
         `elevation_mapping` re-registers its internal buffer using live
         odometry every callback) -- newly-exposed border cells become NaN
         (truly unobserved) again.
      3. Per-cell Kalman update: unchanged in spirit from before, but now
         consumes the per-point variance from (1) instead of a single
         constant.
      4. Drift propagation from pose uncertainty: variance growth for
         already-observed cells is now a proper rate integrated over the
         real elapsed time `dt` since the last `update()` call (previously
         a fixed constant added every call regardless of real timing),
         PLUS an additional term driven by `pose_drift_std` (an assumed
         odometry position-uncertainty growth rate, metres/sqrt(second)),
         standing in for "use robot pose covariance as input" since no real
         pose covariance is available here.

    Explicitly OUT of scope here (per the stage boundary): the lossy
    flatten-to-XYZ step (stage 2->3 handoff, `to_elevation_cloud()` below is
    provided only as a convenience bridge and does not add any new Stage-3
    logic), and the deterministic KD-tree/IDW interpolation stages 4-5,
    which have zero Bayesian/variance reasoning and live entirely in
    `view_depth_heightmap.py` (`interpolate_height_idw`), untouched here.

    IMPORTANT limitation vs. the real node: this class still has no actual
    odometry/pose-covariance input -- `pose_delta` and `pose_drift_std` are
    parameters you must supply yourself (e.g. from a real odometry source)
    for (2) and (4) to do anything beyond their zero/no-op defaults.
    """

    def __init__(self, resolution: float = 0.05, length_x: float = 6.0,
                 length_y: float = 6.0, process_noise_rate: float = 1e-4,
                 pose_drift_std: float = 0.0,
                 noise_a: float = 1e-3, noise_b: float = 2e-3,
                 max_effective_count: int = 8,
                 min_variance: float = 1e-5, max_variance: float = 1.0):
        """
        process_noise_rate: variance growth RATE (metres^2/second) for
            already-observed cells -- integrated over the real elapsed `dt`
            between `update()` calls (see item 4 above), not a fixed
            per-call increment.
        pose_drift_std: assumed odometry position-uncertainty growth rate
            (metres / sqrt(second)); contributes an extra `(pose_drift_std)^2
            * dt` to the same variance growth, standing in for "pose
            covariance as input" (item 4).
        noise_a, noise_b: per-point measurement std-dev model
            `sigma(d) = noise_a + noise_b * d^2` where `d` is the point's
            range from the sensor (item 1); `noise_a` is the noise floor at
            zero range, `noise_b` controls how fast it grows with distance.
        """
        self.resolution = resolution
        self.length_x = length_x
        self.length_y = length_y
        self.process_noise_rate = process_noise_rate
        self.pose_drift_std = pose_drift_std
        self.noise_a = noise_a
        self.noise_b = noise_b
        self.max_effective_count = max_effective_count
        self.min_variance = min_variance
        self.max_variance = max_variance
        self.min_horizontal_variance = resolution ** 2  # matches real system's default order of magnitude

        self.size_x = int(round(length_x / resolution))
        self.size_y = int(round(length_y / resolution))
        self.position = np.array([0.0, 0.0], dtype=np.float64)  # grid center, map frame
        self._last_stamp = None

        shape = (self.size_y, self.size_x)  # row=Y, col=X
        self.elevation = np.full(shape, np.nan, dtype=np.float64)
        self.variance = np.full(shape, np.nan, dtype=np.float64)
        self.time = np.full(shape, np.nan, dtype=np.float64)
        self.n_obs = np.zeros(shape, dtype=np.int64)  # diagnostic only

        # Horizontal position-uncertainty covariance layers (2x2 covariance
        # per cell: [[horizontal_variance_x, horizontal_variance_xy],
        # [horizontal_variance_xy, horizontal_variance_y]]) -- these drive
        # the `fuse()` error-ellipse spatial smoothing below, matching the
        # real `ElevationMap`'s layers of the same name. Reset to
        # `min_horizontal_variance`/0 on every direct observation (matching
        # `ElevationMap::add()`); simplified here to grow at the same
        # (isotropic) rate as `variance` between observations, rather than
        # the real system's full anisotropic lever-arm Jacobian growth.
        self.horizontal_variance_x = np.full(shape, np.nan, dtype=np.float64)
        self.horizontal_variance_y = np.full(shape, np.nan, dtype=np.float64)
        self.horizontal_variance_xy = np.full(shape, 0.0, dtype=np.float64)

        # Populated by `fuse_all()` -- the actual "published" map in the real
        # system, distinct from the raw per-cell Kalman-fused layers above.
        self.fused_elevation = np.full(shape, np.nan, dtype=np.float64)
        self.fused_variance = np.full(shape, np.nan, dtype=np.float64)
        self.lower_bound = np.full(shape, np.nan, dtype=np.float64)
        self.upper_bound = np.full(shape, np.nan, dtype=np.float64)

    def _scroll(self, dx: float, dy: float):
        """Robot-centric scrolling: roll the grid buffer by the nearest
        integer number of cells so it stays centered near the robot as it
        moves `(dx, dy)` metres in the map frame (sub-cell motion is
        ignored). Newly-exposed border cells become NaN (unobserved) again.
        """
        shift_cells_x = int(round(dx / self.resolution))
        shift_cells_y = int(round(dy / self.resolution))
        if shift_cells_x == 0 and shift_cells_y == 0:
            return

        for arr in (self.elevation, self.variance, self.time,
                    self.horizontal_variance_x, self.horizontal_variance_y,
                    self.horizontal_variance_xy):
            arr[:] = np.roll(arr, (-shift_cells_y, -shift_cells_x), axis=(0, 1))
        self.n_obs[:] = np.roll(self.n_obs, (-shift_cells_y, -shift_cells_x), axis=(0, 1))

        def _wipe(y_slice, x_slice):
            self.elevation[y_slice, x_slice] = np.nan
            self.variance[y_slice, x_slice] = np.nan
            self.time[y_slice, x_slice] = np.nan
            self.n_obs[y_slice, x_slice] = 0

        if shift_cells_y > 0:
            _wipe(slice(-shift_cells_y, None), slice(None))
        elif shift_cells_y < 0:
            _wipe(slice(None, -shift_cells_y), slice(None))
        if shift_cells_x > 0:
            _wipe(slice(None), slice(-shift_cells_x, None))
        elif shift_cells_x < 0:
            _wipe(slice(None), slice(None, -shift_cells_x))

        self.position += np.array([dx, dy])

    def update(self, points_xyz_map_frame: np.ndarray, point_ranges: np.ndarray,
               stamp: float, pose_delta=(0.0, 0.0)):
        """Fuse one new frame's points (already in this map's fixed/robot-
        centric frame) into the persistent grid.

        points_xyz_map_frame: (N, 3) points already transformed into the map
            frame (e.g. via `optical_to_map_frame`).
        point_ranges: (N,) per-point distance from the sensor (metres),
            driving the per-point measurement-noise model (item 1).
        stamp: timestamp (seconds) for this frame, used to integrate the
            drift-propagation variance growth (item 4) over real elapsed
            time since the previous `update()` call.
        pose_delta: (dx, dy) metres the robot/sensor has moved in the map
            frame since the last call -- triggers the robot-centric scroll
            (item 2) before fusing. Defaults to (0, 0) (no odometry).

        Cells with no points this frame are left exactly as they were
        (matching the real callback's "empty message -> keep previous
        heights" behavior, just per-cell instead of whole-map).
        """
        if pose_delta[0] != 0.0 or pose_delta[1] != 0.0:
            self._scroll(pose_delta[0], pose_delta[1])

        # --- Item 4: drift propagation from (proxy) pose uncertainty ---
        dt = 0.0 if self._last_stamp is None else max(stamp - self._last_stamp, 0.0)
        self._last_stamp = stamp
        variance_growth = self.process_noise_rate * dt + (self.pose_drift_std ** 2) * dt
        if variance_growth > 0:
            observed = ~np.isnan(self.variance)
            self.variance[observed] = np.minimum(
                self.variance[observed] + variance_growth, self.max_variance)
            # Simplified stand-in for RobotMotionMapUpdater's anisotropic
            # lever-arm growth of horizontal_variance_x/y/xy: grown at the
            # same isotropic rate here (real system's growth also depends on
            # each cell's distance from the robot -- not modeled).
            self.horizontal_variance_x[observed] += variance_growth
            self.horizontal_variance_y[observed] += variance_growth

        if points_xyz_map_frame.shape[0] == 0:
            return

        x = points_xyz_map_frame[:, 0]
        y = points_xyz_map_frame[:, 1]
        z = points_xyz_map_frame[:, 2].astype(np.float64)

        half_x = self.size_x / 2.0 * self.resolution
        half_y = self.size_y / 2.0 * self.resolution
        rel_x = x - (self.position[0] - half_x)
        rel_y = y - (self.position[1] - half_y)
        in_bounds = (rel_x >= 0) & (rel_x < self.size_x * self.resolution) & \
                    (rel_y >= 0) & (rel_y < self.size_y * self.resolution)

        # --- Item 1: per-point measurement variance from a range model ---
        point_var = (self.noise_a + self.noise_b * point_ranges ** 2) ** 2

        rel_x, rel_y, z = rel_x[in_bounds], rel_y[in_bounds], z[in_bounds]
        point_var = point_var[in_bounds]
        if rel_x.shape[0] == 0:
            return

        i = np.clip((rel_x / self.resolution).astype(np.int64), 0, self.size_x - 1)
        j = np.clip((rel_y / self.resolution).astype(np.int64), 0, self.size_y - 1)
        flat_idx = j * self.size_x + i


        n_cells = self.size_x * self.size_y
        # Item 1 (continued): combine this frame's per-point measurements
        # per cell via inverse-variance weighting (proper Bayesian fusion of
        # multiple differently-noisy points), instead of a plain mean.
        precision = 1.0 / np.maximum(point_var, 1e-12)
        sum_precision = np.zeros(n_cells, dtype=np.float64)
        weighted_sum = np.zeros(n_cells, dtype=np.float64)
        counts = np.zeros(n_cells, dtype=np.int64)
        np.add.at(sum_precision, flat_idx, precision)
        np.add.at(weighted_sum, flat_idx, precision * z)
        np.add.at(counts, flat_idx, 1)

        touched = counts > 0
        if not touched.any():
            return
        batch_mean = weighted_sum[touched] / sum_precision[touched]
        # Cap how much a single dense frame can shrink the batch variance,
        # same rationale as the earlier `max_effective_count` (a frame with
        # many points-per-cell shouldn't fully overwrite history in one
        # shot).
        effective_scale = np.minimum(counts[touched], self.max_effective_count) / counts[touched]
        batch_meas_var = 1.0 / (sum_precision[touched] * effective_scale)

        flat_elevation = self.elevation.reshape(-1)
        flat_variance = self.variance.reshape(-1)
        flat_time = self.time.reshape(-1)
        flat_n_obs = self.n_obs.reshape(-1)

        prior_e = flat_elevation[touched]
        prior_v = flat_variance[touched]
        never_seen = np.isnan(prior_e)

        new_e = np.where(never_seen, batch_mean, prior_e)
        new_v = np.where(never_seen, batch_meas_var, prior_v)

        has_prior = ~never_seen
        if has_prior.any():
            kalman_gain = prior_v[has_prior] / (prior_v[has_prior] + batch_meas_var[has_prior])
            new_e[has_prior] = prior_e[has_prior] + kalman_gain * (
                batch_mean[has_prior] - prior_e[has_prior])
            new_v[has_prior] = (1.0 - kalman_gain) * prior_v[has_prior]

        new_v = np.clip(new_v, self.min_variance, self.max_variance)

        flat_elevation[touched] = new_e
        flat_variance[touched] = new_v
        flat_time[touched] = stamp
        flat_n_obs[touched] += counts[touched]

        # Horizontal variances are reset on every direct observation
        # (matches `ElevationMap::add()`: "Horizontal variances are reset").
        flat_hvx = self.horizontal_variance_x.reshape(-1)
        flat_hvy = self.horizontal_variance_y.reshape(-1)
        flat_hvxy = self.horizontal_variance_xy.reshape(-1)
        flat_hvx[touched] = self.min_horizontal_variance
        flat_hvy[touched] = self.min_horizontal_variance
        flat_hvxy[touched] = 0.0

    def fuse_all(self, uncertainty_factor: float = 2.486, max_cells_to_fuse: int = 400):
        """Stage 2's `fuse()` step: produces the actual "published" fused
        map (`fused_elevation`, `fused_variance`, `lower_bound`,
        `upper_bound`) from the raw per-cell Kalman-fused layers, via
        error-ellipse-weighted spatial smoothing over neighboring cells --
        mirroring `ElevationMap::fuse()`.

        For each observed cell, build its horizontal position-uncertainty
        ellipse from `(horizontal_variance_x, horizontal_variance_y,
        horizontal_variance_xy)` (`uncertainty_factor=2.486` => the 95.45%
        confidence ellipse, i.e. 2 sigma for a 2-DOF problem, matching the
        real system's constant), gather nearby observed cells whose center
        falls inside that ellipse, and combine them via a Gaussian-weighted
        (by Mahalanobis distance under that same covariance) average -- an
        approximation of the real system's `WeightedEmpiricalCumulativeDist
        ributionFunction`-based combination (exact weight formula wasn't
        available to inspect, so this uses the standard bivariate-Gaussian
        kernel weight for the same covariance ellipse, which is the natural
        choice for "weight by the error ellipse").

        `lower_bound`/`upper_bound` are the fused mean +/- 2 standard
        deviations (matching the real system's use of a 2-sigma band).

        This is NOT called automatically from `update()` -- exactly like the
        real system, raw-map fusion (`add()`) happens every incoming frame,
        but `fuse()` is a separate, lower-rate pass (call this periodically,
        e.g. every N frames or on a timer, since it's the expensive step).

        max_cells_to_fuse caps how many cells get fused per call (skips the
        rest, leaving their previous fused values stale) as a simple
        performance safeguard for large/dense maps -- the real system has an
        analogous `maxNumberOfCellsToFuse` cap per query area.
        """
        valid_mask = ~np.isnan(self.elevation)
        valid_idx = np.argwhere(valid_mask)
        if valid_idx.shape[0] == 0:
            return

        if valid_idx.shape[0] > max_cells_to_fuse:
            # Prioritize the most recently observed cells if we have to cap.
            order = np.argsort(-self.time[valid_mask])
            valid_idx = valid_idx[order[:max_cells_to_fuse]]

        cell_centers = self._cell_centers_xy_for_indices(valid_idx)
        all_valid_centers = self._cell_centers_xy_for_indices(np.argwhere(valid_mask))
        all_valid_elevation = self.elevation[valid_mask]
        all_valid_variance = self.variance[valid_mask]

        ellipse_extension = np.sqrt(2.0) * self.resolution

        for k in range(valid_idx.shape[0]):
            j, i = valid_idx[k]
            sigma_x2 = self.horizontal_variance_x[j, i]
            sigma_y2 = self.horizontal_variance_y[j, i]
            sigma_xy = self.horizontal_variance_xy[j, i]

            # Closed-form eigen-decomposition of the 2x2 symmetric covariance.
            trace = sigma_x2 + sigma_y2
            det = sigma_x2 * sigma_y2 - sigma_xy ** 2
            disc = max(trace ** 2 / 4.0 - det, 0.0)
            sqrt_disc = np.sqrt(disc)
            eig1 = trace / 2.0 + sqrt_disc  # major
            eig2 = max(trace / 2.0 - sqrt_disc, 1e-12)  # minor, guarded
            if abs(sigma_xy) < 1e-12 and abs(sigma_x2 - sigma_y2) < 1e-12:
                angle = 0.0
            else:
                angle = 0.5 * np.arctan2(2.0 * sigma_xy, sigma_x2 - sigma_y2)

            semi_major = uncertainty_factor * np.sqrt(eig1) + ellipse_extension
            semi_minor = uncertainty_factor * np.sqrt(eig2) + ellipse_extension

            center = cell_centers[k]
            offsets = all_valid_centers - center  # (M, 2)
            cos_a, sin_a = np.cos(-angle), np.sin(-angle)
            local_x = offsets[:, 0] * cos_a - offsets[:, 1] * sin_a
            local_y = offsets[:, 0] * sin_a + offsets[:, 1] * cos_a
            in_ellipse = (local_x / semi_major) ** 2 + (local_y / semi_minor) ** 2 <= 1.0

            if not in_ellipse.any():
                # Nothing to fuse -- fall back to the raw cell (matches the
                # real system's "i == 0: nothing to fuse" branch).
                self.fused_elevation[j, i] = self.elevation[j, i]
                self.fused_variance[j, i] = self.variance[j, i]
                std = np.sqrt(max(self.variance[j, i], 0.0))
                self.lower_bound[j, i] = self.elevation[j, i] - 2.0 * std
                self.upper_bound[j, i] = self.elevation[j, i] + 2.0 * std
                continue

            local_x_in = local_x[in_ellipse]
            local_y_in = local_y[in_ellipse]
            mahalanobis_sq = (local_x_in / max(np.sqrt(eig1), 1e-9)) ** 2 + \
                (local_y_in / max(np.sqrt(eig2), 1e-9)) ** 2
            weights = np.exp(-0.5 * mahalanobis_sq)
            weight_sum = weights.sum()
            if weight_sum <= 0:
                self.fused_elevation[j, i] = self.elevation[j, i]
                self.fused_variance[j, i] = self.variance[j, i]
                std = np.sqrt(max(self.variance[j, i], 0.0))
                self.lower_bound[j, i] = self.elevation[j, i] - 2.0 * std
                self.upper_bound[j, i] = self.elevation[j, i] + 2.0 * std
                continue

            elevations_in = all_valid_elevation[in_ellipse]
            variances_in = all_valid_variance[in_ellipse]
            weights_norm = weights / weight_sum

            fused_mean = float(np.sum(weights_norm * elevations_in))
            # Weighted variance of the neighbors around the fused mean, plus
            # each neighbor's own measurement variance (total spread used
            # for the +/-2 sigma bound), approximating the real system's
            # weighted-empirical-CDF-based lower/upper bound.
            weighted_spread = float(np.sum(weights_norm * (elevations_in - fused_mean) ** 2))
            weighted_own_var = float(np.sum(weights_norm * variances_in))
            fused_var = weighted_spread + weighted_own_var
            std = np.sqrt(max(fused_var, 0.0))

            self.fused_elevation[j, i] = fused_mean
            self.fused_variance[j, i] = fused_var
            self.lower_bound[j, i] = fused_mean - 2.0 * std
            self.upper_bound[j, i] = fused_mean + 2.0 * std

    def _cell_centers_xy_for_indices(self, indices: np.ndarray) -> np.ndarray:
        """(j, i) index pairs -> (x, y) cell-center positions in the map frame."""
        half_x = self.size_x / 2.0 * self.resolution
        half_y = self.size_y / 2.0 * self.resolution
        j = indices[:, 0].astype(np.float64)
        i = indices[:, 1].astype(np.float64)
        x = self.position[0] - half_x + (i + 0.5) * self.resolution
        y = self.position[1] - half_y + (j + 0.5) * self.resolution
        return np.stack([x, y], axis=1)

    def to_elevation_cloud(self, use_fused: bool = True) -> np.ndarray:
        """Stage 3 equivalent: flatten to an (N, 3) XYZ cloud, dropping
        variance/time/NaN semantics -- the same lossy step the real
        `elevation_map_fused_visualization` node performs before publishing
        `.../elevation_cloud`. NaN (never-observed) cells are simply
        omitted.

        use_fused=True (default, and the faithful choice): flattens
        `fused_elevation` -- the actual output of `fuse_all()` -- matching
        the real system, where Stage 3 always reads from the fused map, not
        the raw per-cell Kalman layers. Falls back to the raw `elevation`
        layer for any cell `fuse_all()` hasn't processed yet (still NaN in
        `fused_elevation`), and set use_fused=False to bypass `fuse_all()`
        entirely (e.g. if you never call it)."""
        elevation_layer = self.fused_elevation if use_fused else self.elevation
        if use_fused:
            elevation_layer = np.where(np.isnan(self.fused_elevation), self.elevation, self.fused_elevation)
        valid = ~np.isnan(elevation_layer)
        if not valid.any():
            return np.zeros((0, 3), dtype=np.float32)
        half_x = self.size_x / 2.0 * self.resolution
        half_y = self.size_y / 2.0 * self.resolution
        xs = self.position[0] - half_x + (np.arange(self.size_x) + 0.5) * self.resolution
        ys = self.position[1] - half_y + (np.arange(self.size_y) + 0.5) * self.resolution
        grid_x, grid_y = np.meshgrid(xs, ys)
        xy = np.stack([grid_x, grid_y], axis=-1)[valid]
        z = elevation_layer[valid]
        return np.concatenate([xy, z[:, None]], axis=1).astype(np.float32)

    def elevation_image(self, scale: int = 4, vmin=None, vmax=None, use_fused: bool = True):
        """Colorize the `elevation` (or `fused_elevation`) layer into a BGR
        uint8 image for cv2.imshow, with NaN (unobserved) cells rendered as
        flat dark gray so "unknown" stays visually distinct from any real
        elevation value."""
        import cv2

        source = self.fused_elevation if use_fused else self.elevation
        if use_fused:
            source = np.where(np.isnan(self.fused_elevation), self.elevation, self.fused_elevation)
        grid = source.astype(np.float32)
        valid = ~np.isnan(grid)
        if vmin is None or vmax is None:
            if valid.any():
                vmin_, vmax_ = np.nanmin(grid), np.nanmax(grid)
            else:
                vmin_, vmax_ = 0.0, 1.0
            vmin = vmin_ if vmin is None else vmin
            vmax = vmax_ if vmax is None else vmax
        span = max(vmax - vmin, 1e-6)

        norm = np.zeros_like(grid, dtype=np.uint8)
        norm[valid] = np.clip(((grid[valid] - vmin) / span) * 255.0, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        img[~valid] = (40, 40, 40)  # dark gray = never observed (true "unknown")

        img = np.flipud(img)
        if scale > 1:
            img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                              interpolation=cv2.INTER_NEAREST)
        n_valid = int(valid.sum())
        pct = 100.0 * n_valid / valid.size
        cv2.putText(img, f"Stage 2: elevation grid_map {grid.shape[1]}x{grid.shape[0]} "
                          f"@ {self.resolution:.3f}m  observed={pct:.1f}%  "
                          f"z=[{vmin:.2f},{vmax:.2f}]m (gray=unobserved)",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return img


def colorize_depth_meters(img_raw: np.ndarray, depth_scale: float, vmin: float, vmax: float,
                           colormap: int):
    """Raw uint16 depth (sensor units) -> metric-scaled BGR image + valid mask.

    Unlike a naive `cv2.convertScaleAbs(img, alpha=<arbitrary>)`, this maps
    color using the sensor's real `depth_scale` and an explicit [vmin, vmax]
    metre range, so a given color always means the same real-world distance
    -- and "no return" (raw depth == 0) is rendered as solid black, distinct
    from any valid close-range reading (which JET would otherwise also render
    as a dark color, making holes indistinguishable from very-near surfaces).
    """
    import cv2

    depth_m = img_raw.astype(np.float32) * depth_scale
    valid = img_raw > 0
    span = max(vmax - vmin, 1e-6)
    norm = np.zeros_like(depth_m, dtype=np.uint8)
    norm[valid] = np.clip(((depth_m[valid] - vmin) / span) * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(norm, colormap)
    vis[~valid] = (0, 0, 0)  # no-return pixels: solid black, not "near" JET-blue
    return vis, depth_m, valid


def render_depth_colorbar(height: int, vmin: float, vmax: float, colormap: int,
                           width: int = 46, n_ticks: int = 5) -> np.ndarray:
    """A vertical color-to-distance legend: far (vmax) at the top, near
    (vmin) at the bottom, with metre tick labels -- so the depth colors in
    `colorize_depth_meters()`'s output have an explicit, readable scale
    instead of requiring the viewer to guess from color alone."""
    import cv2

    ramp = np.linspace(255, 0, height, dtype=np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(np.repeat(ramp, width, axis=1), colormap)

    labeled = np.zeros((height, width + 58, 3), dtype=np.uint8)
    labeled[:, :width] = bar
    for k in range(n_ticks):
        frac = k / (n_ticks - 1)          # 0 at top (vmax) .. 1 at bottom (vmin)
        y = int(frac * (height - 1))
        val = vmax - frac * (vmax - vmin)
        cv2.line(labeled, (width - 6, y), (width, y), (255, 255, 255), 1)
        cv2.putText(labeled, f"{val:.2f}m", (width + 3, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(labeled, "no", (2, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(labeled, "return", (2, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                (180, 180, 180), 1, cv2.LINE_AA)
    cv2.rectangle(labeled, (0, height - 30), (width - 1, height - 1), (255, 255, 255), 1)
    labeled[height - 28:height - 2, 1:width - 1] = (0, 0, 0)
    return labeled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.8.122")
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--topic", default="depth")
    ap.add_argument("--points-topic", default="points",
                    help="Organized point-cloud topic from g1_depth_publisher.py "
                         "(used by --visualize-points and --grid-map)")
    ap.add_argument("--frames", type=int, default=0, help="0 = run forever")
    ap.add_argument("--visualize", action="store_true",
                    help="Show the 2D colorized depth image window")
    ap.add_argument("--depth-min", type=float, default=0.2,
                    help="Depth (metres) mapped to the near end of the colormap "
                         "in --visualize. Ignored if --depth-auto-range is set.")
    ap.add_argument("--depth-max", type=float, default=4.0,
                    help="Depth (metres) mapped to the far end of the colormap "
                         "in --visualize. Ignored if --depth-auto-range is set.")
    ap.add_argument("--depth-auto-range", action="store_true",
                    help="Auto-scale the --visualize colormap to the 5th/95th "
                         "percentile of each frame's valid depth, instead of the "
                         "fixed --depth-min/--depth-max range. Makes color more "
                         "sensitive to whatever's in view, at the cost of the "
                         "color->distance mapping shifting frame to frame.")
    ap.add_argument("--visualize-points", action="store_true",

                    help="Show a live 3D point-cloud window (subscribes to --points-topic; "
                         "requires open3d)")
    ap.add_argument("--point-size", type=float, default=2.5,
                    help="Point size for the 3D point-cloud window")
    ap.add_argument("--points-range", type=float, default=2.0,
                    help="For --visualize-points: clip the cloud to a square "
                         "footprint of this size (metres) directly in front of "
                         "the camera (lateral |X| <= range/2, depth 0 <= Z <= "
                         "range, camera optical frame). Set to 0 to disable.")
    ap.add_argument("--points-view-auto-fit", action="store_true",
                    help="For --visualize-points: use Open3D's auto-fit "
                         "reset_view_point() instead of the default, which places "
                         "the virtual camera exactly at the physical camera's own "
                         "pose (eye at origin, looking along +Z)")

    ap.add_argument("--grid-map", action="store_true",
                    help="Fuse the incoming point cloud into a persistent Stage-2 "
                         "elevation grid_map (see ElevationGridMap) and show it live "
                         "(subscribes to --points-topic; requires opencv-python). No "
                         "odometry input, so the grid is anchored wherever the sensor "
                         "starts (see ElevationGridMap docstring for the caveat).")
    ap.add_argument("--grid-resolution", type=float, default=0.05,
                    help="Stage-2 grid_map cell size in metres")
    ap.add_argument("--grid-length-x", type=float, default=6.0,
                    help="Stage-2 grid_map total extent along X (metres)")
    ap.add_argument("--grid-length-y", type=float, default=6.0,
                    help="Stage-2 grid_map total extent along Y (metres)")
    ap.add_argument("--grid-scale", type=int, default=4,
                    help="Pixel scale factor for the grid_map image")
    ap.add_argument("--grid-process-noise-rate", type=float, default=1e-4,
                    help="Stage-2 variance growth RATE (m^2/s) for already-observed "
                         "cells, integrated over real elapsed time between frames")
    ap.add_argument("--grid-pose-drift-std", type=float, default=0.0,
                    help="Assumed odometry position-uncertainty growth rate "
                         "(m/sqrt(s)); adds (this)^2 * dt to the variance growth each "
                         "update, standing in for 'pose covariance as input' since no "
                         "real odometry covariance is available here")
    ap.add_argument("--grid-noise-a", type=float, default=1e-3,
                    help="Per-point measurement noise model: std-dev floor at zero "
                         "range (metres), sigma(d) = noise_a + noise_b * d^2")
    ap.add_argument("--grid-noise-b", type=float, default=2e-3,
                    help="Per-point measurement noise model: quadratic-with-range "
                         "coefficient (metres per metre^2), sigma(d) = noise_a + "
                         "noise_b * d^2 -- larger values model noisier long-range "
                         "depth returns (typical of stereo/ToF depth sensors)")
    ap.add_argument("--grid-fuse-every", type=int, default=5,
                    help="Run the Stage-2 fuse() error-ellipse spatial-smoothing pass "
                         "(see ElevationGridMap.fuse_all()) every N points-frames, "
                         "producing the actual 'published' fused_elevation/lower_bound/"
                         "upper_bound layers -- matching the real system's separate, "
                         "lower-rate raw-map-vs-fused-map update cadence. Set to 0 to "
                         "disable (view falls back to the raw per-cell layer).")
    ap.add_argument("--grid-fuse-max-cells", type=int, default=400,
                    help="Cap on how many cells fuse_all() processes per call (perf "
                         "safeguard for large/dense maps, analogous to the real "
                         "system's maxNumberOfCellsToFuse)")
    ap.add_argument("--cam-height", type=float, default=1.0,
                    help="Camera mount height (metres) used by --grid-map's "
                         "optical_to_map_frame transform")
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0,
                    help="Camera tilt in degrees (positive = tilted downward), used by "
                         "--grid-map's optical_to_map_frame transform")
    ap.add_argument("--save", help="write received frames to this .npz")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="seconds to wait for the first frame")
    args = ap.parse_args()

    cmd_str = " ".join(shlex.quote(a) for a in sys.argv)

    subscribe_points = args.visualize_points or args.grid_map

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, args.topic.encode())
    if subscribe_points:
        sock.setsockopt(zmq.SUBSCRIBE, args.points_topic.encode())
    sock.setsockopt(zmq.RCVHWM, 2)
    sock.setsockopt(zmq.CONFLATE, 0)  # CONFLATE breaks multipart; drop via HWM instead
    sock.connect(f"tcp://{args.host}:{args.port}")
    topics_str = f"{args.topic!r}" + (f", {args.points_topic!r}" if subscribe_points else "")
    print(f"[sub] connected tcp://{args.host}:{args.port} topics={topics_str}")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    vis3d = None
    pcd = None
    first_points = True
    if args.visualize_points:
        try:
            import open3d as o3d
            vis3d = o3d.visualization.Visualizer()
            vis3d.create_window("G1 D435i point cloud", width=1024, height=768)
            pcd = o3d.geometry.PointCloud()
            vis3d.add_geometry(pcd)
            opt = vis3d.get_render_option()
            opt.background_color = np.asarray([0.05, 0.05, 0.05])
            opt.point_size = args.point_size
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            vis3d.add_geometry(axes)
        except ImportError:
            print("[--visualize-points requires open3d: pip install open3d]")
            vis3d = None

    grid_map = None
    grid_map_view = False
    n_grid_frames = 0
    if args.grid_map:
        grid_map = ElevationGridMap(
            resolution=args.grid_resolution,
            length_x=args.grid_length_x,
            length_y=args.grid_length_y,
            process_noise_rate=args.grid_process_noise_rate,
            pose_drift_std=args.grid_pose_drift_std,
            noise_a=args.grid_noise_a,
            noise_b=args.grid_noise_b,
        )
        try:
            import cv2  # noqa: F401  (import check only)
            grid_map_view = True
        except ImportError:
            print("[--grid-map visualization requires opencv-python: pip install opencv-python]")
        print(f"[sub] Stage-2 grid_map: {grid_map.size_x}x{grid_map.size_y} cells @ "
              f"{args.grid_resolution:.3f}m => {args.grid_length_x:.1f}x{args.grid_length_y:.1f}m extent")

    depth_topic_b = args.topic.encode()
    points_topic_b = args.points_topic.encode()

    saved = []
    last_hdr = None
    n = 0
    t0 = None
    t_stat = time.time()
    fps_count = 0
    first_wait = args.timeout

    while True:
        if not poller.poll(first_wait * 1000):
            print(f"[sub] no frame within {first_wait:.0f}s — is the publisher running "
                  f"on the robot, and is port {args.port} reachable?")
            break
        first_wait = 5.0

        topic_recv, msg = sock.recv_multipart()
        hdr, arr = decode(msg)

        if topic_recv == points_topic_b:
            pts_opt = organized_points_to_xyz(arr)

            # Update the 3D raw-cloud window and move on -- doesn't count
            # toward the depth-frame fps/latency/save bookkeeping below.
            if vis3d is not None:
                try:
                    import open3d as o3d
                    pts = pts_opt.astype(np.float64)
                    pts = filter_points_footprint(pts, args.points_range)
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    if pts.shape[0] > 0:
                        z = pts[:, 2]
                        z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)
                        colors = np.stack(
                            [z_norm, 1.0 - z_norm, np.full_like(z_norm, 0.6)], axis=1)
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                    vis3d.update_geometry(pcd)
                    if first_points and pts.shape[0] > 0:
                        if args.points_view_auto_fit:
                            vis3d.reset_view_point(True)
                        else:
                            try:
                                set_open3d_view_to_camera(vis3d, hdr["intrinsics"])
                            except Exception as e:
                                print(f"  [camera-pose view failed ({e}), "
                                      f"falling back to auto-fit]")
                                vis3d.reset_view_point(True)
                        first_points = False
                    vis3d.poll_events()
                    vis3d.update_renderer()
                except Exception as e:
                    print(f"  [points view update failed] {e}")

            if grid_map is not None:
                pts_map = optical_to_map_frame(
                    pts_opt, cam_pitch_deg=args.cam_pitch_deg, cam_offset=(0.0, 0.0, args.cam_height))
                # Item 1's input: per-point range from the sensor, measured
                # in the (pre-transform) camera optical frame.
                point_ranges = np.linalg.norm(pts_opt, axis=1) if pts_opt.shape[0] > 0 \
                    else np.zeros((0,), dtype=np.float32)
                # No real odometry here, so pose_delta stays (0, 0) -- the
                # scrolling/drift machinery (items 2 & 4) is fully wired up
                # and ready to use the moment a real pose source is plugged
                # in (see ElevationGridMap.update()'s docstring).
                grid_map.update(pts_map, point_ranges, stamp=time.time())

                n_grid_frames += 1
                if args.grid_fuse_every > 0 and n_grid_frames % args.grid_fuse_every == 0:
                    t_fuse0 = time.time()
                    grid_map.fuse_all(max_cells_to_fuse=args.grid_fuse_max_cells)
                    if n_grid_frames % (args.grid_fuse_every * 4) == 0:
                        print(f"  [grid_map] fuse_all() took {(time.time() - t_fuse0) * 1e3:.1f}ms")

                if grid_map_view:
                    try:
                        import cv2
                        img_gm = grid_map.elevation_image(
                            scale=args.grid_scale, use_fused=(args.grid_fuse_every > 0))
                        cv2.imshow("Stage 2: elevation grid_map", img_gm)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                    except Exception as e:
                        print(f"  [grid_map view update failed] {e}")
            continue

        img = arr
        last_hdr = hdr
        now = time.time()
        if t0 is None:
            t0 = now
            k = hdr["intrinsics"]
            print(f"[sub] first frame: {img.shape} {img.dtype} "
                  f"depth_scale={hdr['depth_scale']:.6f} m/unit")
            print(f"[sub] intrinsics: fx={k['fx']:.2f} fy={k['fy']:.2f} "
                  f"ppx={k['ppx']:.2f} ppy={k['ppy']:.2f} model={k['model']}")

        n += 1
        fps_count += 1
        latency_ms = (now - hdr["stamp_host"]) * 1e3

        if args.save is not None:
            saved.append(img.copy())

        if now - t_stat >= 2.0:
            valid = float((img > 0).mean()) * 100.0
            d = img[img > 0]
            rng = (f"{d.min() * hdr['depth_scale']:.2f}-"
                   f"{d.max() * hdr['depth_scale']:.2f} m" if d.size else "no returns")
            print(f"[sub] {fps_count / (now - t_stat):5.1f} fps  "
                  f"latency {latency_ms:6.1f} ms  valid {valid:5.1f}%  {rng}")
            fps_count = 0
            t_stat = now

        if args.visualize:
            import cv2
            if args.depth_auto_range:
                d_valid = img[img > 0]
                if d_valid.size:
                    d_valid_m = d_valid.astype(np.float32) * hdr["depth_scale"]
                    vmin, vmax = np.percentile(d_valid_m, [5, 95])
                    vmax = max(vmax, vmin + 1e-3)
                else:
                    vmin, vmax = args.depth_min, args.depth_max
            else:
                vmin, vmax = args.depth_min, args.depth_max

            vis, depth_m, valid_mask = colorize_depth_meters(
                img, hdr["depth_scale"], vmin, vmax, cv2.COLORMAP_JET)

            cv2.putText(vis, f"seq={hdr['seq']} lat={latency_ms:.0f}ms",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if valid_mask.any():
                cv2.putText(vis, f"frame range: {depth_m[valid_mask].min():.2f}-"
                                  f"{depth_m[valid_mask].max():.2f}m  "
                                  f"colormap range: {vmin:.2f}-{vmax:.2f}m"
                                  f"{' (auto)' if args.depth_auto_range else ''}",
                            (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, cmd_str,
                        (8, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 255), 1, cv2.LINE_AA)

            colorbar = render_depth_colorbar(vis.shape[0], vmin, vmax, cv2.COLORMAP_JET)
            vis = np.concatenate([vis, colorbar], axis=1)

            cv2.imshow("G1 D435i depth", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        if args.frames and n >= args.frames:
            break

    if t0 is not None:
        print(f"[sub] received {n} frames in {time.time() - t0:.1f}s")

    if args.save and saved and last_hdr is not None:
        np.savez_compressed(
            args.save,
            depth=np.stack(saved),
            depth_scale=last_hdr["depth_scale"],
            intrinsics=json.dumps(last_hdr["intrinsics"]),
        )
        print(f"[sub] wrote {len(saved)} frames -> {args.save}")

    if vis3d is not None:
        vis3d.destroy_window()
    if grid_map_view:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
    sock.close(linger=0)


if __name__ == "__main__":
    raise SystemExit(main())
