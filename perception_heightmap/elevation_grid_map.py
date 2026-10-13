#!/usr/bin/env python3
"""elevation_grid_map.py
========================
Stage 2 of the real pipeline: a persistent, **fixed-frame** fused elevation
map, mirroring `elevation_mapping`'s internal `grid_map_msgs/GridMap`
(https://github.com/ANYbotics/elevation_mapping /
ArthurAllshire/elevation_mapping_humanoid). This sits BETWEEN:

    Stage 1 (organized point cloud, already implemented in
             `view_depth_heightmap.points_message_to_optical_xyz` /
             `perception_heightmap/g1_depth_subscriber.organized_points_to_xyz`)
and
    Stage 3 (`.../elevation_map_fused_visualization/elevation_cloud`, a
             PointCloud2 *derived from* this grid map by flattening to XYZ
             and dropping the variance/NaN semantics -- see
             `ElevationGridMap.to_elevation_cloud()` below, and Stages 4-6,
             already implemented in `view_depth_heightmap.py`'s
             `interpolate_height_idw` / `build_query_grid` /
             `optical_to_body_frame`, which consume exactly that lossy
             flattened cloud).

Unlike everything built so far in this repo (which processed one point-cloud
frame at a time, egocentrically, with no persistence), Stage 2 is
structurally different:

  - It's a **fixed-size, fixed-resolution grid anchored in a fixed frame**
    (`odom_corrected` in the real system), NOT egocentric/robot-relative --
    the egocentric 11x11 query grid only appears later, at Stage 5, built by
    yaw-rotating+translating a *local* query grid into this fixed frame's
    coordinates and querying it (`getGlobalQueryPoints` in
    `videomimic_inference_real.cpp`).
  - It carries **multiple named layers** over the same grid: `elevation`,
    `variance`, `horizontal_variance_x`, `horizontal_variance_y`, `time`
    (this module also tracks `n_obs`, a diagnostic not in the real system).
  - Cells can be **NaN = explicitly "never observed"** -- a real epistemic
    "unknown" state, which is exactly what gets lost when Stage 3 flattens
    this into a plain XYZ point cloud (dropping variance/NaN and letting
    Stage 4-6's `KNN_DEFAULT_HEIGHT_OFFSET` fallback silently stand in for
    "unknown" as "assume flat ground" instead).

IMPORTANT limitation vs. the real system: `elevation_mapping` keeps this grid
registered in the world/odom frame using live TF (robot odometry), so the
grid stays geometrically correct as the robot walks around, and old cells
behind/beside the robot persist correctly. This module has no odometry/TF
input, so by default the grid is simply anchored wherever the sensor happens
to be pointing when the node starts (i.e. assumes a stationary or slowly
drifting sensor between frames) -- call `shift(dx, dy)` yourself before
`update()` if you do have a pose delta, to approximate the real
re-registration via an integer-cell roll (sub-cell motion is not modeled).
"""

import numpy as np


class ElevationGridMap:
    """A `grid_map_msgs/GridMap`-equivalent: named 2D layers over a fixed,
    world-frame-anchored grid, with a per-cell Kalman-filtered `elevation` +
    `variance`, and NaN cells for "never observed"."""

    def __init__(
        self,
        resolution: float = 0.05,
        length_x: float = 6.0,
        length_y: float = 6.0,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-3,
        max_effective_count: int = 8,
        min_variance: float = 1e-5,
        max_variance: float = 1.0,
    ):
        """
        resolution: metres per cell (real system's elevation_mapping default
            is much finer than the 0.1m Stage-5 query grid, e.g. 0.02-0.05m).
        length_x, length_y: total map extent in metres (must be large enough
            to cover wherever the Stage-5 query grid will be placed).
        process_noise: per-update() variance growth for already-observed
            cells (models the map's confidence slowly decaying if not
            re-observed, same rationale as this repo's earlier
            `ElevationMapFuser`).
        measurement_noise: assumed per-point sensor noise variance (metres^2).
        max_effective_count: caps how many points-per-cell-per-frame can
            shrink the effective measurement variance, so a single dense
            frame can never fully overwrite history (see the note in
            `view_lidar_client.ElevationMapFuser` about why this matters).
        min_variance / max_variance: floor/ceiling for the variance layer,
            once a cell has been observed at least once.
        """
        self.resolution = resolution
        self.length_x = length_x
        self.length_y = length_y
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_effective_count = max_effective_count
        self.min_variance = min_variance
        self.max_variance = max_variance

        self.size_x = int(round(length_x / resolution))
        self.size_y = int(round(length_y / resolution))
        # Map "position": the (x, y) of the grid center in the fixed map
        # frame. Real elevation_mapping moves this as the robot walks
        # (tracking base position) so the grid stays centered near the
        # robot without needing to be huge; here it only changes if you call
        # `shift()` yourself.
        self.position = np.array([0.0, 0.0], dtype=np.float64)

        shape = (self.size_y, self.size_x)  # row=Y, col=X, matches other scripts here
        self.elevation = np.full(shape, np.nan, dtype=np.float64)
        self.variance = np.full(shape, np.nan, dtype=np.float64)
        self.horizontal_variance_x = np.full(shape, np.nan, dtype=np.float64)
        self.horizontal_variance_y = np.full(shape, np.nan, dtype=np.float64)
        self.time = np.full(shape, np.nan, dtype=np.float64)
        self.n_obs = np.zeros(shape, dtype=np.int64)  # diagnostic only

        self._recompute_cell_centers()

    def _recompute_cell_centers(self):
        half_x = self.size_x / 2.0 * self.resolution
        half_y = self.size_y / 2.0 * self.resolution
        xs = self.position[0] - half_x + (np.arange(self.size_x) + 0.5) * self.resolution
        ys = self.position[1] - half_y + (np.arange(self.size_y) + 0.5) * self.resolution
        grid_x, grid_y = np.meshgrid(xs, ys)
        self._cell_centers_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        self._xs, self._ys = xs, ys

    def reset(self):
        self.elevation.fill(np.nan)
        self.variance.fill(np.nan)
        self.horizontal_variance_x.fill(np.nan)
        self.horizontal_variance_y.fill(np.nan)
        self.time.fill(np.nan)
        self.n_obs.fill(0)

    def shift(self, dx: float, dy: float):
        """Approximate re-registering the map after the sensor/robot has
        moved (dx, dy) metres in the map frame, by rolling the grid arrays by
        the nearest integer number of cells (sub-cell motion is ignored) and
        marking the newly-exposed border as unobserved (NaN) again -- this is
        the same coarse stand-in `ElevationMapFuser.shift()` uses, but here
        newly-exposed cells become truly NaN/unknown rather than just
        high-variance, since that's the whole point of this stage."""
        shift_cells_x = int(round(dx / self.resolution))
        shift_cells_y = int(round(dy / self.resolution))
        if shift_cells_x == 0 and shift_cells_y == 0:
            return

        for arr in (self.elevation, self.variance, self.horizontal_variance_x,
                    self.horizontal_variance_y, self.time):
            arr[:] = np.roll(arr, (-shift_cells_y, -shift_cells_x), axis=(0, 1))
        self.n_obs[:] = np.roll(self.n_obs, (-shift_cells_y, -shift_cells_x), axis=(0, 1))

        def _wipe(y_slice, x_slice):
            self.elevation[y_slice, x_slice] = np.nan
            self.variance[y_slice, x_slice] = np.nan
            self.horizontal_variance_x[y_slice, x_slice] = np.nan
            self.horizontal_variance_y[y_slice, x_slice] = np.nan
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
        self._recompute_cell_centers()

    def update(self, points_xyz_map_frame: np.ndarray, stamp: float):
        """Fuse one new frame's points (already expressed in this map's
        fixed frame -- e.g. via `optical_to_body_frame` plus your own
        odometry, if you have any) into the persistent grid.

        Cells with no points this frame are left exactly as they were
        (matching the real callback's "empty message -> keep previous
        heights" behavior, just at the per-cell level instead of whole-map).
        """
        # Predict step: every ALREADY-OBSERVED cell's variance grows a little
        # (unobserved/NaN cells are untouched -- there's nothing to decay).
        observed = ~np.isnan(self.variance)
        self.variance[observed] += self.process_noise
        self.variance[observed] = np.minimum(self.variance[observed], self.max_variance)

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
        rel_x, rel_y, z = rel_x[in_bounds], rel_y[in_bounds], z[in_bounds]
        if rel_x.shape[0] == 0:
            return

        i = np.clip((rel_x / self.resolution).astype(np.int64), 0, self.size_x - 1)
        j = np.clip((rel_y / self.resolution).astype(np.int64), 0, self.size_y - 1)
        flat_idx = j * self.size_x + i

        n_cells = self.size_x * self.size_y
        sums = np.zeros(n_cells, dtype=np.float64)
        sums_sq = np.zeros(n_cells, dtype=np.float64)  # for horizontal variance proxy
        counts = np.zeros(n_cells, dtype=np.int64)
        np.add.at(sums, flat_idx, z)
        np.add.at(counts, flat_idx, 1)

        touched = counts > 0
        if not touched.any():
            return
        batch_mean = sums[touched] / counts[touched]
        effective_counts = np.minimum(counts[touched], self.max_effective_count)
        batch_meas_var = self.measurement_noise / effective_counts

        # A crude proxy for horizontal_variance_{x,y}: how spread out the
        # points landing in each cell are in X/Y this frame (the real system
        # derives this from the sensor's actual noise model + ray geometry;
        # this is a much simpler stand-in, flagged as a known simplification).
        sums_x = np.zeros(n_cells, dtype=np.float64)
        sums_y = np.zeros(n_cells, dtype=np.float64)
        sums_x2 = np.zeros(n_cells, dtype=np.float64)
        sums_y2 = np.zeros(n_cells, dtype=np.float64)
        np.add.at(sums_x, flat_idx, x[in_bounds])
        np.add.at(sums_y, flat_idx, y[in_bounds])
        np.add.at(sums_x2, flat_idx, x[in_bounds] ** 2)
        np.add.at(sums_y2, flat_idx, y[in_bounds] ** 2)
        mean_x = sums_x[touched] / counts[touched]
        mean_y = sums_y[touched] / counts[touched]
        var_x = np.maximum(sums_x2[touched] / counts[touched] - mean_x ** 2, 0.0)
        var_y = np.maximum(sums_y2[touched] / counts[touched] - mean_y ** 2, 0.0)

        flat_elevation = self.elevation.reshape(-1)
        flat_variance = self.variance.reshape(-1)
        flat_hvx = self.horizontal_variance_x.reshape(-1)
        flat_hvy = self.horizontal_variance_y.reshape(-1)
        flat_time = self.time.reshape(-1)
        flat_n_obs = self.n_obs.reshape(-1)

        prior_e = flat_elevation[touched]
        prior_v = flat_variance[touched]
        never_seen = np.isnan(prior_e)

        # First observation: just initialize directly (no prior to blend
        # with yet).
        new_e = np.where(never_seen, batch_mean, prior_e)
        new_v = np.where(never_seen, batch_meas_var, prior_v)

        # Cells that already have a prior: scalar Kalman measurement update.
        has_prior = ~never_seen
        if has_prior.any():
            kalman_gain = prior_v[has_prior] / (prior_v[has_prior] + batch_meas_var[has_prior])
            new_e[has_prior] = prior_e[has_prior] + kalman_gain * (
                batch_mean[has_prior] - prior_e[has_prior])
            new_v[has_prior] = (1.0 - kalman_gain) * prior_v[has_prior]

        new_v = np.clip(new_v, self.min_variance, self.max_variance)

        flat_elevation[touched] = new_e
        flat_variance[touched] = new_v
        flat_hvx[touched] = var_x
        flat_hvy[touched] = var_y
        flat_time[touched] = stamp
        flat_n_obs[touched] += counts[touched]

    def get_layer(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def to_elevation_cloud(self) -> np.ndarray:
        """Stage 3 equivalent: flatten the fused grid to an (N, 3) XYZ point
        cloud, dropping variance/time/NaN semantics -- exactly the lossy
        step the real `elevation_map_fused_visualization` node performs
        before publishing `.../elevation_cloud`. Cells that are still NaN
        (never observed) are simply omitted (not represented at all), which
        is the direct cause of Stage 4-6's `KNN_DEFAULT_HEIGHT_OFFSET`
        fallback being unable to distinguish "confirmed flat ground" from
        "no idea, never looked here"."""
        valid = ~np.isnan(self.elevation)
        if not valid.any():
            return np.zeros((0, 3), dtype=np.float32)
        xy = self._cell_centers_xy[valid.reshape(-1)]
        z = self.elevation[valid]
        return np.concatenate([xy, z[:, None]], axis=1).astype(np.float32)
