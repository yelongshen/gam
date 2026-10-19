#!/usr/bin/env python3
"""height_map.py
================
Build a fixed-size 2D height map from a raw LiDAR point cloud.

This is a generic, dependency-light (numpy only) implementation you can
use with any Nx3 array of (x, y, z) points -- e.g. the output of
`decode_xyz()` in `view_lidar.py` / `lidar_zmq_publisher.py`.

Algorithm
---------
1. Define a square grid of `grid_size x grid_size` cells (e.g. 11x11),
   centered at (center_x, center_y), covering [-half_extent, +half_extent]
   in both X and Y (so `cell_size = 2 * half_extent / grid_size`).
2. Bin every point into a cell based on its (x, y) position.
3. Aggregate the Z values that fall in each cell using `agg` ("max" for a
   terrain/obstacle height map -- useful for foot placement and collision
   checking; "min" for a floor/ground map; "mean"/"median" for a smoothed
   surface). With noisy sources (the D435i depth cloud) "max" picks the
   upward noise tail in every cell -- prefer "median" there.
4. Cells with no points are filled with `empty_value` (default: NaN).

Coordinate convention: z must be "up" -- the input points must already be in
a gravity-aligned frame. Do NOT pass raw `livox_frame` points: the G1's
MID-360 is mounted upside down, so the floor sits at z ~ +1.3 m there and
"max" returns the floor instead of obstacles, and ~30% of its points are
(0, 0, 0) no-returns. Use `g1_frames.lidar_to_level()` /
`g1_frames.depth_to_level()` first, which clean the points and express them
in the gravity-levelled torso frame (still robot-relative, not world).

Usage (standalone self-test with synthetic data):
    python3 gear_sonic_deploy/scripts/height_map.py
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class HeightMapConfig:
    grid_size: int = 11          # NxN cells (e.g. 11x11)
    half_extent: float = 1.0     # meters; grid covers [-half_extent, +half_extent]
    center_x: float = 0.0        # meters, grid center X (in the point cloud's frame)
    center_y: float = 0.0        # meters, grid center Y
    agg: str = "max"             # "max" | "min" | "mean" | "median"
    empty_value: float = float("nan")
    z_min_filter: float = None   # drop points with z < this (e.g. exclude floor)
    z_max_filter: float = None   # drop points with z > this (e.g. exclude ceiling)

    @property
    def cell_size(self) -> float:
        return (2.0 * self.half_extent) / self.grid_size


def build_height_map(points: np.ndarray, config: HeightMapConfig = None) -> np.ndarray:
    """Bin an Nx3 point cloud into a grid_size x grid_size height map.

    Args:
        points: (N, 3) array of (x, y, z) points, in the same frame the
            config's center_x/center_y are expressed in.
        config: HeightMapConfig (uses defaults if omitted).

    Returns:
        (grid_size, grid_size) float32 array. Row 0 = most negative Y,
        Col 0 = most negative X (standard image/matrix indexing with Y as
        the first axis). Empty cells contain config.empty_value.
    """
    config = config or HeightMapConfig()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got shape {points.shape}")

    grid = np.full((config.grid_size, config.grid_size), config.empty_value, dtype=np.float32)
    if points.shape[0] == 0:
        return grid

    x = points[:, 0] - config.center_x
    y = points[:, 1] - config.center_y
    z = points[:, 2]

    mask = (x >= -config.half_extent) & (x < config.half_extent) & \
           (y >= -config.half_extent) & (y < config.half_extent)
    if config.z_min_filter is not None:
        mask &= (z >= config.z_min_filter)
    if config.z_max_filter is not None:
        mask &= (z <= config.z_max_filter)

    x, y, z = x[mask], y[mask], z[mask]
    if x.shape[0] == 0:
        return grid

    cell_size = config.cell_size
    cols = np.clip(((x + config.half_extent) / cell_size).astype(np.int64), 0, config.grid_size - 1)
    rows = np.clip(((y + config.half_extent) / cell_size).astype(np.int64), 0, config.grid_size - 1)

    # Group z-values per (row, col) cell.
    flat_idx = rows * config.grid_size + cols
    n_cells = config.grid_size * config.grid_size

    if config.agg == "max":
        flat = np.full(n_cells, -np.inf, dtype=np.float64)
        np.maximum.at(flat, flat_idx, z)
        flat[flat == -np.inf] = config.empty_value
    elif config.agg == "min":
        flat = np.full(n_cells, np.inf, dtype=np.float64)
        np.minimum.at(flat, flat_idx, z)
        flat[flat == np.inf] = config.empty_value
    elif config.agg == "mean":
        sums = np.zeros(n_cells, dtype=np.float64)
        counts = np.zeros(n_cells, dtype=np.int64)
        np.add.at(sums, flat_idx, z)
        np.add.at(counts, flat_idx, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            flat = sums / counts
        flat[counts == 0] = config.empty_value
    elif config.agg == "median":
        # Sort by (cell, z), then take the middle element of each cell's run.
        order = np.lexsort((z, flat_idx))
        idx_sorted, z_sorted = flat_idx[order], z[order]
        cells, starts, counts = np.unique(idx_sorted, return_index=True, return_counts=True)
        lo = z_sorted[starts + (counts - 1) // 2]
        hi = z_sorted[starts + counts // 2]
        flat = np.full(n_cells, config.empty_value, dtype=np.float64)
        flat[cells] = 0.5 * (lo + hi)
    else:
        raise ValueError(f"Unknown agg mode: {config.agg!r} (expected 'max'/'min'/'mean'/'median')")

    grid = flat.reshape(config.grid_size, config.grid_size).astype(np.float32)
    return grid


def print_height_map(grid: np.ndarray, fmt: str = "{:6.2f}"):
    """Pretty-print a height map grid to the terminal."""
    for row in grid[::-1]:  # flip so +Y is printed at the top
        cells = ["  nan " if np.isnan(v) else fmt.format(v) for v in row]
        print(" ".join(cells))


if __name__ == "__main__":
    # Self-test with synthetic data: a flat floor at z=0 plus a single
    # "obstacle" point at (0.3, 0.3, 0.5).
    rng = np.random.default_rng(0)
    floor_pts = np.column_stack([
        rng.uniform(-1.0, 1.0, 2000),
        rng.uniform(-1.0, 1.0, 2000),
        np.zeros(2000),
    ])
    obstacle_pts = np.column_stack([
        np.full(50, 0.3),
        np.full(50, 0.3),
        np.full(50, 0.5),
    ])
    points = np.vstack([floor_pts, obstacle_pts])

    config = HeightMapConfig(grid_size=11, half_extent=1.0, agg="max")
    grid = build_height_map(points, config)

    print(f"11x11 height map (cell_size={config.cell_size:.3f} m, agg='max'):")
    print_height_map(grid)
