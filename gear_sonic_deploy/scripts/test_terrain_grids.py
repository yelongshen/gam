"""Tests for terrain_grids.py (numpy only): python3 -m pytest test_terrain_grids.py"""
import numpy as np

import terrain_grids as tg

RES, N = 0.04, 200  # an elevation_mapping_cupy-sized map: 8 m at 4 cm


def make_map(cx=0.3, cy=-0.2):
    """grid_map convention: rows along -x, cols along -y; cell (i, j) centre computed like the mapper."""
    E = np.zeros((N, N), np.float32)
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    x = cx + (N / 2.0 - i - 0.5) * RES
    y = cy + (N / 2.0 - j - 0.5) * RES
    box = (np.abs(x - 1.0) < 0.2) & (np.abs(y - 0.5) < 0.2)  # 40 cm box at (1.0, 0.5), 0.45 m tall
    E[np.broadcast_to(box, E.shape)] = 0.45
    E[np.broadcast_to((np.abs(x - 0.3) < 0.15) & (np.abs(y + 0.6) < 0.15), E.shape)] = np.nan  # unseen patch
    return E, cx, cy


def test_layout_and_box_in_the_right_cell():
    E, cx, cy = make_map()
    torso = np.array([0.5, 0.5, 1.0])
    yaw = np.pi / 2  # facing +y: the box (0.5 m along world +x from the torso) is to the robot's right (-y)
    h, valid = tg.torso_grid(E, RES, cx, cy, torso, yaw)
    assert h.shape == (21, 21) and h.dtype == np.float32
    # [y][x] with index 10 at the torso: robot-frame (x=0, y=-0.5) -> row 10 - 5, col 10
    assert abs(h[5, 10] - (1.0 - 0.45)) < 1e-6
    assert abs(h[10, 10] - 1.0) < 1e-6  # floor under the torso
    assert abs(h[10, 15] - 1.0) < 1e-6  # 0.5 m ahead: floor


def test_unseen_and_off_map_cells_are_invalid():
    E, cx, cy = make_map()
    h, valid = tg.torso_grid(E, RES, cx, cy, np.array([0.3, -0.6, 1.0]), 0.0)
    assert not valid[10, 10] and np.isnan(h[10, 10])  # torso over the unseen patch
    assert valid[0, 0]
    h, valid = tg.torso_grid(E, RES, cx, cy, np.array([4.0, 0.0, 1.0]), 0.0)  # map edge at x = 4.3
    assert valid[10, 10 - 3] and not valid[10, 20]  # behind the robot on the map, ahead off it


def test_sample_map_matches_the_publishers_previous_inline_sampling():
    E, cx, cy = make_map()
    local = tg.grid_local(50, 0.04)
    rng = np.random.default_rng(0)
    for _ in range(20):
        pos, yaw = rng.uniform(-2, 2, 3), rng.uniform(-np.pi, np.pi)
        cyaw, syaw = np.cos(yaw), np.sin(yaw)  # the code terrain_publisher.py had inline
        xy = local @ np.array([[cyaw, syaw], [-syaw, cyaw]]) + pos[:2]
        i = np.floor(E.shape[0] / 2.0 - (xy[:, 0] - cx) / RES).astype(np.int64)
        j = np.floor(E.shape[1] / 2.0 - (xy[:, 1] - cy) / RES).astype(np.int64)
        inb = (i >= 0) & (i < E.shape[0]) & (j >= 0) & (j < E.shape[1])
        old = np.full(len(xy), np.nan, dtype=np.float32)
        old[inb] = E[i[inb], j[inb]] - pos[2]
        new = tg.sample_map(E, RES, cx, cy, pos, yaw, local) - np.float32(pos[2])
        np.testing.assert_allclose(new, old, atol=1e-6, equal_nan=True)
