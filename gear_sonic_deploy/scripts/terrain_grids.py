"""terrain_grids.py - sample the fused elevation map on robot-centred grids (numpy only, Python 3.8).

Used by ../perception/terrain_publisher.py for the `terrain` ZMQ topic:

  height_grid    50 x 50 at 4 cm   terrain_z - torso_z   (generic, policy-agnostic)
  torso_grid_21  21 x 21 at 10 cm  torso_z - surface_z   (the heightmap flow policy's map, branch
                                                          mmurray/fm of humanoid-foundation-model:
                                                          gear_sonic/flow/*_collect_callback.py)

Both are centred on torso_link and yaw-aligned with the torso heading, laid out [y][x] (row = y,
col = x, index 0 = most negative, IsaacLab GridPattern "xy" order), and take the map cell nearest
each cell centre - the same point sample the sim's analytic map takes. NaN = unseen.
"""

import numpy as np

TORSO_GRID_N = 21
TORSO_GRID_RES = 0.1


def grid_local(n, res):
    """Cell centres of an n x n grid, flattened [y][x] -> (n*n, 2) in the grid frame."""
    c = (np.arange(n) - (n - 1) / 2.0) * res
    gx, gy = np.meshgrid(c, c, indexing="xy")  # gx[y][x] = c[x]
    return np.stack([gx.ravel(), gy.ravel()], 1)


def sample_map(E, res, cx, cy, pos_xy, yaw, local):
    """Map heights (absolute z) at grid cells rotated by yaw about pos_xy; NaN off the map or unseen.

    E: elevation_mapping_cupy layer (grid_map convention: rows along -x, cols along -y), centred
    at (cx, cy) with resolution res. local: (N, 2) cell centres in the yaw-aligned grid frame.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    xy = local @ np.array([[c, s], [-s, c]]) + np.asarray(pos_xy)[:2]
    i = np.floor(E.shape[0] / 2.0 - (xy[:, 0] - cx) / res).astype(np.int64)   # rows along -x
    j = np.floor(E.shape[1] / 2.0 - (xy[:, 1] - cy) / res).astype(np.int64)   # cols along -y
    inb = (i >= 0) & (i < E.shape[0]) & (j >= 0) & (j < E.shape[1])
    z = np.full(len(xy), np.nan, dtype=np.float32)
    z[inb] = E[i[inb], j[inb]]
    return z


_TORSO_LOCAL = grid_local(TORSO_GRID_N, TORSO_GRID_RES)


def torso_grid(E, res, cx, cy, torso_pos, yaw):
    """(torso_z - surface_z [21, 21] float32 with NaN where unseen, valid [21, 21] bool)."""
    z = sample_map(E, res, cx, cy, torso_pos, yaw, _TORSO_LOCAL)
    h = (np.float32(torso_pos[2]) - z).reshape(TORSO_GRID_N, TORSO_GRID_N)
    return h, np.isfinite(h)
