"""pose_guard.py - reject an odometry pose that cannot be right before a policy sees a map built on it.

The terrain map reaches a policy re-centred on the torso pose from odometry (DLIO). TerrainInput in
the deploy runner already drops a pose that is too OLD; a pose that is fresh but wrong passed: on the
gantry (2026-10-20) DLIO diverged twice under CPU load -- once to 59 m from its start in seconds,
once to 3.5 m and back -- and the map, re-centred on those poses and fused from points placed with
them, held phantom surfaces up to ~60 cm around a robot standing on a flat floor.

`PoseGuard` flags a pose as implausible when, for the robot it serves (a G1 walking at <= ~1.5 m/s):

  speed        the torso moved faster than `max_speed` (2 m/s) over the last `window` (0.1 s)
  climb        or faster than `max_climb` (1 m/s) vertically
  turn         or turned faster than `max_turn` (6 rad/s) about any axis
  height       the torso sits outside `height_range` (0.4-1.3 m) above the floor the map shows around
               it (median of the observed cells 0.3-0.8 m away; checked only with >= 20 such cells)

and keeps rejecting it for `hold` (2 s) after the last violation, so a pose that has just come back
from a jump is not trusted straight away. `state(t)` -> (ok, reasons); `new_episode` is True once per
rejection episode (the publisher clears the mapper then, so the phantom surfaces do not linger).
"""
import collections

import numpy as np

FLOOR_RING = (0.3, 0.8)
MIN_FLOOR_CELLS = 20


def _rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PoseGuard:
    def __init__(self, max_speed=2.0, max_climb=1.0, max_turn=6.0, height_range=(0.4, 1.3),
                 window=0.1, hold=2.0):
        self.max_speed, self.max_climb, self.max_turn = max_speed, max_climb, max_turn
        self.height_range, self.window, self.hold = height_range, window, hold
        self.hist = collections.deque()     # (t, pos, R)
        self.last_bad = None
        self.reasons = []
        self.in_episode = False
        self.new_episode = False

    def odom(self, t, pos, quat_xyzw):
        """Feed every odometry pose (stamp t, s)."""
        pos = np.asarray(pos, float)
        R = _rot(quat_xyzw)
        self.hist.append((t, pos, R))
        while len(self.hist) > 2 and t - self.hist[1][0] >= self.window:
            self.hist.popleft()
        t0, p0, R0 = self.hist[0]
        dt = t - t0
        if dt >= 0.5 * self.window:
            v = (pos - p0) / dt
            ang = np.arccos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1.0, 1.0)) / dt
            bad = []
            if np.linalg.norm(v[:2]) > self.max_speed:
                bad.append(f"speed {np.linalg.norm(v[:2]):.1f} m/s")
            if abs(v[2]) > self.max_climb:
                bad.append(f"climb {v[2]:+.1f} m/s")
            if ang > self.max_turn:
                bad.append(f"turn {ang:.1f} rad/s")
            if bad:
                self._flag(t, bad)

    def floor(self, t, height_grid, valid, cell_xy):
        """Feed a sampled map: heights terrain_z - torso_z at cell_xy (N, 2) around the torso."""
        h = np.asarray(height_grid, float).ravel()
        v = np.asarray(valid, bool).ravel() & np.isfinite(h)
        r = np.hypot(cell_xy[:, 0], cell_xy[:, 1])
        ring = v & (r >= FLOOR_RING[0]) & (r <= FLOOR_RING[1])
        if ring.sum() < MIN_FLOOR_CELLS:
            return None
        above = -float(np.median(h[ring]))
        if not self.height_range[0] <= above <= self.height_range[1]:
            self._flag(t, [f"torso {above:.2f} m above the mapped floor"])
        return above

    def _flag(self, t, reasons):
        self.last_bad, self.reasons = t, reasons
        if not self.in_episode:
            self.in_episode, self.new_episode = True, True

    def state(self, t):
        """(ok, reasons) at time t; ends the rejection episode `hold` s after the last violation."""
        if self.last_bad is not None and t - self.last_bad <= self.hold:
            return False, self.reasons
        self.in_episode = False
        return True, []

    def take_new_episode(self):
        out, self.new_episode = self.new_episode, False
        return out
