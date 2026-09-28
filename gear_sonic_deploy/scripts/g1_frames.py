#!/usr/bin/env python3
"""g1_frames.py
===============
Single source of truth for the G1 head-sensor extrinsics, plus the helpers
needed to turn raw sensor points into a gravity-levelled, torso-centred
frame suitable for height mapping. numpy only (runs on the robot's
Python 3.8 / numpy 1.17 as well as on the workstation).

Frames
------
torso    : G1 `torso_link` (x forward, y left, z up).
livox    : raw MID-360 points as published on `rt/utlidar/cloud_livox_mid360`
           (`frame_id="livox_frame"`). The sensor is mounted UPSIDE DOWN:
           floor points have z ~ +1.3 m and the built-in IMU reads gravity
           along -z. Do NOT bin these points directly into a height map.
optical  : D435i depth optical frame, i.e. the frame of the XYZ points from
           `rs.pointcloud()` in g1_depth_publisher.py (z forward, x right,
           y down).
level    : torso frame rotated so +z is exactly opposite gravity (roll/pitch
           removed, yaw kept), origin unchanged.

Provenance of the constants (verified 2026-09-28 on the gantry, see
~/g1_debug on the workstation):
  * T_TORSO_LIVOX: rpy=(0, 3.101, 3.1415), xyz from the MID-360 mount.
    Confirmed against the MID-360's own IMU (gravity along livox -z) and a
    floor-plane fit, and front/back/left/right confirmed by cross-checking
    objects against the forward-facing depth camera and by moving a chair
    1.7 m along +x. NOTE: the `mid360_joint` in the repo's
    decoupled_wbc/**/g1*.urdf (rpy=(0, 0.0401, 0)) does NOT describe the
    livox_frame point data -- using it leaves the cloud upside down.
  * T_TORSO_D435_LINK: URDF d435_joint was rpy=(0, 0.8308, 0) (47.6 deg
    pitch). Re-fitted against the MID-360 by point-to-plane ICP
    (rotation + z free, x/y held at URDF); pitch is ~51.3 deg. With the URDF
    value the depth floor rose +75..+100 mm at 2.5-4 m range; with the fitted
    value it is within ~30 mm out to 4 m.
"""

import numpy as np


def rpy_to_matrix(roll, pitch, yaw):
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def make_transform(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*rpy)
    T[:3, 3] = xyz
    return T


# d435_link (x fwd, y left, z up) -> optical (z fwd, x right, y down)
R_D435_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0]])

T_TORSO_LIVOX = make_transform([0.0002835, 0.00003, 0.41618], [0.0, 3.101, 3.1415])
T_TORSO_D435_LINK = make_transform([0.0862, 0.0238, 0.4893], [-0.0123, 0.8952, 0.0258])
T_TORSO_D435_LINK_URDF = make_transform([0.0576235, 0.01753, 0.41987], [0.0, 0.8307767239493009, 0.0])

T_TORSO_OPTICAL = T_TORSO_D435_LINK.copy()
T_TORSO_OPTICAL[:3, :3] = T_TORSO_D435_LINK[:3, :3] @ R_D435_LINK_OPTICAL

# MID-360 no-returns are published as exact (0, 0, 0) (~30% of points on this
# robot) and the head/shoulders produce self-hits inside ~0.3 m.
LIDAR_MIN_RANGE = 0.35
# Beyond ~2.5 m the D435i depth noise is 5-8 cm (1-sigma) at 320x240.
DEPTH_MAX_RANGE = 2.5


def transform_points(T, points):
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def clean_lidar(points, min_range=LIDAR_MIN_RANGE):
    """Drop (0,0,0) no-returns, self-hits and non-finite points (raw livox frame)."""
    points = np.asarray(points)
    r = np.linalg.norm(points, axis=1)
    return points[np.isfinite(r) & (r > min_range)]


def clean_depth(points, max_range=DEPTH_MAX_RANGE):
    """Flatten an organized (H, W, 3) or (N, 3) optical-frame cloud, dropping
    RealSense's (0,0,0) invalid pixels and anything beyond max_range."""
    points = np.asarray(points).reshape(-1, 3)
    r = np.linalg.norm(points, axis=1)
    keep = np.isfinite(r) & (r > 0)
    if max_range:
        keep &= r < max_range
    return points[keep]


def level_rotation(up_in_torso):
    """Rotation taking the measured 'up' vector (torso frame) to +z, keeping yaw."""
    u = np.asarray(up_in_torso, dtype=np.float64)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    s, c = np.linalg.norm(v), float(u @ z)
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def up_from_livox_accel(accel_livox):
    """'Up' in the torso frame from the MID-360 IMU's linear_acceleration.

    Only valid when the robot is (roughly) static: a static accelerometer
    measures the reaction to gravity, which points up. For a walking robot,
    use the base orientation from rt/lowstate (or FAST-LIO) instead.
    """
    a = np.asarray(accel_livox, dtype=np.float64)
    return T_TORSO_LIVOX[:3, :3] @ (a / np.linalg.norm(a))


def estimate_floor_z(points_level, search=0.3, bin_size=0.01, refine=0.03):
    """Floor height in a levelled frame: the most populated 1 cm z-bin within
    `search` m above the 2nd-percentile z, refined by the median of points
    within `refine` m of it. Assumes the floor is the dominant low surface.
    (Taking a low percentile directly lands in the noise tail: ~7-9 cm below
    the true floor for D435 depth at 2.5 m, ~2.5 cm for the MID-360.)"""
    z = np.asarray(points_level)[:, 2]
    z = z[np.isfinite(z)]
    if z.size == 0:
        return float("nan")
    lo = np.percentile(z, 2)
    zl = z[z < lo + search]
    hist, edges = np.histogram(zl, bins=np.arange(lo - bin_size, lo + search + bin_size, bin_size))
    hist = np.convolve(hist, np.ones(5) / 5.0, mode="same")
    peak = edges[np.argmax(hist)] + bin_size / 2
    return float(np.median(zl[np.abs(zl - peak) < refine]))


def lidar_to_level(points_livox, accel_livox, min_range=LIDAR_MIN_RANGE):
    """Raw livox points -> cleaned, gravity-levelled torso frame (N, 3)."""
    p = transform_points(T_TORSO_LIVOX, clean_lidar(points_livox, min_range))
    return p @ level_rotation(up_from_livox_accel(accel_livox)).T


def depth_to_level(points_optical, accel_livox, max_range=DEPTH_MAX_RANGE):
    """Depth optical-frame points -> cleaned, gravity-levelled torso frame (N, 3)."""
    p = transform_points(T_TORSO_OPTICAL, clean_depth(points_optical, max_range))
    return p @ level_rotation(up_from_livox_accel(accel_livox)).T


if __name__ == "__main__":
    # Self-test: a floor 0.9 m below the pelvis, seen by an upside-down lidar.
    rng = np.random.default_rng(0) if hasattr(np.random, "default_rng") else np.random
    floor_level = np.column_stack([rng.uniform(-3, 3, 5000), rng.uniform(-3, 3, 5000), np.full(5000, -0.9)])
    floor_level = floor_level[np.hypot(floor_level[:, 0], floor_level[:, 1]) > 1.0]
    T_inv = np.linalg.inv(T_TORSO_LIVOX)
    raw = transform_points(T_inv, floor_level)
    raw = np.vstack([raw, np.zeros((2000, 3))])  # no-returns
    accel = T_inv[:3, :3] @ np.array([0.0, 0.0, 1.0])
    print("raw livox floor z (median):   %+.3f  <- upside down" % np.median(raw[:len(floor_level), 2]))
    out = lidar_to_level(raw, accel)
    print("levelled floor z (median):    %+.3f  (expect -0.900)" % np.median(out[:, 2]))
    print("points kept: %d of %d (zeros dropped)" % (len(out), len(raw)))
    assert abs(np.median(out[:, 2]) + 0.9) < 1e-6 and len(out) == len(floor_level)
    print("OK")
