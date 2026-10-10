#!/usr/bin/env python3
"""check_floor_tilt.py
=====================
Measure how far a candidate elevation-map frame is from being gravity-aligned,
by fitting a plane to the floor points of `/cloud_registered` expressed in that
frame.

This is the quantitative test behind `gravity_align_publisher.py`. The logic:
the floor is genuinely flat, so ANY residual tilt you measure in a world frame
is that frame's misalignment with gravity -- which shows up in the elevation
map as a phantom ramp of `tan(tilt) * map_length` metres.

Typical output on this G1 before the fix:
    odom           floor tilt 1.67 deg   ramp over 4.0m = 0.117 m
    odom_torso     floor tilt 1.14 deg   ramp over 4.0m = 0.080 m
    odom_corrected floor tilt 1.16 deg   ramp over 4.0m = 0.081 m   <- no better
    torso_link     floor tilt 0.75 deg   (sanity: floor really is flat, 5mm rms)

Usage:
    conda activate ros_noetic && source ~/ros_ws/devel/setup.bash
    python perception_heightmap/check_floor_tilt.py
    python perception_heightmap/check_floor_tilt.py --frames odom_torso odom_gravity
"""
import argparse
import sys
import time

import numpy as np

import rospy
import tf2_ros
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


def fit_floor_plane(pts: np.ndarray, floor_pct: float, inlier: float):
    """Iteratively-reweighted plane fit z = ax + by + c on the lowest points.

    Taking only the lowest `floor_pct` of points seeds the fit on the floor
    rather than on walls/furniture, then the inlier loop refines it.
    Returns (a, b, c, inlier_mask, rms).
    """
    q = pts[pts[:, 2] < np.percentile(pts[:, 2], floor_pct)]
    if len(q) < 10:
        return None
    idx = np.ones(len(q), dtype=bool)
    coef = np.zeros(3)
    for _ in range(20):
        A = np.c_[q[idx, 0], q[idx, 1], np.ones(int(idx.sum()))]
        coef, *_ = np.linalg.lstsq(A, q[idx, 2], rcond=None)
        res = q[:, 2] - (coef[0] * q[:, 0] + coef[1] * q[:, 1] + coef[2])
        new = np.abs(res) < inlier
        if new.sum() < 10 or (new == idx).all():
            idx = new if new.sum() >= 10 else idx
            break
        idx = new
    res = q[idx, 2] - (coef[0] * q[idx, 0] + coef[1] * q[idx, 1] + coef[2])
    return coef, idx, float(np.sqrt((res ** 2).mean())), len(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud-topic", default="/cloud_registered")
    ap.add_argument("--frames", nargs="+",
                    default=["odom", "odom_torso", "odom_corrected",
                             "odom_gravity", "torso_link"])
    ap.add_argument("--map-length", type=float, default=4.0,
                    help="elevation map side length, for the phantom-ramp figure")
    ap.add_argument("--floor-pct", type=float, default=40.0)
    ap.add_argument("--inlier", type=float, default=0.04)
    ap.add_argument("--frames-avg", type=int, default=5,
                    help="average the tilt over this many LiDAR frames")
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    rospy.init_node("check_floor_tilt", anonymous=True, disable_signals=True)
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf)
    time.sleep(2.0)

    clouds = []
    for _ in range(args.frames_avg):
        try:
            clouds.append(rospy.wait_for_message(args.cloud_topic, PointCloud2,
                                                 timeout=args.timeout))
        except rospy.ROSException:
            break
    if not clouds:
        print(f"[tilt] TIMEOUT on {args.cloud_topic} -- is FAST-LIO running?",
              file=sys.stderr)
        return 1

    print(f"[tilt] {len(clouds)} cloud(s) from {args.cloud_topic} "
          f"(source frame {clouds[0].header.frame_id})")
    print(f"[tilt] {'frame':<16} {'tilt':>8}  {'ramp/%.1fm' % args.map_length:>10}  "
          f"{'rms':>7}  {'inliers':>12}")

    for frame in args.frames:
        tilts, rmss, ratios = [], [], []
        for msg in clouds:
            try:
                tr = buf.lookup_transform(frame, msg.header.frame_id,
                                          rospy.Time(0), rospy.Duration(1.0))
            except Exception:
                tilts = None
                break
            pts = np.array(list(pc2.read_points(do_transform_cloud(msg, tr),
                                                field_names=("x", "y", "z"),
                                                skip_nans=True)), dtype=np.float64)
            out = fit_floor_plane(pts, args.floor_pct, args.inlier)
            if out is None:
                continue
            coef, idx, rms, n = out
            n_vec = np.array([-coef[0], -coef[1], 1.0])
            n_vec /= np.linalg.norm(n_vec)
            tilts.append(np.degrees(np.arccos(np.clip(n_vec[2], -1, 1))))
            rmss.append(rms)
            ratios.append(f"{int(idx.sum())}/{n}")

        if tilts is None:
            print(f"[tilt] {frame:<16} {'--':>8}  (frame does not exist in TF)")
            continue
        if not tilts:
            print(f"[tilt] {frame:<16} {'--':>8}  (plane fit failed)")
            continue
        t = float(np.mean(tilts))
        print(f"[tilt] {frame:<16} {t:7.2f}d  {np.tan(np.radians(t)) * args.map_length:9.3f}m  "
              f"{np.mean(rmss):6.3f}m  {ratios[-1]:>12}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
