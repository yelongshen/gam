#!/usr/bin/env python3
"""gravity_align_publisher.py
============================
Publish a CONTINUOUSLY-ESTIMATED, gravity-aligned world frame
(`odom_torso` -> `odom_gravity`) so `elevation_mapping` can build its grid in a
frame whose +z really is "up".

WHY THIS EXISTS
---------------
FAST-LIO's world frame (`odom`, and hence `odom_torso`) is **not**
gravity-aligned: it is simply the LiDAR/IMU pose at initialization. FAST-LIO
tracks gravity as an internal state vector but never rotates its output frame
to match. On this G1 the Mid-360 is mounted with a ~2.33 deg pitch
(`main.urdf`: rpy = (0, 3.101, 3.1415)), so `odom_torso` inherits that tilt
plus whatever the torso's attitude was at startup.

Measured consequence: the floor is flat to 5 mm when expressed in
`torso_link`, but reads as a 1.1-3.1 deg RAMP in `odom_torso` -- i.e. a
0.08-0.22 m phantom height change across the 4 x 4 m elevation map. That is a
fake slope big enough to ruin footstep planning.

The stock `gravity_correction_publisher.py` tries to fix this but does not
work in practice:
  * it averages only the FIRST ~10 IMU samples, once, then unregisters;
  * it applies a hand-tuned `R = -I` "trial and error" IMU->LiDAR flip;
  * it never re-estimates, so any FAST-LIO drift is baked in;
  * measured: it produced a 2.29 deg correction for a 1.14 deg actual tilt,
    leaving 1.16 deg -- no better than doing nothing.
Most importantly, `elevation_mapping`'s `map_frame_id` is `odom_torso`, so its
output was never corrected at all; `odom_corrected` only ever affected the
rviz *display*.

HOW THIS WORKS
--------------
A stationary accelerometer measures SPECIFIC FORCE f = a - g, so at rest
f points UP with magnitude |g|. (Verified on this robot: the Mid-360 IMU reads
[-0.04, 0.015, -0.996] in units of g -- note it reports g-units, not m/s^2,
and its +z axis points DOWN, which is why `publish_tf.py` flips
`odom` -> `odom_torso` by roll = -180 deg.)

So, each cycle:
  1. read the IMU's specific force `f` in the LiDAR/IMU body frame;
  2. rotate it into `odom_torso` using FAST-LIO's current attitude
     (TF `odom_torso` -> `lidar_link`), giving the true "up" vector;
  3. low-pass filter it, and REJECT samples taken while accelerating
     (||f| - 1g| too large) or rotating fast -- during those, f is not gravity;
  4. publish `odom_torso` -> `odom_gravity` as the minimal rotation that takes
     +z onto that measured up vector.

Because the transform is parent->child, a point expressed in `odom_gravity` is
`p_g = R^T (p_t - t)`. We want `R^T * up = +z`, hence R = rotation taking +z to
`up` -- which is exactly the minimal (geodesic) rotation computed below. Yaw is
deliberately left unconstrained (gravity says nothing about heading), so this
only ever removes roll/pitch.

USAGE
-----
    conda activate ros_noetic && source ~/ros_ws/devel/setup.bash
    python perception_heightmap/gravity_align_publisher.py

Then point elevation_mapping at it (in `simple_demo_robot.yaml`):
    map_frame_id: odom_gravity

Validate with:
    python perception_heightmap/check_floor_tilt.py
"""
import argparse
import sys

import numpy as np

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu

G_MS2 = 9.80665


def quat_from_z_to(up: np.ndarray) -> np.ndarray:
    """Minimal rotation quaternion (x, y, z, w) taking +z onto `up`.

    Uses the half-way-vector formulation, which is numerically stable except at
    the antipode (up ~ -z), handled explicitly. Adding no yaw component is
    precisely what we want: gravity constrains only roll and pitch.
    """
    z = np.array([0.0, 0.0, 1.0])
    up = up / np.linalg.norm(up)
    d = float(np.dot(z, up))
    if d > 1.0 - 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if d < -1.0 + 1e-9:
        # 180 deg: any axis perpendicular to z works.
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.cross(z, up)
    w = 1.0 + d
    q = np.array([axis[0], axis[1], axis[2], w])
    return q / np.linalg.norm(q)


class GravityAlignPublisher:
    def __init__(self, args):
        self.args = args
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()

        self.up = None          # filtered "up" unit vector, in odom_torso
        self.n_used = 0
        self.n_rejected = 0

        self.sub = rospy.Subscriber("/livox/imu", Imu, self.imu_cb, queue_size=50)

    def imu_cb(self, msg: Imu):
        f = np.array([msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z], dtype=np.float64)
        norm = np.linalg.norm(f)
        if norm < 1e-6:
            return

        # The Mid-360 reports acceleration in g-units (|f| ~ 1.0), but other
        # firmwares use m/s^2 (|f| ~ 9.81). Normalising makes us unit-agnostic;
        # we only ever use the DIRECTION of f.
        g_units = norm < 3.0
        expected = 1.0 if g_units else G_MS2

        # Reject samples where the robot is clearly accelerating: then
        # f = a - g is not parallel to gravity and would bias the estimate.
        if abs(norm - expected) / expected > self.args.accel_tol:
            self.n_rejected += 1
            return
        w = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                      msg.angular_velocity.z])
        if np.linalg.norm(w) > self.args.gyro_tol:
            self.n_rejected += 1
            return

        # Rotate f (body frame) into odom_torso using FAST-LIO's attitude.
        try:
            tr = self.buffer.lookup_transform(
                self.args.parent_frame, self.args.body_frame,
                rospy.Time(0), rospy.Duration(0.05))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, f"[gravity] TF not ready: {e}")
            return

        q = tr.transform.rotation
        R = quaternion_matrix_3x3(np.array([q.x, q.y, q.z, q.w]))
        up_meas = R @ (f / norm)

        if self.up is None:
            self.up = up_meas
            rospy.loginfo(f"[gravity] first estimate, up={np.round(self.up, 4)} "
                          f"(IMU reports {'g-units' if g_units else 'm/s^2'})")
        else:
            a = self.args.alpha
            self.up = (1.0 - a) * self.up + a * up_meas
            self.up /= np.linalg.norm(self.up)
        self.n_used += 1

    def publish(self):
        if self.up is None:
            rospy.logwarn_throttle(2.0, "[gravity] waiting for usable IMU + TF ...")
            return
        q = quat_from_z_to(self.up)

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.args.parent_frame
        t.child_frame_id = self.args.child_frame
        # Pure rotation: the gravity-aligned frame shares its origin with
        # odom_torso, so heights stay referenced to the same point.
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x, t.transform.rotation.y = q[0], q[1]
        t.transform.rotation.z, t.transform.rotation.w = q[2], q[3]
        self.broadcaster.sendTransform(t)

    def run(self):
        rate = rospy.Rate(self.args.rate)
        last_log = rospy.Time.now()
        while not rospy.is_shutdown():
            self.publish()
            now = rospy.Time.now()
            if self.up is not None and (now - last_log).to_sec() > 5.0:
                tilt = np.degrees(np.arccos(np.clip(self.up[2], -1, 1)))
                rospy.loginfo(f"[gravity] correcting {tilt:.2f} deg tilt "
                              f"(up={np.round(self.up, 4)}), "
                              f"used={self.n_used} rejected={self.n_rejected}")
                self.n_used = self.n_rejected = 0
                last_log = now
            rate.sleep()


def quaternion_matrix_3x3(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from an (x, y, z, w) quaternion (no tf dependency)."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-frame", default="odom_torso")
    ap.add_argument("--child-frame", default="odom_gravity")
    ap.add_argument("--body-frame", default="lidar_link",
                    help="frame the IMU's linear_acceleration is expressed in "
                         "(Mid-360 IMU->LiDAR extrinsic is ~identity)")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--alpha", type=float, default=0.02,
                    help="EMA weight per accepted sample (~1s settling at 150Hz)")
    ap.add_argument("--accel-tol", type=float, default=0.06,
                    help="reject IMU samples whose |f| deviates from 1g by more "
                         "than this fraction (robot is accelerating)")
    ap.add_argument("--gyro-tol", type=float, default=0.35,
                    help="reject IMU samples with |omega| above this (rad/s)")
    # roslaunch appends __name:=... / __log:=... to argv, which argparse would
    # reject (exit code 2). rospy.myargv() strips those ROS remapping args.
    args = ap.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("gravity_align_publisher")
    node = GravityAlignPublisher(args)
    rospy.loginfo(f"[gravity] publishing {args.parent_frame} -> {args.child_frame}")
    node.run()


if __name__ == "__main__":
    main()
