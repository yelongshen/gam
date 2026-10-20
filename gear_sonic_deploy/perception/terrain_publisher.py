#!/usr/bin/env python3
"""terrain_publisher.py - policy-agnostic terrain input over ZMQ (gear_sonic packed-message format).

Samples the latest fused elevation map (elevation_mapping_cupy, ~5 Hz) around the CURRENT torso pose
(TF robot/odom -> torso_link from DLIO, ~100 Hz) at `rate` Hz (default 50, gear_sonic's control rate)
and publishes one message per sample on ZMQ PUB `bind` (default tcp://127.0.0.1:5559), topic prefix
`terrain`, in the format of ../scripts/zmq_packed.py:

  height_grid          f32 [50, 50]  terrain_z - torso_z (m), gravity-aligned, yaw-aligned with the
                                     torso heading; 4 cm cells, centres -0.98..+0.98 m around
                                     torso_link; [y][x], index 0 = most negative; NaN where unseen
  height_grid_valid    bool [50, 50] cell observed
  terrain_height       f32 [11, 11]  VideoMimic window (0.1 m, +-0.5 m, torso_z - terrain_z, [y][x],
                                     0.85 where unseen) - see ../scripts/videomimic_obs.py
  terrain_height_valid bool [11, 11] window cell observed (nearest map point within 0.15 m)
  grid_resolution      f32 [1]       0.04
  grid_origin          f32 [2]       x, y of height_grid[0][0]'s centre, torso heading frame (m)
  torso_pos            f64 [3]       torso position in robot/odom used for this sample
  torso_quat           f64 [4]       torso orientation in robot/odom, x y z w
  torso_grid_21        f32 [21, 21]  torso_z - surface_z (m), 10 cm cells, centres -1.0..+1.0 m around
                                     torso_link, torso heading frame, [y][x]; NaN where unseen. The
                                     heightmap flow policy's map (humanoid-foundation-model mmurray/fm)
  torso_grid_21_valid  bool [21, 21] cell observed
  timestamp            f64 [1]       when this sample was taken (local clock, s)
  map_stamp            f64 [1]       stamp of the map it was sampled from; staleness = timestamp - map_stamp
  pose_stamp           f64 [1]       stamp of the torso pose (odometry, or TF) the grids are centred on
  pose_ok              bool [1]      the pose passed ../scripts/pose_guard.py; when False every grid is
                                     all-unseen (NaN / not valid), so any consumer gets "no map"

Forgetting: cells not updated for more than `max_cell_age` seconds (default 20; 0 = off) are treated
as unseen, using the mapper's "time" layer (seconds since each cell's last update; it must be in the
map_topic's published layers - see g1_desktop_overrides.yaml). elevation_mapping_cupy itself never
expires a cell that no later ray passes through, so something that stood in the LiDAR's blind ring
(a person, the gantry) would otherwise stay in the policy input indefinitely. While walking, cells
under the feet were seen ~1-3 s earlier, well inside the default.

Pose: the torso pose comes from DLIO's odometry topic (`odom_topic`, default /dlio/odom: 100 Hz,
~5 ms old on arrival), received on its own thread. The TF robot/odom -> torso_link only updates at
the LiDAR's 10 Hz, and with its listener on this node's single thread it went unread while the map
callback ran: grids were centred on poses up to 0.7 s old (gantry, 2026-10-19). An empty
`odom_topic` falls back to that TF.

Pose guard: a fresh pose can still be wrong -- DLIO diverged twice on the gantry under CPU load
(2026-10-20), to 59 m and to 3.5 m, and the map re-centred on it showed phantom surfaces up to
~60 cm around a robot on a flat floor. ../scripts/pose_guard.py rejects a pose that moves or turns
faster than a G1 can, or puts the torso implausibly high or low above the floor the map shows, and
keeps rejecting it for 2 s after the last violation. While rejected the message carries pose_ok =
False and no observed cells; once per rejection the mapper's map is cleared (`clear_service`,
default /elevation_mapping_node/clear_map; empty = off), since points fused with a wrong pose
would otherwise stay. `pose_guard` = False turns the guard off.

Run inside the perception container (run_emc.sh starts it).
"""
import sys
import threading
import time

import numpy as np
import rclpy
import zmq
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

import terrain_grids as tg
from pose_guard import PoseGuard
import videomimic_obs as vo
import zmq_packed
from elevation_mapping_cupy.gridmap_utils import decode_multiarray_to_rows_cols


class TerrainPublisher(Node):
    def __init__(self):
        super().__init__("terrain_publisher")
        p = {k: self.declare_parameter(k, v).value for k, v in [
            ("map_topic", "/elevation_mapping_node/elevation_map_raw"), ("layer", "elevation"),
            ("map_frame", "robot/odom"), ("base_frame", "torso_link"), ("odom_topic", "/dlio/odom"),
            ("bind", "tcp://127.0.0.1:5559"), ("topic", "terrain"), ("rate", 50.0),
            ("grid_size", 50), ("grid_resolution", 0.04),
            ("max_cell_age", 20.0), ("age_layer", "time"),
            ("pose_guard", True), ("clear_service", "/elevation_mapping_node/clear_map")]}
        self.p = p
        n, res = int(p["grid_size"]), float(p["grid_resolution"])
        c = (np.arange(n) - (n - 1) / 2.0) * res
        gx, gy = np.meshgrid(c, c, indexing="xy")          # gx[y][x] = c[x]  ->  [y][x] layout
        self.local = np.stack([gx.ravel(), gy.ravel()], 1)
        self.grid_shape = (n, n)
        self.grid_origin = np.array([c[0], c[0]], dtype=np.float32)
        self.grid_res = np.array([res], dtype=np.float32)

        self.map = None       # dict: E, res, cx, cy, stamp, points, tree
        self.odom = None      # (pos, quat xyzw, stamp) of the latest odometry message
        self.guard = PoseGuard() if p["pose_guard"] else None
        self.guard_lock = threading.Lock()
        self.n_rejected = 0
        self.clear_client = self.create_client(Trigger, p["clear_service"]) if p["pose_guard"] and p["clear_service"] else None
        if p["odom_topic"]:
            # Own callback group: the multi-threaded executor runs it while the map callback or a
            # tick is busy, so the pose stays current.
            self.create_subscription(Odometry, p["odom_topic"], self.on_odom, 10,
                                     callback_group=MutuallyExclusiveCallbackGroup())
        else:
            self.tf = Buffer()
            TransformListener(self.tf, self)
        self.create_subscription(GridMap, p["map_topic"], self.on_map, 2)
        self.sock = zmq.Context.instance().socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 4)
        self.sock.bind(p["bind"])
        self.topic = p["topic"].encode()
        self.n_sent, self.t_stat = 0, time.time()
        self.create_timer(1.0 / float(p["rate"]), self.tick)
        pose_src = p["odom_topic"] or f"TF {p['map_frame']}->{p['base_frame']}"
        self.get_logger().info(f"publishing '{p['topic']}' on {p['bind']} at {p['rate']} Hz "
                               f"from {p['map_topic']} [{p['layer']}] + pose {pose_src}")

    def on_odom(self, msg):
        if msg.header.frame_id != self.p["map_frame"] or msg.child_frame_id != self.p["base_frame"]:
            self.get_logger().warn(f"odometry is {msg.header.frame_id}->{msg.child_frame_id}, expected "
                                   f"{self.p['map_frame']}->{self.p['base_frame']}", throttle_duration_sec=10.0)
            return
        t, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.odom = (np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w]),
                     msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        if self.guard is not None:
            with self.guard_lock:
                self.guard.odom(self.odom[2], self.odom[0], self.odom[1])

    def on_map(self, msg):
        if self.p["layer"] not in msg.layers:
            self.get_logger().warn(f"layer {self.p['layer']!r} not in {list(msg.layers)}", throttle_duration_sec=5.0)
            return
        layers = list(msg.layers)
        E = decode_multiarray_to_rows_cols(self.p["layer"], msg.data[layers.index(self.p["layer"])])
        n_stale = 0
        if float(self.p["max_cell_age"]) > 0:
            if self.p["age_layer"] in layers:
                age = decode_multiarray_to_rows_cols(self.p["age_layer"], msg.data[layers.index(self.p["age_layer"])])
                stale = np.isfinite(E) & (age > float(self.p["max_cell_age"]))
                n_stale = int(stale.sum())
                E = np.where(stale, np.nan, E)
            else:
                self.get_logger().warn(f"max_cell_age set but layer {self.p['age_layer']!r} not published "
                                       f"(have {layers}); not forgetting", throttle_duration_sec=30.0)
        res = float(msg.info.resolution)
        cx, cy = msg.info.pose.position.x, msg.info.pose.position.y
        pts = vo.gridmap_to_points(E, res, (cx, cy))
        self.map = dict(E=E, res=res, cx=cx, cy=cy, stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                        n_stale=n_stale,
                        points=pts, tree=cKDTree(pts[:, :2]) if len(pts) >= vo.KNN_K else None)

    def tick(self):
        m = self.map
        if m is None:
            return
        if self.p["odom_topic"]:
            if self.odom is None:
                self.get_logger().warn(f"no odometry on {self.p['odom_topic']} yet", throttle_duration_sec=5.0)
                return
            pos, quat, pose_stamp = self.odom
        else:
            try:
                tf = self.tf.lookup_transform(self.p["map_frame"], self.p["base_frame"], rclpy.time.Time())
            except Exception as e:  # noqa: BLE001 - TF not up yet
                self.get_logger().warn(f"no TF yet: {e}", throttle_duration_sec=5.0)
                return
            t, q = tf.transform.translation, tf.transform.rotation
            pose_stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            pos = np.array([t.x, t.y, t.z])
            quat = np.array([q.x, q.y, q.z, q.w])
            if self.guard is not None:
                with self.guard_lock:
                    self.guard.odom(pose_stamp, pos, quat)
        yaw = Rotation.from_quat(quat).as_euler("zyx")[0]

        # generic grid: nearest map cell (same 4 cm resolution as the mapper) at each rotated cell centre
        h = tg.sample_map(m["E"], m["res"], m["cx"], m["cy"], pos, yaw, self.local) - np.float32(pos[2])
        valid = np.isfinite(h)
        torso_h, torso_valid = tg.torso_grid(m["E"], m["res"], m["cx"], m["cy"], pos, yaw)

        if m["tree"] is not None:
            obs, seen = vo.terrain_obs(m["points"], pos, yaw, return_mask=True, tree=m["tree"])
        else:
            obs, seen = np.full((vo.GRID_N, vo.GRID_N), vo.DEFAULT_HEIGHT), np.zeros((vo.GRID_N, vo.GRID_N), bool)

        pose_ok = True
        if self.guard is not None:
            with self.guard_lock:
                self.guard.floor(pose_stamp, h, valid, self.local)
                pose_ok, why = self.guard.state(pose_stamp)
                new_episode = self.guard.take_new_episode()
            if not pose_ok:
                # no observed cells: every consumer (old runners included) sees "no map"
                h = np.full_like(h, np.nan)
                valid = np.zeros_like(valid)
                torso_h, torso_valid = np.full_like(torso_h, np.nan), np.zeros_like(torso_valid)
                seen = np.zeros_like(seen)
                self.n_rejected += 1
                self.get_logger().warn(f"pose rejected ({', '.join(why)}): publishing no map",
                                       throttle_duration_sec=1.0)
            if new_episode and self.clear_client is not None:
                if self.clear_client.service_is_ready():
                    self.clear_client.call_async(Trigger.Request())
                    self.get_logger().warn(f"cleared the elevation map ({self.p['clear_service']})")
                else:
                    self.get_logger().warn(f"{self.p['clear_service']} not available; map not cleared",
                                           throttle_duration_sec=10.0)

        self.sock.send(zmq_packed.pack(self.topic, {
            "height_grid": h.reshape(self.grid_shape),
            "height_grid_valid": valid.reshape(self.grid_shape),
            "terrain_height": obs.astype(np.float32),
            "terrain_height_valid": seen,
            "torso_grid_21": torso_h.astype(np.float32),
            "torso_grid_21_valid": torso_valid,
            "grid_resolution": self.grid_res,
            "grid_origin": self.grid_origin,
            "torso_pos": pos,
            "torso_quat": quat,
            "timestamp": np.array([time.time()]),
            "map_stamp": np.array([m["stamp"]]),
            "pose_stamp": np.array([pose_stamp]),
            "pose_ok": np.array([pose_ok]),
        }))
        self.n_sent += 1
        now = time.time()
        if now - self.t_stat >= 10.0:
            self.get_logger().info(f"{self.n_sent / (now - self.t_stat):.1f} Hz; grid {valid.mean()*100:.0f}% valid, "
                                   f"window {seen.mean()*100:.0f}% seen; map age {now - m['stamp']:.2f} s, "
                                   f"pose age {now - pose_stamp:.3f} s; "
                                   f"{m['n_stale']} map cells forgotten (> {self.p['max_cell_age']} s old); "
                                   f"pose rejected in {self.n_rejected} samples")
            self.n_sent, self.t_stat, self.n_rejected = 0, now, 0


def main():
    rclpy.init(args=sys.argv)
    node = TerrainPublisher()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
