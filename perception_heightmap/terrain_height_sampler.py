#!/usr/bin/env python3
"""terrain_height_sampler.py
===========================
Turn the live `elevation_mapping` grid into the EXACT `terrain_height`
observation VideoMimic's G1 policy was trained on: an **11 x 11 patch,
0.1 m resolution, centred on `torso_link`, yaw-aligned**, whose values are
*downward ray distances* (not absolute heights).

This is the real-robot replacement for
`videomimic_gym/legged_gym/utils/raycaster/sensors.py::HeightfieldSensor`.
In sim that class raycasts a warp mesh of known ground truth; on hardware
there is no mesh, only a sparse 2.5-D grid with holes, so the raycast becomes
a **resample**. Every convention below is copied from the sim code so the
observation distribution matches.

--------------------------------------------------------------------------
WHAT IS REPLICATED FROM VIDEOMIMIC (and where)
--------------------------------------------------------------------------
1. Grid geometry -- `raycaster_patterns.py::grid_pattern`:
       nx = int(size[0]/resolution);  x = linspace(-size[0]/2, size[0]/2, nx+1)
   With size=(1.0, 1.0), resolution=0.1 this gives 11 samples per axis,
   i.e. the grid SPANS the full +-0.5 m (endpoints included), and
   `grid_width = int(size/res) + 1 = 11` as in `HeightfieldSensor.__init__`.
   `ordering="xy"` -> `torch.meshgrid(x, y, indexing="xy")`, which makes the
   FIRST array axis y and the SECOND x. We reproduce that so the flattened
   order fed to the policy is identical.

2. Yaw-only alignment -- `sensors.py::update_buffers`:
       quat_to_apply = calc_heading_quat(body_quat) if cfg.only_heading ...
   `HeightfieldCfg.only_heading = True` for this sensor, and
   `torch_jit_utils.py::calc_heading` is
       ref_dir = (1,0,0); rot_dir = quat_rotate(q, ref_dir)
       heading = atan2(rot_dir.y, rot_dir.x)
   so the patch is rotated about world +z by the torso's heading ONLY --
   roll and pitch are discarded and the grid stays horizontal.

   >>> This is why the map must be sampled in a GRAVITY-ALIGNED frame.
   Sim's grid is horizontal by construction; FAST-LIO's `odom`/`odom_torso`
   is tilted by the LiDAR mount (~2.3 deg), which would shear the patch.
   `gravity_align_publisher.py` provides `odom_gravity` for exactly this.

3. Value convention -- `sensors.py::_update_depth_map` with `use_float=True`:
       heights = distances.clone()
   i.e. the stored value is the ray's travel distance from the sensor origin
   (the torso) down to the terrain, NOT the terrain's absolute z. Rays are
   vertical, so distance = torso_z - terrain_z.

4. NaN / inf policy -- `sensors.py::_update_depth_map`:
       filled      = where(isinf|isnan, 0, depth_map)
       mean_filled = mean(filled)
       depth_map   = where(isnan|isinf, mean_filled, depth_map)
   Note the quirk faithfully reproduced here: the mean is taken over an array
   in which invalid cells were set to ZERO (not excluded), so it is a
   *sum of valid / total count* -- biased low when holes are present. We
   copy it anyway, because matching the training distribution matters more
   than being statistically tidy. `--true-mean-fill` opts out.

--------------------------------------------------------------------------
WHAT IS DELIBERATELY NOT REPLICATED
--------------------------------------------------------------------------
The domain-randomisation of `terrain_height_noisy` (white/offset/roll/pitch/
yaw noise, delay, `bad_distance_prob`). Those exist to make the policy robust
to a real sensor; the real sensor supplies its own noise. The policy consumes
the CLEAN `terrain_height` name -- see `g1_deepmimic_config.py`, where the
only sensor wired into `ObsProcActor`/`ObsProcCritic` is `terrain_height`.

--------------------------------------------------------------------------
KNOWN SIM-TO-REAL GAP (read this before trusting the output)
--------------------------------------------------------------------------
The patch spans +-0.5 m around the torso. On this G1 the Mid-360 has a
measured ~0.95 m blind-cone radius (360 x 59 deg FOV, -7..+52 deg vertical,
sensor ~1.29 m above the floor => 1.286/tan(52 deg) ~ 1.0 m). So while the
robot STANDS STILL, every one of the 121 cells lies inside the hole and is
mean-filled. In sim a raycast always returns a hit, so the policy never saw
this. The map is world-fixed with `enable_visibility_cleanup: false`, so the
centre should fill in once the robot walks. `--warn-coverage` logs the
fraction of genuinely-observed cells so you can watch this.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    conda activate ros_noetic && source ~/ros_ws/devel/setup.bash
    python perception_heightmap/terrain_height_sampler.py

    # inspect once, with an ASCII dump, without publishing
    python perception_heightmap/terrain_height_sampler.py --once --ascii

Publishes:
    /terrain_height        std_msgs/Float32MultiArray  (11x11 row-major,
                           dim[0]="y"/rows, dim[1]="x"/cols -- matches
                           depth_map.view(-1, grid_height, grid_width))
    /terrain_height_cloud  sensor_msgs/PointCloud2     (debug, in map frame)
"""
import argparse
import sys

import numpy as np

import rospy
import tf2_ros
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from sensor_msgs.msg import PointCloud2, PointField


# ----------------------------------------------------------------------------
# Ported from videomimic_gym/legged_gym/tensor_utils/torch_jit_utils.py
# ----------------------------------------------------------------------------
def quat_rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector `v` by quaternion `q` (x, y, z, w).

    Same algebra as `torch_jit_utils.my_quat_rotate`, written for a single
    quaternion and an (N, 3) array of vectors.
    """
    x, y, z, w = q_xyzw
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w ** 2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * w)
    c = q_vec[None, :] * (v @ q_vec)[:, None] * 2.0
    return a + b + c


def calc_heading(q_xyzw: np.ndarray) -> float:
    """Heading (yaw) of a quaternion, per `torch_jit_utils.calc_heading`.

    Rotates the reference direction +x and takes atan2(y, x) of the result.
    This is NOT the same as the roll-pitch-yaw 'yaw' when the body is heavily
    tilted, which is precisely why we copy it rather than using euler angles.
    """
    ref_dir = np.array([[1.0, 0.0, 0.0]])
    rot_dir = quat_rotate(q_xyzw, ref_dir)[0]
    return float(np.arctan2(rot_dir[1], rot_dir[0]))


def grid_pattern(size=(1.0, 1.0), resolution=0.1):
    """Ray start offsets, per `raycaster_patterns.grid_pattern` (ordering='xy').

    Returns (ray_starts (N,3), grid_height, grid_width) where N = 11*11 and
    the flatten order matches sim's `depth_map.view(-1, grid_h, grid_w)`.
    """
    nx = int(size[0] / resolution)
    ny = int(size[1] / resolution)
    x = np.linspace(-size[0] / 2.0, size[0] / 2.0, nx + 1)
    y = np.linspace(-size[1] / 2.0, size[1] / 2.0, ny + 1)

    # torch.meshgrid(x, y, indexing="xy") -> arrays of shape (len(y), len(x)).
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")

    ray_starts = np.zeros((grid_x.size, 3), dtype=np.float64)
    ray_starts[:, 0] = grid_x.ravel()
    ray_starts[:, 1] = grid_y.ravel()
    # grid_height/grid_width per HeightfieldSensor.__init__
    return ray_starts, int(size[1] / resolution) + 1, int(size[0] / resolution) + 1


# ----------------------------------------------------------------------------
# GridMap decoding (same convention as check_elevation_map.py)
# ----------------------------------------------------------------------------
def gridmap_layer_to_array(msg: GridMap, layer: str) -> np.ndarray:
    """(n_rows, n_cols) float32 view of a GridMap layer.

    grid_map serializes column-major with dim[0]=columns, dim[1]=rows, so we
    reshape to (n_cols, n_rows) and transpose. Row/col 0 is the +x/+y corner.
    """
    if layer not in msg.layers:
        raise KeyError(f"layer {layer!r} not in {list(msg.layers)}")
    arr = msg.data[msg.layers.index(layer)]
    n_cols = arr.layout.dim[0].size
    n_rows = arr.layout.dim[1].size
    return np.asarray(arr.data, dtype=np.float32).reshape(n_cols, n_rows).T


def sample_bilinear(grid: np.ndarray, msg: GridMap, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """NaN-aware bilinear sample of the elevation grid at map-frame (xs, ys).

    grid_map's x/y DECREASE with increasing row/column index and the map is
    centred on `msg.info.pose.position`, so the fractional index is

        fx = (cx + (n_rows-1)/2*res - x) / res

    Corners that are NaN (unobserved) are dropped and the remaining weights
    renormalised; a sample is NaN only if all four corners are NaN. This
    degrades gracefully at hole boundaries instead of eroding them.
    """
    n_rows, n_cols = grid.shape
    res = msg.info.resolution
    cx, cy = msg.info.pose.position.x, msg.info.pose.position.y

    fx = (cx + (n_rows - 1) / 2.0 * res - xs) / res
    fy = (cy + (n_cols - 1) / 2.0 * res - ys) / res

    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    tx = fx - x0
    ty = fy - y0

    out = np.full(xs.shape, np.nan, dtype=np.float64)
    acc = np.zeros(xs.shape)
    wsum = np.zeros(xs.shape)

    for dx, wx in ((0, 1.0 - tx), (1, tx)):
        for dy, wy in ((0, 1.0 - ty), (1, ty)):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < n_rows) & (yi >= 0) & (yi < n_cols)
            if not ok.any():
                continue
            w = wx * wy
            vals = np.full(xs.shape, np.nan)
            vals[ok] = grid[xi[ok], yi[ok]]
            good = ok & np.isfinite(vals) & (w > 0)
            acc[good] += vals[good] * w[good]
            wsum[good] += w[good]

    hit = wsum > 1e-9
    out[hit] = acc[hit] / wsum[hit]
    return out


def make_cloud(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * points.shape[0]
    msg.is_dense = False
    msg.data = points.astype(np.float32).tobytes()
    return msg


class TerrainHeightSampler:
    def __init__(self, args):
        self.args = args
        self.ray_starts, self.grid_h, self.grid_w = grid_pattern(
            (args.size, args.size), args.resolution)
        rospy.loginfo(f"[terrain] pattern {self.grid_h}x{self.grid_w} "
                      f"({self.ray_starts.shape[0]} rays), size={args.size} m, "
                      f"res={args.resolution} m")

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

        self.pub = rospy.Publisher(args.out_topic, Float32MultiArray, queue_size=1)
        self.pub_cloud = rospy.Publisher(args.out_cloud_topic, PointCloud2, queue_size=1)
        self.sub = rospy.Subscriber(args.map_topic, GridMap, self.cb, queue_size=1)
        self.n_pub = 0
        self.t_log = rospy.Time.now()

    def cb(self, msg: GridMap):
        map_frame = msg.info.header.frame_id
        try:
            tr = self.buffer.lookup_transform(map_frame, self.args.body_frame,
                                              rospy.Time(0), rospy.Duration(0.1))
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[terrain] TF {map_frame}<-{self.args.body_frame}: {e}")
            return

        t = tr.transform.translation
        q = tr.transform.rotation
        body_pos = np.array([t.x, t.y, t.z])
        body_quat = np.array([q.x, q.y, q.z, q.w])

        # --- yaw-only rotation, exactly as sim's calc_heading_quat path ---
        heading = calc_heading(body_quat)
        ch, sh = np.cos(heading), np.sin(heading)
        rot = np.array([[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]])
        pts_w = self.ray_starts @ rot.T + body_pos          # ray_starts_w

        try:
            grid = gridmap_layer_to_array(msg, self.args.layer)
        except KeyError as e:
            rospy.logwarn_throttle(5.0, f"[terrain] {e}")
            return

        terrain_z = sample_bilinear(grid, msg, pts_w[:, 0], pts_w[:, 1])

        # Rays point straight down from the torso => distance is a pure z delta.
        distances = body_pos[2] - terrain_z
        valid = np.isfinite(distances)
        coverage = float(valid.mean())

        # --- sim-matched NaN handling (sensors.py::_update_depth_map) ---
        if self.args.true_mean_fill:
            fill = float(np.nanmean(distances)) if valid.any() else 0.0
        else:
            filled = np.where(valid, distances, 0.0)
            fill = float(filled.mean())          # note: zeros included, as in sim
        distances = np.where(valid, distances, fill)
        distances = np.clip(distances, 0.0, self.args.max_distance)

        depth_map = distances.reshape(self.grid_h, self.grid_w).astype(np.float32)

        out = Float32MultiArray()
        out.layout = MultiArrayLayout(dim=[
            MultiArrayDimension(label="y", size=self.grid_h, stride=self.grid_h * self.grid_w),
            MultiArrayDimension(label="x", size=self.grid_w, stride=self.grid_w),
        ], data_offset=0)
        out.data = depth_map.ravel().tolist()
        self.pub.publish(out)

        if self.pub_cloud.get_num_connections() > 0:
            cloud_pts = pts_w.copy()
            cloud_pts[:, 2] = body_pos[2] - distances
            self.pub_cloud.publish(make_cloud(cloud_pts, map_frame, msg.info.header.stamp))

        self.n_pub += 1
        now = rospy.Time.now()
        if self.args.warn_coverage and (now - self.t_log).to_sec() > 5.0:
            rospy.loginfo(
                f"[terrain] {self.n_pub / 5.0:.1f} Hz | observed {100*coverage:5.1f}% "
                f"({int(valid.sum())}/{valid.size}) | fill={fill:.3f} | "
                f"dist min={distances.min():.3f} max={distances.max():.3f} | "
                f"heading={np.degrees(heading):+.1f} deg")
            if coverage < 0.05:
                rospy.logwarn(
                    "[terrain] patch is almost entirely unobserved -- expected while "
                    "standing still (Mid-360 blind cone ~0.95 m vs patch radius 0.5 m). "
                    "Should fill in once walking.")
            self.n_pub = 0
            self.t_log = now

        if self.args.once:
            self.report_once(depth_map, coverage, heading)
            rospy.signal_shutdown("done")

    def report_once(self, depth_map, coverage, heading):
        print(f"\n[terrain] shape={depth_map.shape}  observed={100*coverage:.1f}%  "
              f"heading={np.degrees(heading):+.1f} deg")
        print(f"[terrain] distance min={depth_map.min():.3f} max={depth_map.max():.3f} "
              f"mean={depth_map.mean():.3f} std={depth_map.std():.3f} m")
        if self.args.ascii:
            print("[terrain] distance grid (metres below torso), row 0 = +y:")
            for row in depth_map:
                print("   " + " ".join(f"{v:5.2f}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-topic", default="/elevation_mapping/elevation_map",
                    help="fused map (has the inpainted layer); use "
                         "'/elevation_mapping/elevation_map_raw' for lowest latency")
    ap.add_argument("--layer", default="elevation",
                    help="'elevation_inpainted' uses the postprocessed layer if present")
    ap.add_argument("--body-frame", default="torso_link",
                    help="must match HeightfieldCfg.body_name in g1_deepmimic_config.py")
    ap.add_argument("--size", type=float, default=1.0, help="patch side length [m]")
    ap.add_argument("--resolution", type=float, default=0.1, help="patch cell size [m]")
    ap.add_argument("--max-distance", type=float, default=5.0,
                    help="matches HeightfieldCfg.max_distance")
    ap.add_argument("--out-topic", default="/terrain_height")
    ap.add_argument("--out-cloud-topic", default="/terrain_height_cloud")
    ap.add_argument("--true-mean-fill", action="store_true",
                    help="fill holes with the mean of VALID cells instead of sim's "
                         "zero-included mean (diverges from training distribution)")
    ap.add_argument("--warn-coverage", action="store_true", default=True)
    ap.add_argument("--once", action="store_true", help="print one patch and exit")
    ap.add_argument("--ascii", action="store_true", help="with --once, dump the grid")
    args, _ = ap.parse_known_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("terrain_height_sampler")
    TerrainHeightSampler(args)
    rospy.loginfo(f"[terrain] {args.map_topic}[{args.layer}] -> {args.out_topic}")
    rospy.spin()


if __name__ == "__main__":
    main()
