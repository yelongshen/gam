# G1 LiDAR → Elevation Map → Policy Heightmap

End-to-end pipeline that turns the G1's onboard **Livox Mid-360** into

1. a **2.5-D elevation map** (`elevation_mapping` + FAST-LIO), and
2. the **11 × 11 `terrain_height` patch** that VideoMimic's G1 policy consumes.

The LiDAR is attached to the **robot**, but all mapping runs on the
**workstation**, bridged over ZMQ. The upstream demo
(`realsense_demo.launch`) cannot be used here — see
[Why not the upstream demo](#why-not-the-upstream-demo).

---

## 1. Architecture

```
┌─ ROBOT (Jetson, 192.168.8.227) ──────────────────────────────────┐
│  Mid-360 ──DDS──> g1_lidar_publisher.py --fastlio                │
│        rt/utlidar/cloud_livox_mid360                             │
│        rt/utlidar/imu_livox_mid360                               │
└───────────────────────────┬──────────────────────────────────────┘
                            │ ZMQ tcp://…:5558
                            │   "lidar_cloud"  (xyz + time/intensity/ring)
                            │   "lidar_imu"    (quat / gyro / accel)
┌───────────────────────────▼─ DESKTOP ────────────────────────────┐
│  zmq_to_ros_livox_bridge.py                                      │
│      -> /livox/lidar  (livox_ros_driver2/CustomMsg)              │
│      -> /livox/imu    (sensor_msgs/Imu)                          │
│                            │                                     │
│  fast_lio (laserMapping)   │  LiDAR-inertial odometry            │
│      -> /cloud_registered           frame "odom"                 │
│      -> TF  lidar_link -> odom                                   │
│                            │                                     │
│  publish_tf.py             │  static torso_link -> lidar_link,   │
│                            │         odom -> odom_torso          │
│  gravity_align_publisher.py│  odom_torso -> odom_gravity  ★      │
│                            │                                     │
│  pc_filter.py              │  z <= -0.5 in torso_link            │
│      -> /cloud_registered/filtered                               │
│                            │                                     │
│  elevation_mapping         │  80x80 @ 5cm, frame odom_gravity    │
│      -> /elevation_mapping/elevation_map{,_raw}                  │
│                            │                                     │
│  terrain_height_sampler.py │  11x11 @ 10cm, torso-centred, yaw   │
│      -> /terrain_height  (Float32MultiArray)                     │
│      -> /terrain_height_cloud (PointCloud2, debug)               │
└──────────────────────────────────────────────────────────────────┘
```

★ = added by us; everything else is upstream
[`elevation_mapping_humanoid`](https://github.com/ArthurAllshire/elevation_mapping_humanoid),
cloned at `~/ros_ws/src/elevation_mapping_humanoid`.

### Frames

| Frame | Meaning |
|---|---|
| `odom` | FAST-LIO world frame. **Not gravity-aligned** (= LiDAR pose at init). |
| `odom_torso` | `odom` rolled by 180° (Mid-360 is mounted upside-down). Still tilted. |
| **`odom_gravity`** | **Gravity-aligned.** Map frame. Yaw preserved, roll/pitch removed. |
| `torso_link` | Robot body frame; policy patch is centred here. |
| `lidar_link` | Mid-360, at `(0.0003, 0.00003, 0.41618)` from torso, rpy `(π, 2.3°, 0)`. |

---

## 2. Running it

Every desktop terminal needs:

```bash
conda activate ros_noetic
source ~/ros_ws/devel/setup.bash
```

### Robot side — 1 process

```bash
ssh grease@192.168.8.227

# once per boot: power the LiDAR on
python3 -c "
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
ChannelFactoryInitialize(0)
pub = ChannelPublisher('rt/utlidar/switch', String_); pub.Init()
pub.Write(String_(data='ON'))"

cd ~/gam/sim2real
python3 g1_lidar_publisher.py --fastlio --iface eth0 --port 5558
```

> `--fastlio` is **required**. It forwards per-point `time`/`intensity`/`ring`
> and disables throttling + range cropping. Without the per-point `time`,
> FAST-LIO cannot de-skew the non-repetitive scan and odometry degrades badly
> during motion.

### Desktop side — 5 terminals

```bash
# 1  ROS master
roscore

# 2  ZMQ -> ROS bridge (replaces the Livox driver)
cd ~/gam
python perception_heightmap/zmq_to_ros_livox_bridge.py \
       --host 192.168.8.227 --use-host-stamp

# 3  FAST-LIO + gravity alignment + elevation mapping + visualization
roslaunch elevation_mapping_demos g1_zmq_demo.launch

# 4  11x11 policy patch
cd ~/gam
python perception_heightmap/terrain_height_sampler.py --warn-coverage

# 5  visualization
rviz -d ~/gam/perception_heightmap/rviz/g1_terrain.rviz
```

**Wait ~30 s after step 3** for FAST-LIO to converge before judging the map.

Start order matters only in that `roscore` must be first; the bridge and the
launch file can start in either order.

---

## 3. Healthy startup

```
# terminal 2 (bridge)
[bridge] per-point time: max=1.0032e+08 -> using --time-scale 1 (to nanoseconds)
[bridge] cloud=9.3 Hz (20064 pts/frame), imu=151.9 Hz

# terminal 3 (launch)
[gravity] first estimate, up=[-0.023 -0.0047 0.9997] (IMU reports g-units)
[gravity] correcting 0.35 deg tilt (up=[...]), used=828 rejected=0
First corresponding point cloud and pose found, elevation mapping started.
```

### Expected-but-harmless messages

| Message | Why |
|---|---|
| `[gravity] TF not ready … not part of the same tree` | fires < 1 s, before FAST-LIO's first odometry links the tree |
| `bad callback … pc_filter.py … ConnectivityException` | same cause, once at startup |
| `Could not get pose information from robot … Buffer empty?` | pose cache empty when the first cloud lands. Logged at ERROR, should be WARN_ONCE |
| `The oldest pose available is at X, requested pose at Y` | same; note `Y < X` |
| `No point, skip this scan!` | first scan before IMU init completes |
| `imread_('/home/godm/…/grab_0.bmp')` | dead code baked into the upstream FAST-LIO binary |

If any of these **repeat past the first few seconds**, that is a real problem.

---

## 4. Verification scripts

All in `~/gam/perception_heightmap/`, all read-only.

### `check_elevation_map.py` — map contents

```bash
python perception_heightmap/check_elevation_map.py --ascii
python perception_heightmap/check_elevation_map.py --save map.png --save-npz map.npz
python perception_heightmap/check_elevation_map.py \
       --topic /elevation_mapping/elevation_map --layer elevation_inpainted
```

Decodes grid_map's column-major / centre-origin layout (the usual source of
transposed maps), lifts the grid into an `(N,3)` cloud, reports NaN coverage
with targeted diagnostics when empty.

Reference (stationary, indoors):
```
frame_id    : odom_gravity
valid cells : 3427/6400 (53.5%)
elevation   : min=-1.290 max=-0.864 mean=-1.219 std=0.093 m
```

### `check_floor_tilt.py` — gravity alignment

```bash
python perception_heightmap/check_floor_tilt.py
```

Fits a plane to the floor in each candidate frame. The floor is genuinely
flat, so residual tilt **is** that frame's gravity misalignment, which shows
up in the map as a phantom ramp of `tan(tilt) × map_length`.

```
frame            tilt    ramp/4.0m     rms    inliers
odom            2.31d      0.161m   0.022m   284/1847
odom_torso      1.31d      0.091m   0.007m  1525/1847
odom_gravity    0.42d      0.029m   0.007m  1525/1847   <- map frame
torso_link      0.98d      0.069m   0.007m  1525/1847
```

### `terrain_height_sampler.py` — the policy patch

```bash
python perception_heightmap/terrain_height_sampler.py --once --ascii
python perception_heightmap/terrain_height_sampler.py --warn-coverage
```

| Flag | Effect |
|---|---|
| `--map-topic` | default fused; `…_raw` for lowest latency |
| `--layer` | `elevation` or `elevation_inpainted` |
| `--body-frame` | `torso_link` (matches `HeightfieldCfg.body_name`) |
| `--size` / `--resolution` | `1.0` / `0.1` → 11 × 11 |
| `--true-mean-fill` | fill holes with mean of *valid* cells instead of sim's zero-biased mean |
| `--once --ascii` | print one patch and exit |

---

## 5. The heightmap: matching VideoMimic exactly

`terrain_height_sampler.py` is the hardware replacement for
`videomimic_gym/legged_gym/utils/raycaster/sensors.py::HeightfieldSensor`.
In sim that raycasts a warp mesh of ground truth; here there is no mesh, so
the raycast becomes a **resample** of the elevation map.

| VideoMimic source | Convention reused |
|---|---|
| `raycaster_patterns.py::grid_pattern` | `linspace(-size/2, size/2, nx+1)`, `meshgrid(indexing="xy")` → 11 × 11, endpoints included |
| `sensors.py::update_buffers` | `only_heading=True` → rotate by **yaw only** |
| `torch_jit_utils.py::calc_heading` | `atan2` of rotated `+x` ref dir — **not** Euler yaw |
| `sensors.py::_update_depth_map` | value = **downward ray distance**, not absolute z |
| same | NaN/inf → mean-fill (zero-biased quirk preserved) |

The policy config (`g1_deepmimic_config.py`) defines four sensors, but
**only `terrain_height` is wired into `ObsProcActor`/`ObsProcCritic`**:

| Sensor | Shape | Body | In policy obs? |
|---|---|---|---|
| `terrain_height` | **11 × 11** | `torso_link` | ✅ |
| `terrain_height_noisy` | 11 × 11 | `torso_link` | ❌ (DR variant) |
| `root_height` | 1 × 1 | `pelvis` | ❌ |
| `link_heights` | 13 × 1 | 13 tracked bodies | ❌ |

`link_heights` is **not** a heightmap — one downward ray per link, i.e. ground
clearance per body.

Sim's DR envelope, useful as an accuracy target:
```
white_noise 0.02   offset_noise 0.02          # 2 cm
roll/pitch  0.04 rad (2.3°)   yaw 0.08 rad
max_delay 3        bad_distance_prob 0.01
```
Our measured gravity residual (0.2–0.4°) and floor rms (5–8 mm) sit inside it.
Uncorrected `odom_torso` (1.1–3.1°) did **not**.

### Converting map → patch

```
z_grav   = bilinear_sample(elevation_map, p_xy)   # odom_gravity
distance = torso_z_in_gravity − z_grav            # sim's "ray distance"
```
Floor at `z ≈ −1.19` with torso at `0` → `distance ≈ 1.19 m`.

---

## 6. What the map does and does not contain

`pc_filter.py` keeps `z ≤ -0.5` in **`torso_link`**. With the floor at
`z ≈ -0.87`, that is **~0.37 m above whatever surface the robot stands on** —
a robot-relative window that rises as the robot climbs.
`elevation_mapping` applies a second cutoff, `sensor_processor/ignore_points_above: 0.4`.

| Object | In map? |
|---|---|
| Floor, cables, thresholds | ✅ |
| Kerbs, low platforms, ramps | ✅ |
| **Stairs descending** | ✅ fully |
| **Stairs ascending** | ✅ next ~2 risers (0.17 m each); window slides as you climb |
| Chair/table **legs** (lower part) | ⚠️ partial, thin → unreliable |
| Chair **seats** (~0.45 m) | ❌ |
| Table/desk **tops** (0.7–0.8 m) | ❌ |
| Walls, people, doors | ❌ |

Measured: raw cloud spans `z = -0.90 … +1.91` in `torso_link`;
after `pc_filter` it is `-0.94 … -0.50`. Map elevation spans ~0.47 m on flat
floor — floor plus low clutter only.

**This is deliberate.** The map is 2.5-D (one height per cell, no "under"),
so a retained desk becomes a solid plateau: the planner would treat the
desktop as walkable and the floor beneath it as nonexistent. Overhead
obstacles need a 3-D representation.

To raise the ceiling anyway:
```xml
<node pkg="elevation_mapping_demos" type="pc_filter.py" name="pc_filter">
  <param name="distance_threshold" value="-0.1"/>   <!-- ~0.77 m above floor -->
</node>
```
and raise `sensor_processor/ignore_points_above` to match. Not recommended
for anything feeding control.

---

## 7. The central hole (not a bug)

The Mid-360 has a **360° × 59° FOV, −7°…+52° vertical**, so it cannot see
steeply downward. Measured on this robot:

```
floor slab, raw /cloud_registered in torso_link:
  horiz radius min = 0.948 m,  p1 = 1.069 m
  bins  <0.25 | 0.25-0.5 | 0.5-0.75 | 0.75-1 | 1-1.25 | ...
           0  |     0    |     0    |   27   |  1950  | ...
```

Predicted: sensor 1.286 m above floor → `1.286 / tan(52°) = 1.005 m`.
Measured 0.948 m. **Agrees within a few percent** — it is the sensor's blind
cone, not `blind:`, not `pc_filter`, not a config error.

Consequence for the policy: the patch is ±0.5 m, **entirely inside the hole**
while stationary, so all 121 cells get mean-filled. Sim never produced this
(raycasting always hits). The map is world-fixed with
`enable_visibility_cleanup: false`, so the centre should fill in once walking
— **verify this with `--warn-coverage` before closing the loop.**

---

## 8. Our changes to the upstream repo

All under `~/ros_ws/src/elevation_mapping_humanoid/…/elevation_mapping_demos/`,
each with a `.bak` alongside. Copies of our versions live in
`~/gam/perception_heightmap/ros_patches/`.

| File | Change | Why |
|---|---|---|
| `scripts/gravity_align_publisher.py` | **new** | continuous gravity estimation → `odom_gravity` |
| `config/robots/simple_demo_robot.yaml` | `map_frame_id: odom_torso` → `odom_gravity` | build the map in a gravity-aligned frame |
| same | `fused_map_publishing_rate: 50` → `5` | 50 Hz inpainting starves the node |
| `launch/g1_zmq_demo.launch` | run new gravity node; `pose_publisher map_frame:=odom_gravity` | pose must share the map's frame |
| `config/postprocessing/postprocessor_pipeline.yaml` | drop `NormalVectorsFilter`; inpaint radius `0.05` → `0.15` | see below |
| `config/visualization/fused.yaml` | add `elevation_inpainted_cloud` | view the hole-filled layer |
| `rviz/elevation_map_visualization.rviz` | Fixed Frame → `odom_gravity` | — |

Revert any of it with the matching `.bak`; the launch file also accepts
`use_legacy_gravity:=true`.

### Why the gravity node was needed

FAST-LIO's world frame is the LiDAR pose at init, **not** gravity-aligned, and
the Mid-360 is mounted with ~2.33° pitch. The floor therefore read as a
1.1–3.1° ramp — 0.08–0.22 m of phantom height across the 4 m map.

The stock `gravity_correction_publisher.py` was supposed to fix this but
measured **worse than no correction** (1.55° residual vs 0.79° uncorrected):
it averages only the first ~10 IMU samples, once, applies a hand-tuned
`R = -I` flip, never re-estimates — and `map_frame_id` was `odom_torso`
anyway, so `odom_corrected` only ever rotated the **rviz display**.

`gravity_align_publisher.py` reads the IMU's specific force, rejects samples
taken while accelerating or rotating, rotates into the world frame via
FAST-LIO's live attitude, EMA-filters, and publishes the minimal rotation
taking `+z` onto true up (**yaw untouched**). Result: **0.2–0.4°**.

### Why the filter pipeline was edited

Filter chains are **all-or-nothing**. `gridMapFilters/NormalVectorsFilter`
lives in `grid_map_filters`, which is not in this RoboStack env and will not
compile (bundled `EigenLab.h` vs this Eigen/GCC). That one missing plugin
silently disabled the `inpaint` step too:

```
Couldn't find filter of type gridMapFilters/NormalVectorsFilter
Could not configure the filter chain. Will publish the raw elevation map without postprocessing!
```

Nothing consumes `normal_vectors_*` (verified by grep across yaml/py/rviz/cpp),
so the step is removed rather than fixed. `InpaintFilter` comes from
`grid_map_cv`, which **is** installed.

---

## 9. Why not the upstream demo

`roslaunch elevation_mapping_demos realsense_demo.launch` **cannot work here.**
It includes `livox_ros_driver2/msg_MID360.launch`, which opens UDP sockets
directly to the LiDAR:

```
[error] Create detection socket failed.  →  Init lds lidar failed!  →  bind failed
/livox/lidar : NO DATA   →   elevation map entirely NaN (0/6400)
```

| | |
|---|---|
| Driver expects host | `192.168.1.5` |
| Driver expects LiDAR | `192.168.1.12` (unreachable) |
| This workstation | `192.168.8.192/24` |
| LiDAR actually on | robot, `192.168.8.227` |

Their setup assumes the LiDAR is cabled to the ROS machine. `g1_zmq_demo.launch`
is the same stack minus that driver, fed by the ZMQ bridge instead.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `cannot launch node of type [grid_map_visualization/…]` | stale entry in `devel/.catkin`. Fix: `cd ~/ros_ws && catkin build && source devel/setup.bash`. Never `catkin build <single_pkg>` in this merged devel space. |
| Map 0 % valid, topics alive | give FAST-LIO ~30 s. Then check `check_floor_tilt.py`, then `/pose` vs `/cloud_registered` stamp age. |
| `rostopic hz` prints nothing though data flows | **artifact**: `timeout … \| tail` SIGTERMs python before stdout flushes. Use `python -u`, redirect to a file, or `timeout -s INT`. |
| Bridge alive but ROS topics silent | stale bridge instance holding the ZMQ sub. `pkill -f zmq_to_ros_livox_bridge` and restart. |
| Map tilted / ramped | gravity node not running, or map frame not `odom_gravity`. Check `check_floor_tilt.py`. |
| Big hole in the middle | expected — blind cone, §7. |
| Bridge drops 9 Hz → 2 Hz | `msg.points = [CustomPoint(...) for i in range(20064)]` — rospy serializes 20 k sub-messages in Python. Fix by packing the buffer with numpy. |

### Clock skew

The bridge stamps with `--use-host-stamp` (robot wall clock) while
`pose_publisher` uses the desktop clock. After a **robot reboot** the Jetson
may not have re-synced NTP; `time_tolerance` is 1.0 s. Check:

```
/cloud_registered            stamp_age=+0.121 s
/cloud_registered/filtered   stamp_age=+0.185 s
/pose                        stamp_age=+0.008 s
```

---

## 11. File map

```
~/gam/perception_heightmap/
  zmq_to_ros_livox_bridge.py     ZMQ -> /livox/lidar + /livox/imu
  gravity_align_publisher.py     odom_torso -> odom_gravity        [new]
  terrain_height_sampler.py      map -> 11x11 policy patch         [new]
  check_elevation_map.py         map inspector (ascii/png/npz)     [new]
  check_floor_tilt.py            gravity-alignment measurement     [new]
  view_lidar_client.py           raw ZMQ stream viewer
  rviz/g1_terrain.rviz           map + magenta 11x11 patch         [new]
  ros_patches/                   copies of our edits to ros_ws

~/gam/sim2real/
  g1_lidar_publisher.py          runs ON THE ROBOT

~/ros_ws/src/elevation_mapping_humanoid/   upstream, see §8
```

---

## 12. Open items

1. **Blind-cone coverage while walking** — unverified. If the patch stays
   ~0% observed in motion, the 11 × 11 cannot come from this LiDAR alone
   (options: D435 at `camera_depth_optical_frame`, second LiDAR, or retrain
   with a hole mask).
2. **`publish_map_frame.py` is broken** — looks up parent
   `odom_torso_corrected`, which no node publishes, so the `height_map` frame
   is never broadcast. Harmless today (nothing consumes it) but it is the
   yaw-only robot-centred frame the upstream authors intended.
3. **Bridge throughput** under sustained motion (see §10).
4. **Inpaint value unproven** — the chain now loads and produces
   `elevation_inpainted`, but its benefit was never compared side by side.
