# G1 depth / height map: quickstart

How to use the corrected LiDAR and depth data, and how to deploy the pipeline on this robot, on
another G1, or on another workstation. Background, validation numbers and the full setup live in
[README.md](README.md).

**What "corrected" means.** All sensor geometry lives in `../scripts/g1_frames.py`:
- The MID-360 is mounted **upside down**: raw `livox_frame` points have the floor at z ≈ +1.3 m.
- About 30% of each LiDAR frame is **(0,0,0) no-returns**, which `np.isfinite()` does not remove.
- The D435 is pitched **49.7°**, not the URDF's 47.6°.

`g1_frames` fixes all three and returns points in the **levelled torso frame**: origin at
`torso_link`, x forward, y left, z up against gravity. Do not build TF from the repo's URDFs
(`mid360_joint`, `d435_joint`); they are wrong for this hardware.

## 1. Use the corrected data

### From the robot's streams (any machine, robot standing still)

Start the publishers on the robot (skip this if `run_onboard.sh start` is already running; it
starts them too):
```bash
S=~/gam/gear_sonic_deploy/scripts
python3 $S/g1_lidar_publisher.py --iface eth0 --port 5558 --fastlio
python3 $S/g1_depth_publisher.py --port 5557 --decimate 2 --fps 15 --max-range 2.5 --no-points
```
Then, on the robot (`--host 127.0.0.1`) or any machine on its network:
```bash
python3 gear_sonic_deploy/scripts/example_corrected_points.py --host 192.168.8.227
```
It is about 100 lines and shows the whole path. In your own code:
```python
import g1_frames, videomimic_obs
from height_map import HeightMapConfig, build_height_map

lidar = g1_frames.lidar_to_level(raw_livox_points, imu_accel)   # (N,3), cleaned + levelled
cam   = g1_frames.depth_to_level(optical_points, imu_accel)     # (N,3), <= 2.5 m
floor_z = g1_frames.estimate_floor_z(lidar)                      # torso-frame z of the floor
hm  = build_height_map(np.vstack([lidar, cam]), HeightMapConfig(agg="median")) - floor_z
obs = videomimic_obs.terrain_obs(np.vstack([lidar, cam]))        # VideoMimic 11x11 input
```
- `imu_accel` is the MID-360 IMU's `linear_acceleration` (from the `lidar_imu` topic).
- Levelling with the accelerometer is only valid while the robot is roughly still. For a moving
  robot, use the onboard pipeline (section 2), which takes attitude from DLIO.
- Use `agg="median"` for anything with depth in it. `"max"` picks the upward noise tail.

Ready-made viewers:
- `view_lidar_client.py --host <ip> --stream cloud --local-heightmap`: live levelled LiDAR
  height map in the terminal.
- `g1_depth_subscriber.py --host <ip> --height-map --imu-port 5558`: the same from depth.
- `videomimic_obs.py --host <ip> --gantry`: the live VideoMimic window.

### From the onboard pipeline (ROS 2, inside the robot's container)

| Topic / frame | What |
|---|---|
| `/g1/lidar/points` (`mid360_link`) | LiDAR, no-returns removed, per-point `time` in seconds |
| `/g1/depth/points` (`d435_optical`) | depth points, <= 2.5 m |
| `/g1/imu` (`mid360_imu`) | MID-360 IMU in m/s² |
| TF `robot/odom -> torso_link -> {mid360_link, mid360_imu, d435_optical}` | DLIO pose + `g1_frames` extrinsics |
| `/elevation_mapping_node/elevation_map_raw`, `.../filtered_elevation_map` | fused height map (grid_map), 5 Hz |

These run on `ROS_DOMAIN_ID=42`, and on the robot they stay on the loopback interface (see
section 2). To read them, work inside the container:
```bash
docker exec -it g1_perception bash
source /opt/ros/humble/install/setup.bash && source /ws/install/setup.bash && export ROS_DOMAIN_ID=42
python3 /perception/dump_gridmap.py /logs/map     # -> ~/g1_perception_logs/map.npz on the robot
```
`videomimic_obs.gridmap_to_points()` turns a dumped layer back into points; the policy window is
then `videomimic_obs.terrain_obs(points, torso_xyz, torso_yaw)`.

### From the `terrain` ZMQ topic (for policies; no ROS needed)

`run_emc.sh` (so also `run_onboard.sh start`) starts `terrain_publisher.py`. At 50 Hz, gear_sonic's
control rate, it samples the latest fused map around the **current** DLIO torso pose and publishes
on `tcp://127.0.0.1:5559` (`TERRAIN_BIND` to change), topic prefix `terrain`. The format is
gear_sonic's ZMQ packed message: a 1280-byte JSON header, then the fields. That is what
`ZMQPackedMessageSubscriber` in g1_deploy_onnx_ref parses; `scripts/zmq_packed.py` does it in Python.

| Field | dtype, shape | Meaning |
|---|---|---|
| `height_grid` | f32 [50, 50] | terrain z - torso z (m), gravity-aligned, yaw-aligned with the torso; 4 cm cells, centres -0.98..+0.98 m; `[y][x]`, index 0 = most negative; NaN = unseen |
| `height_grid_valid` | bool [50, 50] | cell observed |
| `terrain_height` | f32 [11, 11] | VideoMimic window: 0.1 m, +-0.5 m, torso_z - terrain_z, `[y][x]`, 0.85 = unseen |
| `terrain_height_valid` | bool [11, 11] | window cell observed |
| `torso_grid_21` | f32 [21, 21] | torso z - surface z (m), 10 cm cells, centres -1.0..+1.0 m around torso_link, torso heading frame, `[y][x]`; NaN = unseen. The heightmap flow policy's map (humanoid-foundation-model `mmurray/fm`); sampled by `scripts/terrain_grids.py` |
| `torso_grid_21_valid` | bool [21, 21] | torso grid cell observed |
| `grid_resolution`, `grid_origin` | f32 [1], f32 [2] | 0.04; x, y of `height_grid[0][0]` in the torso heading frame |
| `torso_pos`, `torso_quat` | f64 [3], f64 [4] | DLIO pose used (robot/odom), quaternion x y z w |
| `timestamp`, `map_stamp` | f64 [1] | sample time, and the stamp of the map it came from (local clock, s) |

- A new policy resamples `height_grid` (plus its mask) into its own scan pattern. VideoMimic uses
  `terrain_height` as-is.
- `python3 gear_sonic_deploy/scripts/terrain_subscriber.py --show` checks a live stream: rate,
  staleness, coverage, and a cross-check that `height_grid` reproduces `terrain_height`.
- Measured on the gantry:
  - 50 Hz, about 1 ms from publish to receive.
  - `timestamp - map_stamp` is about 230-360 ms median: the LiDAR's own latency, the map's 5 Hz
    cycle, and where the sample falls in it.
  - The two fields agree within about 2 mm.
  - Against a window computed straight from the raw streams: median difference 0 mm, and the same
    cells seen.
- Forgetting: elevation_mapping_cupy never expires a cell that no later measurement updates, and
  it tends to reject a floor reading far below an existing obstacle as an outlier. So a person or
  object that has gone can linger even where the camera sees; one such cell went un-updated for
  212 s. `terrain_publisher.py` therefore treats cells older than `max_cell_age` (default 20 s)
  as unseen, using the mapper's `time` layer. That layer is only true seconds with
  `jetson/elevation_mapping_cupy_g1.patch` applied; unpatched it ran at 0.5x under load.
  - Tested: after an object and the arm removing it were gone, the window's 2 obstacle cells and
    ~70 grid cells vanished in one step ~21 s later.
  - While walking, cells under the feet were seen ~1-3 s earlier, well inside the limit.
  - A robot standing still for longer than the limit loses its under-foot cells. They then read
    as unseen (0.85 in the window, masked out), the same as at startup.

No policy in gam consumes this yet. Wiring one up means a matching training observation plus a
`Gather*` entry in g1_deploy_onnx_ref that subscribes to this topic.

## 2. Deploy

### A. On this robot (already set up)
```bash
cd ~/gam
GANTRY_FILTER=true LOG_DIR=~/g1_perception_logs gear_sonic_deploy/perception/jetson/run_onboard.sh start
gear_sonic_deploy/perception/jetson/run_onboard.sh status        # rates, DLIO, mapper
gear_sonic_deploy/perception/jetson/run_onboard.sh stop
```
- Keep the robot still for about 5 s after `start`, for DLIO's IMU calibration.
- Walking the robot with the stack running: follow [WALKING_TEST.md](WALKING_TEST.md).
- `GANTRY_FILTER=true` is for gantry tests only. It drops everything taller than 0.45 m within
  0.8 m of the torso.
- The container is pinned to cores 4-7. gear_sonic's walking controller pins its control thread to
  CPU 0. Override with `CPUS=...`.

### B. On another G1 (Jetson Orin, JetPack 5.1.1 / L4T R35.3.1)
1. **Check the platform:** `head -1 /etc/nv_tegra_release` should show `R35 (release), REVISION: 3.1`.
   - A different R35.x needs the matching `dustynv/ros:humble-pytorch-l4t-r35.x.x` base in
     `jetson/Dockerfile`.
   - JetPack 6 (Ubuntu 22.04) needs a different base image and is untested.
2. **Code:** `git clone -b daphne/debug-depth https://github.com/yelongshen/gam.git ~/gam`
3. **Publisher deps on the host:** `numpy pyzmq lz4 pyrealsense2`, plus
   `pip3 install -e ~/gam/external_dependencies/unitree_sdk2_python`. Check with
   `python3 -c "import pyrealsense2, unitree_sdk2py, zmq, lz4"`.
4. **Docker access:** `sudo usermod -aG docker $USER`, then log in again. The NVIDIA runtime ships
   with JetPack.
5. **Image and workspace:** run the two blocks in README.md, "Onboard (Jetson Orin NX)":
   - `docker build ...` pulls a ~6 GB base image, then builds on the Jetson; the result is about
     15 GB.
   - The workspace clones need both patches from `jetson/`.
6. **Start it** as in A.
7. **Verify the calibration on that robot** (section 3) before trusting its maps.

### C. On a workstation, over Wi-Fi (robot standing still, or for inspection)
Needs an NVIDIA GPU and Docker with the NVIDIA runtime.
1. Build the `elevation_mapping_cupy` Jazzy image and workspace: README.md, "Desktop setup".
   Remember the `cupy-cuda12x<14` patch and `USERNAME=$USER`.
2. On the robot, start only the publishers (section 1).
3. On the workstation, inside the container:
   ```bash
   docker exec emc bash -lc 'ODOM_SOURCE=dlio ROBOT_HOST=<robot ip> EMC_WS=~/ros_ws \
     <repo>/gear_sonic_deploy/perception/run_emc.sh'
   docker exec -d emc bash -lc 'source ~/ros_ws/install/setup.bash; rviz2 -d <repo>/gear_sonic_deploy/perception/g1_emc.rviz'
   ```

Do not run B/A and C against the same publishers at once: each subscriber adds about 4.5 MB/s of
Wi-Fi traffic from the robot.

### Seeing the onboard map from another machine
The onboard container keeps ROS traffic on loopback (`jetson/cyclonedds_lo.xml`) so that point
clouds never reach `eth0`, which carries Unitree's low-level control DDS. For now, look at it with
`run_onboard.sh status`, or dump it inside the container (section 1) and copy the `.npz` off.
Streaming it to RViz on a workstation would need a CycloneDDS config on `wlan0` only, never
`eth0`. That is **not tested yet**.

## 3. Calibration on a different robot

The constants in `g1_frames.py` were measured on this robot on 2026-09-28. Another unit's mounts
may differ by a degree or two, which is enough to matter: 1° of camera pitch is about 2 cm of height
error at 1 m.
- **LiDAR mount:** with the robot still, `view_lidar_client.py --stream cloud` should report a
  sensible floor (torso about 0.8-1.0 m above it), and the levelled floor should be flat. If the
  floor is at +1.3 m or tilted, the mount differs.
- **Camera:** record a bag while holding the robot still for about 15 s at each of several torso
  heights (20 cm or more apart) and a couple of headings. Then run
  `perception/calib_depth.py <bag> out.npz` and copy the reported `d435_joint` into
  `T_TORSO_D435_LINK`.
  - Use the multi-height recording. A single height cannot separate camera pitch from mounting
    height, and looks fine only at that height.
  - The script reports leave-one-pose-out errors. Adopt the fit only if they stay at a few mm.

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Floor at +1.3 m, obstacles below the floor | raw `livox_frame` points used; go through `g1_frames.lidar_to_level` |
| A spike about 1.3 m tall under the robot | (0,0,0) no-returns kept; use `g1_frames.clean_lidar` or the publisher's default `--min-range 0.35` |
| Depth floor tilts or reads high near the robot | old D435 extrinsic; use current `g1_frames.py`, or recalibrate (section 3) |
| DLIO position runs away within seconds | bursty or late IMU/LiDAR; check `run_onboard.sh status` shows a steady 10 Hz / ~175 Hz and the bridge well under one core |
| Map shifted up or down by several cm | robot raised or lowered on the gantry: DLIO tracks that as motion; in `ODOM_SOURCE=static` the bridge re-estimates the torso height every frame (~1 s to settle) |
| Obstacles right next to the robot that aren't there | people or gantry inside the LiDAR's ~1 m blind ring stay in the map; `GANTRY_FILTER=true`, or restart the stack before a test |
| Only the front of the 11x11 window filled | expected while static; the rest fills as the robot walks over mapped ground |
| `pkill -f <name>` over ssh kills your own shell | the pattern matches the ssh command line; use `pkill -f "[n]ame"` or put the commands in a script |
| Robot `docker build`: `EXPKEYSIG ... packages.ros.org` | the base image's ROS key expired; the Dockerfile refreshes it |
| `numpy has no attribute typeDict` importing cupy | focal apt scipy with numpy 1.24; the image installs scipy from pip |
