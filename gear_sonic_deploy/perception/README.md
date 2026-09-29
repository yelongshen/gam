# G1 perception: height map from the head LiDAR + depth camera

**New here? Start with [QUICKSTART.md](QUICKSTART.md)**: how to use the corrected depth/LiDAR data, and how to deploy on this robot, another G1 or a workstation.

Desktop-side pipeline that turns the G1's MID-360 LiDAR and D435i depth streams (published by
`../scripts/g1_lidar_publisher.py` and `../scripts/g1_depth_publisher.py` on the robot) into a
fused elevation map (elevation_mapping_cupy) with LiDAR-inertial odometry (DLIO), and from that
into VideoMimic's `terrain_height_noisy` policy input (`../scripts/videomimic_obs.py`).

```
robot (Jetson)                                   desktop (Docker, ROS 2 Jazzy, GPU)
g1_lidar_publisher.py --fastlio  --ZMQ 5558-->  g1_zmq_bridge.py --> /g1/lidar/points (x,y,z,time[s])
                                                                 --> /g1/imu (m/s^2)
g1_depth_publisher.py --no-points --ZMQ 5557->                   --> /g1/depth/points
                                                  dlio_odom_node  --> TF robot/odom -> torso_link -> mid360_link
                                                  elevation_mapping_node --> /elevation_mapping_node/*
```

All sensor geometry comes from `../scripts/g1_frames.py` (MID-360 mounted upside down; D435
extrinsic re-fitted with `calib_depth.py`: pitch 49.7 deg). The bridge maps the sensor clock (~18 min off the Jetson) and the Jetson
clock onto the desktop clock, and converts the MID-360 IMU from g to m/s^2 and its per-point time
from ns to s.

## Robot side

```bash
S=~/gam/gear_sonic_deploy/scripts
python3 $S/g1_lidar_publisher.py --iface eth0 --port 5558 --fastlio     # full 10 Hz + per-point time
python3 $S/g1_depth_publisher.py --port 5557 --decimate 2 --fps 15 --max-range 2.5 --no-points
```
`--no-points` matters: the bridge deprojects the depth image itself, and the point stream alone is
~8 MB/s, which saturates the robot's Wi-Fi (total drops from ~11 to ~4.5 MB/s). Without
`--fastlio` the LiDAR is throttled and has no per-point time, so DLIO cannot de-skew; use
`ODOM_SOURCE=static` then.

## Desktop setup (once)

```bash
# elevation_mapping_cupy, G1 branch (tested at 449f7d4)
git clone -b dev/mheyrman/g1 https://github.com/leggedrobotics/elevation_mapping_cupy.git
cd elevation_mapping_cupy
git apply <this folder>/elevation_mapping_cupy_dockerfile.patch   # pin cupy-cuda12x<14 (14.x needs numpy 2, breaks ROS Jazzy)
# the CUDA base image already has user `ubuntu` at 1000: build with your own name/uid
docker build -f docker/Dockerfile.x64 --build-arg USERNAME=$USER \
  --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t elevation_mapping_cupy:jazzy docker/

mkdir -p ~/ros_ws/src && cd ~/ros_ws/src
ln -s <path>/elevation_mapping_cupy .
git clone -b jazzy https://github.com/Box-Robotics/ros2_numpy.git          # vcs in the image is broken
git clone -b feature/ros2 https://github.com/vectr-ucla/direct_lidar_inertial_odometry.git dlio  # tested at c8acc37

docker run -d --name emc --gpus all --net=host --ipc=host -e ROS_DOMAIN_ID=42 \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e DISPLAY=$DISPLAY -e XAUTHORITY=/tmp/.xauth \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v $XAUTHORITY:/tmp/.xauth:ro \
  -v $HOME:$HOME -w $HOME/ros_ws elevation_mapping_cupy:jazzy sleep infinity
docker exec emc bash -lc 'source /opt/ros/jazzy/setup.bash && cd ~/ros_ws && \
  rosdep install --from-paths src --ignore-src -y -r; \
  python3 -m pip install --user --break-system-packages pyzmq lz4; \
  sudo apt-get install -y python3-matplotlib; \
  colcon build --symlink-install --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release'
```
Pass your own X authority cookie (as above) rather than `xhost +local:` on a shared machine.

## Running

```bash
docker exec emc bash -lc 'ODOM_SOURCE=dlio GANTRY_FILTER=true <repo>/gear_sonic_deploy/perception/run_emc.sh'
docker exec -d emc bash -lc 'source ~/ros_ws/install/setup.bash; rviz2 -d <repo>/gear_sonic_deploy/perception/g1_emc.rviz'
```
- `ODOM_SOURCE=static` (default): no odometry; `robot/odom -> torso_link` is levelled from the IMU
  with the floor at z=0 and the torso height tracked every LiDAR frame. Only valid while the robot
  does not move. `ODOM_SOURCE=dlio`: DLIO (keep the robot still for its 3 s IMU calibration at start).
- `GANTRY_FILTER=true`: drop points > 0.45 m above the floor within 0.8 m of the torso (gantry,
  harness, people handling the robot). The robot yaws relative to the gantry, so fixed boxes do not
  work. Gantry tests only: it would also remove real tall obstacles next to the robot.
- Logs: `$LOG_DIR` (default `/tmp/g1_perception_logs`). Stop: `pkill -f "[g]1_zmq_bridge.py";
  pkill -f "[e]levation_mapping_node"; pkill -f "[d]lio_odom_node"` (the bracket keeps pkill from
  matching its own shell).

Tools (run inside the container, workspace sourced):
- `dump_gridmap.py <prefix>` - save the map layers to `<prefix>.npz/.png` (grid_map convention:
  rows along -x, columns along -y; `videomimic_obs.gridmap_to_points` reads it back).
- `watch_odom.py <secs>` / `rec_odom.py <secs> <out.npy>` - live / recorded DLIO pose.
- `eval_motion.py <bag> <prefix> [baseline_capture.npz]` - static segments + an independent
  scan-matching (ICP) check of DLIO's pose change.
- `find_attached.py <bag>` - voxels that move with the robot (gantry/self) during a recording.
- `watch_pose.py <secs>` - live torso height (from the LiDAR floor) and roll/pitch/yaw (DLIO).
- `terrain_publisher.py` - the policy-facing `terrain` ZMQ topic (50 Hz; layout in QUICKSTART.md);
  forgets cells not updated for `max_cell_age` (20 s) using the mapper's `time` layer; test it with
  `../scripts/terrain_subscriber.py`.
- `calib_depth.py <bag> <out.npz>` - fit the D435 extrinsic (and optional range models) against the
  LiDAR from a bag with the robot held still at several heights/headings; reports
  leave-one-pose-out floor errors so a fit is only adopted if it generalises.

### Re-calibrating the depth camera
Record `/g1/lidar/points /g1/imu /g1/depth/points /dlio/odom /tf /tf_static` while the robot is held
still ~15 s at each of several torso heights (>= 20 cm spread) and a few headings, then run
`calib_depth.py`. A single height is not enough: camera pitch, mounting height and any range bias
are then indistinguishable (an earlier single-height fit read the near-field floor ~4 cm high at
other heights).

## Onboard (Jetson Orin NX) - for walking

`jetson/` runs the same stack on the robot, so nothing depends on Wi-Fi. The publishers run on the
host and the bridge + DLIO + elevation_mapping_cupy run in a container, over localhost.

```bash
# once: image (ROS 2 Humble for JetPack 5.1.1 / L4T R35.3.1, torch, cupy-cuda11x 12.x), ~15 GB
docker build -t g1_perception:humble-r35.3.1 gear_sonic_deploy/perception/jetson
# once: workspace (same pins as the desktop) + the two patches in jetson/
mkdir -p ~/g1_perception_ws/src && cd ~/g1_perception_ws/src
git clone -b dev/mheyrman/g1 https://github.com/leggedrobotics/elevation_mapping_cupy.git   # 449f7d4
git -C elevation_mapping_cupy apply <repo>/gear_sonic_deploy/perception/jetson/elevation_mapping_cupy_g1.patch
git clone -b feature/ros2 https://github.com/vectr-ucla/direct_lidar_inertial_odometry.git dlio  # c8acc37
git -C dlio apply <repo>/gear_sonic_deploy/perception/jetson/dlio_pcl110.patch
git clone -b humble https://github.com/Box-Robotics/ros2_numpy.git
# run (first start builds the workspace inside the container); keep the robot still ~5 s at start
GANTRY_FILTER=true gear_sonic_deploy/perception/jetson/run_onboard.sh start   # | stop | status
```
- The container is pinned to cores 4-7 (`CPUS`) at reduced CPU shares; gear_sonic's
  `g1_deploy_onnx_ref` pins its control thread to CPU 0 at SCHED_FIFO.
- `ROS_DOMAIN_ID=42` + `cyclonedds_lo.xml` keep all ROS traffic on loopback, off `eth0`, which
  carries Unitree's low-level control DDS.
- `onboard_overrides.yaml`: map published at 5 Hz.
- Patches: `elevation_mapping_cupy_g1.patch` (a `from __future__ import annotations` for JetPack 5's
  Python 3.8, and the time layer counting real elapsed seconds instead of a fixed 0.1 per timer tick,
  which ran at ~0.5x real time under load; apply it on the desktop too) and `dlio_pcl110.patch` (`std::make_shared` -> `pcl::make_shared`; Ubuntu 20.04's
  PCL 1.10 uses boost::shared_ptr). Both still build on the desktop.
- Image notes: the base image's ROS apt key had expired (refreshed in the Dockerfile); scipy comes
  from pip because focal's apt scipy breaks on numpy 1.24.

Two performance traps found on the Jetson (both fixed): `sensor_msgs_py.create_cloud` packs point by
point in Python on Humble (the bridge now builds PointCloud2 from numpy buffers), and numpy's
OpenBLAS spun one busy thread per core (`OPENBLAS_NUM_THREADS=1` in `run_emc.sh`). Together they
cost the bridge ~3.5 cores, delayed the IMU/LiDAR stream and made DLIO diverge.

Onboard numbers (robot on the gantry): bridge 10 Hz LiDAR / 15 Hz depth / ~175 Hz IMU, ~0.4 cores;
DLIO 9 ms per scan (max 27), 0.2 cores, static drift 4.8 mm / 0.11 deg over 90 s; mapper ~0.8 cores,
5 Hz, 48 ms map latency; ~6.5 GB RAM in use system-wide; policy window from depth 1.2 cm high.
Onboard motion test (gantry pushed ~1 m, turned ~90 deg, returned; 4 min; `eval_motion.py` checks
each static segment against the first by ICP on the raw scans): DLIO error 4.1 mm / 0.34 deg after
the 1.02 m push, 10.3 mm / 0.19 deg after the turn, 2.5 mm / 0.24 deg back at the start.

## Validation (2026-09-28, robot on the gantry)

| check | result |
|---|---|
| DLIO static drift, 90 s | max 5.4 mm / 0.18 deg, final 1.0 mm / 0.12 deg |
| DLIO vs independent ICP after 1 m push + 95 deg turn + return (4 min) | 2.2 mm, 0.09 deg |
| DLIO gyro-bias signs vs raw IMU | consistent with the upside-down mount (y, z flipped) |
| mapper vs offline height map (static) | median |dz| 2.8 cm; chair 0.94 vs 0.96 m |
| VideoMimic window coverage | ~30% static (front columns only), 100% after the robot moved |
| window cells mapped by LiDAR | within ~1 cm of the true torso height |
| D435 extrinsic, leave-one-pose-out (7 poses, 0.83-1.05 m) | floor error 2-5 mm mean per bin, 0.2-2.5 m |
| window cells from depth, live (new extrinsic) | raw depth +8.6 mm, through the mapper ~+1.9 cm (was ~+7.7 cm) |

## Known limitations

- **D435 near-field:** resolved by the multi-height extrinsic fit; no range-bias model is needed
  (every range model tested generalised worse). The mapper adds ~1 cm on top of the raw depth in
  the near field. The RealSense on-chip calibration fails with `HW not ready` on this unit
  (firmware 5.15.1.55), also with Unitree's videohub stopped; it turned out not to be needed.
- **Blind-ring memory:** anything seen inside the LiDAR's ~1 m blind ring (e.g. a person next to the
  robot) stays in the map, because no later ray passes through those cells. Keep people away or
  call the mapper's clear-map service before a policy test.
- **Odometry over Wi-Fi:** fine on the gantry, but a walking deployment should run DLIO on the
  robot (the elevation_mapping_cupy G1 branch assumes onboard DLIO under the `robot` namespace).
- The repo URDFs' `mid360_joint` (upright) and `d435_joint` (47.6 deg; 49.7 measured) do not match
  the hardware; use `g1_frames.py`.
