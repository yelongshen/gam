# G1 perception: height map from the head LiDAR + depth camera

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

All sensor geometry comes from `../scripts/g1_frames.py` (MID-360 mounted upside down; D435 pitch
re-fitted to ~51.3 deg). The bridge maps the sensor clock (~18 min off the Jetson) and the Jetson
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

## Validation (2026-09-28, robot on the gantry)

| check | result |
|---|---|
| DLIO static drift, 90 s | max 5.4 mm / 0.18 deg, final 1.0 mm / 0.12 deg |
| DLIO vs independent ICP after 1 m push + 95 deg turn + return (4 min) | 2.2 mm, 0.09 deg |
| DLIO gyro-bias signs vs raw IMU | consistent with the upside-down mount (y, z flipped) |
| mapper vs offline height map (static) | median |dz| 2.8 cm; chair 0.94 vs 0.96 m |
| VideoMimic window coverage | ~30% static (front columns only), 100% after the robot moved |
| window cells mapped by LiDAR | within ~1 cm of the true torso height |

## Known limitations

- **D435 near-field bias:** the floor within ~1.5 m reads 2-8 cm high, above the ~2 cm offset
  noise VideoMimic trained with. The camera extrinsic was fitted at a single robot height, where
  pitch, height offset and range bias cannot be separated (a fit at 0.90 m fails at 0.82 m); needs a
  multi-height/tilt recording. The RealSense on-chip calibration fails with `HW not ready` on this
  unit (firmware 5.15.1.55), also with Unitree's videohub stopped.
- **Blind-ring memory:** anything seen inside the LiDAR's ~1 m blind ring (e.g. a person next to the
  robot) stays in the map, because no later ray passes through those cells. Keep people away or
  call the mapper's clear-map service before a policy test.
- **Odometry over Wi-Fi:** fine on the gantry, but a walking deployment should run DLIO on the
  robot (the elevation_mapping_cupy G1 branch assumes onboard DLIO under the `robot` namespace).
- The repo URDFs' `mid360_joint` (upright) and `d435_joint` (47.6 deg) do not match the hardware; use
  `g1_frames.py`.
