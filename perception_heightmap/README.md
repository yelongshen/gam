# `perception_heightmap/`

All LiDAR / depth-camera → **heightmap (elevation map)** code for the G1, collected in one
place. Everything here is workstation-side tooling and experiment stages; the on-robot
publishers (`sim2real/g1_depth_publisher.py`, `sim2real/g1_lidar_publisher.py`) stay with the
other robot-side scripts because they are deployed to the robot alongside them.

## Contents

| file | role |
|---|---|
| `g1_depth_subscriber.py` | **Main** workstation subscriber for the robot's depth stream: decode → deproject → optical→body transform → `ElevationGridMap`, with 2D/3D visualisation and `.npz` capture. |
| `g1_depth_subscriber_stage2.py` … `_stage6.py` | Incremental experiment stages kept for reference/bisection (each stage adds one processing step on top of the previous). |
| `g1_depth_subscriber_real_emc.py` | Variant that drives the real elevation-map-cupy (EMC) fuser using the robot's own base pose; runs under `.venv_emc`. |
| `g1_lidar_subscriber_elevation.py` | **LiDAR path** (verified against VideoMimic's own paper/code as the sensor it actually uses on real hardware, per `dev_notes/heightmap_architecture_analysis.md`): subscribes `sim2real/g1_lidar_publisher.py`'s raw `lidar_cloud` ZMQ stream, fuses it into the same `ElevationGridMap` accumulator used by the depth path, resamples onto the sim-matching 11x11/0.1m egocentric grid, and republishes as `lidar_heightmap` for `view_lidar_client.py --view-heightmap`. Ships with a **known limitation**: no real translation odometry yet (only heading rotates) — see its module docstring before trusting it for anything beyond a stationary bench test. |
| `view_depth_heightmap.py` | Standalone viewer for the depth ZMQ stream; owns `optical_to_body_frame` and `interpolate_height_idw` (the reference implementations of the frame math and hole filling). |
| `view_lidar_client.py` | Viewer/client for the LiDAR ZMQ stream; owns the wire format (`parse_message`, `HEADER_SIZE`), `print_heightmap` / `heightmap_to_image`, and `ElevationMapFuser`. |
| `height_map.py` | ROS→ZMQ bridge: republishes a ROS heightmap topic (from the real `elevation_mapping_humanoid` ROS node) on the `view_lidar_client.py` wire format. Needs `rospy` + a sourced ROS environment on the robot; `g1_lidar_subscriber_elevation.py` is a ROS-free alternative that does its own fusion instead of bridging an existing ROS-side fuser. |
| `zmq_to_ros_livox_bridge.py` | **ZMQ→ROS bridge (the reverse direction)**: consumes `sim2real/g1_lidar_publisher.py --fastlio`'s `lidar_cloud`/`lidar_imu` ZMQ streams and republishes them as the ROS topics FAST-LIO expects — `/livox/lidar` (`livox_ros_driver2/CustomMsg`, with per-point `offset_time` preserved for de-skewing) and `/livox/imu` (`sensor_msgs/Imu`). This is what lets the **real** `elevation_mapping_humanoid` stack run entirely on the workstation with **no ROS on the robot** ("Option 3"). Runs in the `ros_noetic` conda env. |
| `elevation_grid_map.py` | The elevation grid-map accumulator itself (point cloud → 2.5D grid), shared by both the depth and LiDAR paths. |

## Two pipelines

**(1) From-scratch Python** (no ROS anywhere) — quick, approximate:
`g1_lidar_publisher.py` → `g1_lidar_subscriber_elevation.py` (own `ElevationGridMap` fusion + Open3D/OpenCV views).

**(2) Real `elevation_mapping_humanoid`** (ROS on the workstation only) — the
stack VideoMimic actually deploys, incl. FAST-LIO odometry and rviz's
`grid_map_rviz_plugin`:
`g1_lidar_publisher.py --fastlio` → `zmq_to_ros_livox_bridge.py` → FAST-LIO →
`elevation_mapping` → rviz.

Workstation ROS env (RoboStack, Ubuntu 22.04-compatible):
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros_noetic
source ~/ros_ws/devel/setup.bash
```

## Typical use

```bash
# on the robot -- depth-camera path
python3 sim2real/g1_depth_publisher.py
# on the robot -- LiDAR path, pipeline (1): throttled/cropped, fine for viewing
python3 sim2real/g1_lidar_publisher.py --iface eth0 --port 5558
# on the robot -- LiDAR path, pipeline (2): FULL cloud + per-point time, for FAST-LIO
python3 sim2real/g1_lidar_publisher.py --iface eth0 --port 5558 --fastlio

# on the workstation -- pipeline (1)
.venv_sim/bin/python perception_heightmap/g1_depth_subscriber.py --visualize
.venv_sim/bin/python perception_heightmap/g1_lidar_subscriber_elevation.py \
    --host 192.168.8.227 --port 5558 --visualize-3d
.venv_sim/bin/python perception_heightmap/view_lidar_client.py --host 192.168.8.227 --port 5558

# on the workstation -- pipeline (2), each in its own terminal (env sourced as above)
roscore
python perception_heightmap/zmq_to_ros_livox_bridge.py --host 192.168.8.227
roslaunch fast_lio mapping_mid360.launch
roslaunch elevation_mapping_demos realsense_demo.launch   # vestigial name; it's the LiDAR pipeline
```

Note the two ZMQ streams use **different wire formats**: the LiDAR/heightmap path is defined by
`view_lidar_client.py` (and produced by `height_map.py` / `g1_lidar_subscriber_elevation.py` /
`g1_lidar_publisher.py`), while the depth path is defined by `g1_depth_publisher.py` /
`g1_depth_subscriber.py`. Keep encoder and decoder in sync when changing either.


