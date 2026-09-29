#!/usr/bin/env bash
# Run inside the `emc` container: G1 ZMQ bridge + elevation_mapping_cupy (dev/mheyrman/g1 config
# + desktop overrides), and with ODOM_SOURCE=dlio also DLIO odometry. See README.md in this folder.
#   ODOM_SOURCE=static|dlio GANTRY_FILTER=true|false ROBOT_HOST=192.168.8.227 \
#   EMC_WS=~/ros_ws LOG_DIR=/tmp/g1_perception_logs [EMC_EXTRA_PARAMS=<yaml>]  gear_sonic_deploy/perception/run_emc.sh
# Stop with: pkill -f "[g]1_zmq_bridge.py"; pkill -f "[e]levation_mapping_node"; pkill -f "[d]lio_odom_node"
set -eo pipefail
source "${ROS_SETUP:-/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash}"
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
source "${EMC_WS:-$HOME/ros_ws}/install/setup.bash"
SHARE=$(ros2 pkg prefix elevation_mapping_cupy)/share/elevation_mapping_cupy
DLIO_SHARE=$(ros2 pkg prefix direct_lidar_inertial_odometry)/share/direct_lidar_inertial_odometry
LOG=${LOG_DIR:-/tmp/g1_perception_logs}; mkdir -p "$LOG"
export PYTHONPATH="$HERE/../scripts:$PYTHONPATH"
# numpy's OpenBLAS spawns one busy-waiting thread per core for every small matrix product; on the
# Jetson that alone cost the bridge ~3.5 cores and starved DLIO. One thread is plenty here.
# (OMP_NUM_THREADS is left alone: DLIO's GICP uses OpenMP.)
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
ODOM_SOURCE=${ODOM_SOURCE:-static}

nohup python3 "$HERE"/g1_zmq_bridge.py --ros-args -p host:=${ROBOT_HOST:-192.168.8.227} \
  -p odom_source:=$ODOM_SOURCE -p gantry_filter:=${GANTRY_FILTER:-false} > "$LOG/bridge.log" 2>&1 &

if [ "$ODOM_SOURCE" = "dlio" ]; then
  nohup ros2 run direct_lidar_inertial_odometry dlio_odom_node --ros-args -r __node:=dlio_odom \
    --params-file "$DLIO_SHARE/cfg/dlio.yaml" --params-file "$DLIO_SHARE/cfg/params.yaml" \
    --params-file "$HERE"/dlio_g1.yaml \
    -r pointcloud:=/g1/lidar/points -r imu:=/g1/imu \
    -r odom:=/dlio/odom -r pose:=/dlio/pose -r path:=/dlio/path -r deskewed:=/dlio/deskewed \
    > "$LOG/dlio.log" 2>&1 &
fi

nohup ros2 run elevation_mapping_cupy elevation_mapping_node.py --ros-args \
  --params-file "$SHARE/config/core/core_param.yaml" \
  --params-file "$SHARE/config/setups/g1/g1_parameters.yaml" \
  --params-file "$SHARE/config/setups/g1/g1_sensor_parameter.yaml" \
  --params-file "$HERE"/g1_desktop_overrides.yaml \
  ${EMC_EXTRA_PARAMS:+--params-file "$EMC_EXTRA_PARAMS"} \
  -p plugin_config_file:="$SHARE/config/setups/g1/g1_plugin_config.yaml" \
  > "$LOG/emc.log" 2>&1 &
echo "started bridge ($ODOM_SOURCE) + elevation_mapping_node$( [ "$ODOM_SOURCE" = dlio ] && echo ' + dlio_odom_node'); logs in $LOG"
