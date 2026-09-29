#!/usr/bin/env bash
# Run the perception stack ON THE ROBOT (Jetson Orin NX): sensor publishers on the host, and the
# ZMQ bridge + DLIO + elevation_mapping_cupy in the g1_perception container, talking over localhost.
#
#   gear_sonic_deploy/perception/jetson/run_onboard.sh start   # publishers + container
#   gear_sonic_deploy/perception/jetson/run_onboard.sh stop
#   gear_sonic_deploy/perception/jetson/run_onboard.sh status
#
# Env: CPUS (container cpuset, default 4-7), GANTRY_FILTER (true|false, default false),
#      WS (colcon workspace with elevation_mapping_cupy, dlio, ros2_numpy; default ~/g1_perception_ws),
#      IMAGE (default g1_perception:humble-r35.3.1), PUB_TIMEOUT (s, default 3600).
set -eo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
SCRIPTS=$(readlink -f "$HERE/../../scripts")
REPO=$(readlink -f "$HERE/../../..")
WS=${WS:-$HOME/g1_perception_ws}
IMAGE=${IMAGE:-g1_perception:humble-r35.3.1}
CPUS=${CPUS:-4-7}
PUB_TIMEOUT=${PUB_TIMEOUT:-3600}
LOG=${LOG_DIR:-/tmp/g1_perception_logs}; mkdir -p "$LOG"
# Host-side publishers: one OpenBLAS thread (numpy otherwise spins a busy thread on every core,
# including CPU 0 where gear_sonic's control thread runs).
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

start_publishers() {
  pkill -f "$SCRIPTS/g1_lidar_publisher.py" || true
  pkill -f "$SCRIPTS/g1_depth_publisher.py" || true
  sleep 1
  setsid nohup timeout "$PUB_TIMEOUT" python3 "$SCRIPTS/g1_lidar_publisher.py" --iface eth0 --port 5558 --fastlio \
    > "$LOG/lidar_pub.log" 2>&1 < /dev/null &
  setsid nohup timeout "$PUB_TIMEOUT" python3 "$SCRIPTS/g1_depth_publisher.py" --port 5557 --decimate 2 --fps 15 \
    --max-range 2.5 --no-points > "$LOG/depth_pub.log" 2>&1 < /dev/null &
}

case "${1:-start}" in
  start)
    start_publishers
    docker rm -f g1_perception > /dev/null 2>&1 || true
    docker run -d --name g1_perception --runtime nvidia --net=host --ipc=host \
      --cpuset-cpus "$CPUS" --cpu-shares 512 \
      -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
      -e CYCLONEDDS_URI=file:///perception/jetson/cyclonedds_lo.xml \
      -v "$REPO/gear_sonic_deploy/perception:/perception:ro" \
      -v "$REPO/gear_sonic_deploy/scripts:/scripts:ro" \
      -v "$WS:/ws" -v "$LOG:/logs" \
      "$IMAGE" sleep infinity > /dev/null
    if [ ! -f "$WS/install/setup.bash" ]; then
      echo "building workspace $WS (first run) ..."
      docker exec g1_perception bash -lc 'source /opt/ros/humble/install/setup.bash && cd /ws && \
        colcon build --symlink-install --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release' > "$LOG/colcon.log" 2>&1 \
        || { echo "workspace build failed, see $LOG/colcon.log"; exit 1; }
    fi
    docker exec g1_perception bash -lc "ROS_SETUP=/opt/ros/humble/install/setup.bash \
      ODOM_SOURCE=dlio GANTRY_FILTER=${GANTRY_FILTER:-false} ROBOT_HOST=127.0.0.1 EMC_WS=/ws LOG_DIR=/logs \
      EMC_EXTRA_PARAMS=/perception/jetson/onboard_overrides.yaml \
      /perception/run_emc.sh"
    echo "logs: $LOG"
    ;;
  stop)
    docker rm -f g1_perception > /dev/null 2>&1 || true
    pkill -f "$SCRIPTS/g1_lidar_publisher.py" || true
    pkill -f "$SCRIPTS/g1_depth_publisher.py" || true
    echo stopped
    ;;
  status)
    pgrep -fa "$SCRIPTS/g1_(lidar|depth)_publisher.py" | cut -c1-120 || echo "publishers: not running"
    docker ps --filter name=g1_perception --format "container: {{.Status}}" | grep . || echo "container: not running"
    [ -f "$LOG/bridge.log" ] && { echo "--- bridge"; tail -1 "$LOG/bridge.log" | cut -c1-200; }
    [ -f "$LOG/dlio.log" ] && { echo "--- dlio"; grep -E "Sensor Rates|Distance to Origin|Computation Time" "$LOG/dlio.log" | tail -3; }
    [ -f "$LOG/emc.log" ] && { echo "--- emc"; grep -iE "error|warn|Traceback" "$LOG/emc.log" | tail -2 | cut -c1-200; echo "(emc warnings shown above, if any)"; }
    ;;
esac
