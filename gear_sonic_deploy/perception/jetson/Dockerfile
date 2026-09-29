# Onboard perception image for the G1's Jetson Orin NX (JetPack 5.1.1 / L4T R35.3.1, CUDA 11.4):
# ROS 2 Humble (built from source by jetson-containers, Python 3.8) + PyTorch + cupy, with the
# ROS packages DLIO and elevation_mapping_cupy need that the base image lacks, built from source.
#
#   docker build -t g1_perception:humble-r35.3.1 gear_sonic_deploy/perception/jetson
FROM dustynv/ros:humble-pytorch-l4t-r35.3.1

ENV DEBIAN_FRONTEND=noninteractive
ARG ROS_WS=/opt/ros_deps

# The base image's ROS apt key expired (EXPKEYSIG F42ED6FBAB17C654); refresh it before apt-get update.
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpcl-dev libeigen3-dev libboost-all-dev python3-shapely python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# cupy 13 needs Python >= 3.9; 12.x is the last line with Python 3.8 + CUDA 11 aarch64 wheels.
# scipy from pip: the focal apt scipy (1.3) breaks on numpy 1.24 (np.typeDict), which cupy imports.
RUN python3 -m pip install --no-cache-dir "cupy-cuda11x>=12.3,<13" "numpy<2" "scipy==1.10.1" simple-parsing \
      ruamel.yaml pyzmq lz4 transforms3d

# ROS 2 packages missing from this ros-base build (checked: cv_bridge, message_filters, tf2_ros,
# tf2_eigen, tf2_sensor_msgs, rosbag2_py are present), built from source on the humble branches.
RUN mkdir -p ${ROS_WS}/src && cd ${ROS_WS}/src \
 && git clone --depth 1 -b humble https://github.com/ros-perception/perception_pcl.git \
 && git clone --depth 1 -b ros2 https://github.com/ros-perception/pcl_msgs.git \
 && git clone --depth 1 -b humble https://github.com/ANYbotics/grid_map.git _grid_map \
 && git clone --depth 1 -b humble https://github.com/ros2/common_interfaces.git _common_interfaces \
 && git clone --depth 1 -b main https://github.com/DLu/tf_transformations.git \
 && mkdir grid_map && mv _grid_map/grid_map_msgs _grid_map/grid_map_cmake_helpers grid_map/ && rm -rf _grid_map \
 && mv _common_interfaces/sensor_msgs_py . && rm -rf _common_interfaces \
 && cd ${ROS_WS} && . /opt/ros/humble/install/setup.sh \
 && colcon build --merge-install --install-base /opt/ros/humble/install \
      --packages-up-to pcl_ros grid_map_msgs sensor_msgs_py tf_transformations \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
 && rm -rf ${ROS_WS}/build ${ROS_WS}/log

WORKDIR /ws
