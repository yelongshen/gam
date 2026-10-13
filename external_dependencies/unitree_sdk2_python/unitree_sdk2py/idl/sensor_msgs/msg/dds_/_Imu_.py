"""
  Manually added to match unitree/idl/ros2/Imu_.hpp (sensor_msgs::msg::dds_::Imu_)
  Module: sensor_msgs.msg.dds_
  IDL file: Imu_.idl

"""

from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

# root module import for resolving types
# import sensor_msgs


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Imu_(idl.IdlStruct, typename="sensor_msgs.msg.dds_.Imu_"):
    header: 'unitree_sdk2py.idl.std_msgs.msg.dds_.Header_'
    orientation: 'unitree_sdk2py.idl.geometry_msgs.msg.dds_.Quaternion_'
    orientation_covariance: types.array[types.float64, 9]
    angular_velocity: 'unitree_sdk2py.idl.geometry_msgs.msg.dds_.Vector3_'
    angular_velocity_covariance: types.array[types.float64, 9]
    linear_acceleration: 'unitree_sdk2py.idl.geometry_msgs.msg.dds_.Vector3_'
    linear_acceleration_covariance: types.array[types.float64, 9]
