#!/usr/bin/env python3
"""Probe what the G1's Mid-360 DDS topics actually contain.

RUNS ON THE ROBOT. Answers the two questions that decide whether the
ZMQ->ROS bridge (Option 3) can drive FAST-LIO:

  1. Does `rt/utlidar/cloud`'s PointCloud2 carry PER-POINT TIMESTAMPS
     (a field named offset_time / t / time / timestamp) and line/ring ids?
     FAST-LIO's `lidar_type: 1` path expects Livox `CustomMsg`, whose
     CustomPoint has {offset_time, x, y, z, reflectivity, tag, line} --
     we can only synthesize that from PointCloud2 if the per-point time
     is present (otherwise de-skewing is impossible, see
     `dev_notes/heightmap_architecture_analysis.md`).

  2. Is there a LiDAR IMU stream? FAST-LIO is LiDAR-INERTIAL odometry --
     IMU is mandatory. The vendored unitree_sdk2py has no sensor_msgs/Imu
     IDL, so we check both the robot's own `rt/lowstate` IMU (pelvis-
     mounted, needs extrinsics vs. the LiDAR) and any utlidar imu topic.

Usage (on robot):
    python3 g1_probe_lidar_fields.py --iface eth0 --seconds 5
"""
import argparse
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

_PF_NAME = {1: "int8", 2: "uint8", 3: "int16", 4: "uint16",
            5: "int32", 6: "uint32", 7: "float32", 8: "float64"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--domain-id", type=int, default=0)
    ap.add_argument("--cloud-topic", default="rt/utlidar/cloud")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    ChannelFactoryInitialize(args.domain_id, args.iface)

    state = {"n": 0, "printed": False}

    def cloud_cb(msg: PointCloud2_):
        state["n"] += 1
        if state["printed"]:
            return
        state["printed"] = True
        print("=" * 68)
        print(f"TOPIC: {args.cloud_topic}")
        print(f"  frame_id   : {msg.header.frame_id!r}")
        print(f"  stamp      : sec={msg.header.stamp.sec} nanosec={msg.header.stamp.nanosec}")
        print(f"  height x width : {msg.height} x {msg.width}")
        print(f"  point_step : {msg.point_step}   row_step: {msg.row_step}")
        print(f"  is_dense   : {msg.is_dense}   is_bigendian: {msg.is_bigendian}")
        print(f"  data bytes : {len(msg.data)}")
        print("  FIELDS:")
        for f in msg.fields:
            print(f"    - name={f.name!r:<16} offset={f.offset:<3} "
                  f"datatype={_PF_NAME.get(int(f.datatype), f.datatype):<8} count={f.count}")

        names = {f.name for f in msg.fields}
        time_like = names & {"offset_time", "t", "time", "timestamp", "time_offset"}
        line_like = names & {"line", "ring"}
        print()
        print(f"  >> per-point TIME field present? {'YES: ' + str(time_like) if time_like else 'NO'}")
        print(f"  >> per-point LINE/RING present?  {'YES: ' + str(line_like) if line_like else 'NO'}")
        if not time_like:
            print("  >> WARNING: without per-point time, FAST-LIO cannot de-skew;")
            print("     CustomMsg.offset_time would have to be faked (degrades odometry).")
        print("=" * 68)

    sub = ChannelSubscriber(args.cloud_topic, PointCloud2_)
    sub.Init(cloud_cb, 10)

    # Robot's own IMU (pelvis) via LowState -- candidate IMU source for FAST-LIO
    # if the Mid-360's own IMU isn't exposed over DDS.
    imu_state = {"n": 0, "printed": False}
    try:
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HGLowState
        def lowstate_cb(msg):
            imu_state["n"] += 1
            if imu_state["printed"]:
                return
            imu_state["printed"] = True
            imu = msg.imu_state
            print("\n--- rt/lowstate IMU (robot pelvis/torso IMU) ---")
            print(f"  quaternion      : {list(imu.quaternion)}")
            print(f"  gyroscope       : {list(imu.gyroscope)}")
            print(f"  accelerometer   : {list(imu.accelerometer)}")
            print("  (usable for FAST-LIO, but mounted on the ROBOT not the LiDAR ->")
            print("   needs correct extrinsic_T/extrinsic_R in fast_lio's mid360.yaml)")
        sub_ls = ChannelSubscriber("rt/lowstate", HGLowState)
        sub_ls.Init(lowstate_cb, 10)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not subscribe rt/lowstate: {e}")

    t0 = time.time()
    while time.time() - t0 < args.seconds:
        time.sleep(0.2)

    dt = time.time() - t0
    print(f"\nRATES over {dt:.1f}s:  cloud={state['n']/dt:.1f} Hz   "
          f"lowstate={imu_state['n']/dt:.1f} Hz")
    if state["n"] == 0:
        print("!! no cloud messages -- check --cloud-topic / LiDAR switched ON")


if __name__ == "__main__":
    main()
