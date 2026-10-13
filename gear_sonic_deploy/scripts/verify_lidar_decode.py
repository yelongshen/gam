#!/usr/bin/env python3
"""Verify decode_pointcloud2_xyz() works against a live Mid-360 frame."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

from g1_lidar_publisher import decode_pointcloud2_xyz

iface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
topic = sys.argv[2] if len(sys.argv) > 2 else "rt/utlidar/cloud_livox_mid360"

ChannelFactoryInitialize(0, iface)
sub = ChannelSubscriber(topic, PointCloud2_)
sub.Init()

print(f"waiting for a frame on {topic} ...", flush=True)
t0 = time.time()
while time.time() - t0 < 8:
    msg = sub.Read()
    if msg is not None:
        print(f"got frame: width={msg.width} point_step={msg.point_step} "
              f"frame_id={msg.header.frame_id}", flush=True)
        xyz = decode_pointcloud2_xyz(msg)
        print(f"DECODE OK -> shape={xyz.shape} dtype={xyz.dtype}", flush=True)
        if xyz.shape[0]:
            print(f"  x range [{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]", flush=True)
            print(f"  y range [{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]", flush=True)
            print(f"  z range [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]", flush=True)
            print(f"  first 3 points:\n{xyz[:3]}", flush=True)
        break
    time.sleep(0.05)
else:
    print("NO FRAME RECEIVED", flush=True)
