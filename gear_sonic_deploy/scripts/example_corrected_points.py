#!/usr/bin/env python3
"""Minimal example: read the robot's LiDAR + depth streams and get CORRECTED points.

Corrected = in the gravity-levelled torso frame (x forward, y left, z up, origin at torso_link),
with the MID-360's upside-down mount, its (0,0,0) no-returns and the D435 extrinsic handled by
g1_frames. From there: a height map (height_map.py) and VideoMimic's 11x11 policy input
(videomimic_obs.py).

Needs on the robot: g1_lidar_publisher.py (port 5558, with or without --fastlio) and
g1_depth_publisher.py (port 5557, with or without --no-points). Runs anywhere with
numpy + pyzmq + lz4 (+ scipy for videomimic_obs):

    python3 gear_sonic_deploy/scripts/example_corrected_points.py --host 192.168.8.227
    python3 gear_sonic_deploy/scripts/example_corrected_points.py --host 127.0.0.1   # on the robot

Levelling uses the MID-360's accelerometer, so it is only right while the robot is roughly still;
for a moving robot use the onboard pipeline (perception/jetson), which gets attitude from DLIO.
"""
import argparse
import json
import struct
import time

import numpy as np
import zmq

import g1_frames
from height_map import HeightMapConfig, build_height_map

try:
    import lz4.frame as lz4frame
except ImportError:
    lz4frame = None

LIDAR_HEADER = 1280
DTYPES = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64, "bool": np.bool_}


def lidar_fields(raw, topic):
    """g1_lidar_publisher.py message -> dict of numpy arrays."""
    hdr = json.loads(raw[len(topic):len(topic) + LIDAR_HEADER].rstrip(b"\x00"))
    pos, out = len(topic) + LIDAR_HEADER, {}
    for f in hdr["fields"]:
        dt = np.dtype(DTYPES[f["dtype"]])
        n = int(np.prod(f["shape"])) * dt.itemsize
        out[f["name"]] = np.frombuffer(raw[pos:pos + n], dtype=dt).reshape(f["shape"])
        pos += n
    return out


def depth_points(part):
    """g1_depth_publisher.py 'depth' message -> (N, 3) points in the camera optical frame."""
    n = struct.unpack("<I", part[:4])[0]
    hdr = json.loads(part[4:4 + n])
    body = part[4 + n:]
    if hdr["compress"] == "lz4":
        body = lz4frame.decompress(body)
    z = np.frombuffer(body, dtype=hdr["dtype"]).reshape(hdr["shape"]).astype(np.float32) * hdr["depth_scale"]
    k = hdr["intrinsics"]
    v, u = np.mgrid[0:z.shape[0], 0:z.shape[1]].astype(np.float32)
    return np.stack([(u - k["ppx"]) / k["fx"] * z, (v - k["ppy"]) / k["fy"] * z, z], axis=-1).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.8.227")
    ap.add_argument("--seconds", type=float, default=3.0, help="how long to collect")
    a = ap.parse_args()

    ctx = zmq.Context()
    lid = ctx.socket(zmq.SUB); lid.connect(f"tcp://{a.host}:5558"); lid.setsockopt(zmq.SUBSCRIBE, b"")
    dep = ctx.socket(zmq.SUB); dep.connect(f"tcp://{a.host}:5557"); dep.setsockopt(zmq.SUBSCRIBE, b"depth")
    poller = zmq.Poller(); poller.register(lid, zmq.POLLIN); poller.register(dep, zmq.POLLIN)

    accel, clouds, depth = None, [], []
    t0 = time.time()
    while time.time() - t0 < a.seconds or accel is None or not clouds or not depth:
        for s, _ in poller.poll(200):
            if s is lid:
                raw = lid.recv()
                if raw.startswith(b"lidar_imu"):
                    accel = lidar_fields(raw, b"lidar_imu")["linear_acceleration"].astype(np.float64)
                elif raw.startswith(b"lidar_cloud"):
                    clouds.append(lidar_fields(raw, b"lidar_cloud")["points"])
            else:
                depth.append(depth_points(dep.recv_multipart()[1]))
        if time.time() - t0 > 30:
            raise SystemExit("no data: are both publishers running on the robot?")

    # 1. corrected points (levelled torso frame)
    lidar = np.vstack([g1_frames.lidar_to_level(c, accel) for c in clouds])
    cam = np.vstack([g1_frames.depth_to_level(d, accel) for d in depth[-5:]])
    floor_z = g1_frames.estimate_floor_z(lidar)
    print(f"{len(clouds)} lidar scans, {len(depth)} depth frames; torso {-floor_z:.3f} m above the floor")

    # 2. height map (median per cell; heights relative to the floor)
    hm = build_height_map(np.vstack([lidar, cam]),
                          HeightMapConfig(grid_size=40, half_extent=2.0, center_x=1.5, agg="median")) - floor_z
    seen_hm = np.isfinite(hm)
    floor_cells = hm[seen_hm][np.abs(hm[seen_hm]) < 0.1]
    print(f"height map 40x40 @ 0.1 m: {seen_hm.mean()*100:.0f}% of cells seen, "
          f"floor cells read {np.median(floor_cells)*100:+.1f} cm")

    # 3. VideoMimic policy input (needs scipy)
    try:
        import videomimic_obs
        obs, seen = videomimic_obs.terrain_obs(np.vstack([lidar, cam]), return_mask=True)
        print(f"VideoMimic window: {seen.mean()*100:.0f}% seen (static robot: front columns only), "
              f"median {np.median(obs[seen]):.3f} vs torso height {-floor_z:.3f}")
    except ImportError:
        print("(scipy not installed: skipping videomimic_obs)")


if __name__ == "__main__":
    main()
