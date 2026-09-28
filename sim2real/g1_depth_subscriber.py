#!/usr/bin/env python3
"""Receive live D435i depth frames from the G1 (see `g1_depth_publisher.py`).

RUNS ON THE WORKSTATION.

Examples:
    # live colourised view + latency/FPS readout
    .venv_sim/bin/python sim2real/g1_depth_subscriber.py --visualize

    # headless stats only
    .venv_sim/bin/python sim2real/g1_depth_subscriber.py --frames 100

    # record to an .npz for offline heightmap work
    .venv_sim/bin/python sim2real/g1_depth_subscriber.py --frames 300 --save /tmp/g1_depth.npz

    # live median height map in the torso frame, levelled with the MID-360 IMU
    # (needs g1_lidar_publisher.py running on the robot for --imu-port)
    .venv_sim/bin/python sim2real/g1_depth_subscriber.py --height-map --imu-port 5558

Geometry: points are deprojected in the camera optical frame and moved to the
torso frame with `g1_frames.T_TORSO_OPTICAL` (D435 pitch re-calibrated to
~51.3 deg against the MID-360; the URDF's 47.6 deg makes the floor appear to
rise ~10 cm at 3-4 m). Levelling needs gravity, taken from the MID-360 IMU
stream; without it heights are in the (possibly tilted) torso frame.

Keys in the viewer window: q / ESC to quit.
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gear_sonic_deploy" / "scripts"))
import g1_frames  # noqa: E402
from height_map import HeightMapConfig, build_height_map, print_height_map  # noqa: E402

try:
    import lz4.frame as lz4frame
except ImportError:
    lz4frame = None


def decode(msg):
    """Split a payload into (header_dict, depth_uint16_array)."""
    (hdr_len,) = struct.unpack("<I", msg[:4])
    hdr = json.loads(msg[4:4 + hdr_len])
    blob = msg[4 + hdr_len:]
    if hdr["compress"] == "lz4":
        if lz4frame is None:
            raise RuntimeError("frame is lz4-compressed but lz4 is not installed")
        blob = lz4frame.decompress(blob)
    img = np.frombuffer(blob, dtype=hdr["dtype"]).reshape(hdr["shape"])
    return hdr, img


def deproject(img, hdr, stride=4):
    """Depth image -> Nx3 point cloud in the camera optical frame (metres)."""
    k = hdr["intrinsics"]
    scale = hdr["depth_scale"]
    sub = img[::stride, ::stride]
    ys, xs = np.nonzero(sub)
    z = sub[ys, xs].astype(np.float32) * scale
    # undo the striding to get true pixel coords
    px = xs * stride
    py = ys * stride
    x = (px - k["ppx"]) / k["fx"] * z
    y = (py - k["ppy"]) / k["fy"] * z
    return np.stack([x, y, z], axis=1)


LIDAR_HEADER_SIZE = 1280  # g1_lidar_publisher.py / view_lidar_client.py wire format


def parse_lidar_imu_accel(raw, topic=b"lidar_imu"):
    """linear_acceleration (3,) from a g1_lidar_publisher.py IMU message."""
    hdr = json.loads(raw[len(topic):len(topic) + LIDAR_HEADER_SIZE].rstrip(b"\x00"))
    pos = len(topic) + LIDAR_HEADER_SIZE
    sizes = {"f32": 4, "f64": 8, "i32": 4, "i64": 8, "bool": 1}
    for f in hdr["fields"]:
        n = int(np.prod(f["shape"])) * sizes[f["dtype"]]
        if f["name"] == "linear_acceleration":
            return np.frombuffer(raw[pos:pos + n], dtype=np.float32).astype(np.float64)
        pos += n
    return None


def depth_height_map(img, hdr, accel, max_range):
    """Median height map (11x11, 2 m square starting 0.5 m ahead) from one depth frame."""
    pts = g1_frames.clean_depth(deproject(img, hdr, stride=2), max_range)
    if accel is not None:
        pts = g1_frames.depth_to_level(pts, accel, max_range)
        pts[:, 2] -= g1_frames.estimate_floor_z(pts)  # floor = lowest large surface in view
        frame = "levelled, m above floor"
    else:
        pts = g1_frames.transform_points(g1_frames.T_TORSO_OPTICAL, pts)
        frame = "torso frame, NOT levelled: no IMU"
    cfg = HeightMapConfig(grid_size=11, half_extent=1.0, center_x=1.5, agg="median")
    return build_height_map(pts, cfg), frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.8.122")
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--topic", default="depth")
    ap.add_argument("--frames", type=int, default=0, help="0 = run forever")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--save", help="write received frames to this .npz")
    ap.add_argument("--height-map", action="store_true",
                    help="print a median height map (torso frame, z up) with the stats")
    ap.add_argument("--imu-port", type=int, default=0,
                    help="g1_lidar_publisher.py port (e.g. 5558) to level with the MID-360 IMU; 0 = off")
    ap.add_argument("--max-range", type=float, default=g1_frames.DEPTH_MAX_RANGE,
                    help="ignore depth beyond this for --height-map (m; 0 = no limit)")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="seconds to wait for the first frame")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, args.topic.encode())
    sock.setsockopt(zmq.RCVHWM, 2)
    sock.setsockopt(zmq.CONFLATE, 0)  # CONFLATE breaks multipart; drop via HWM instead
    sock.connect(f"tcp://{args.host}:{args.port}")
    print(f"[sub] connected tcp://{args.host}:{args.port} topic={args.topic!r}")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    imu_sock = None
    accel = None  # EMA of MID-360 IMU linear_acceleration
    accel_log = []
    if args.imu_port:
        imu_sock = ctx.socket(zmq.SUB)
        imu_sock.setsockopt(zmq.SUBSCRIBE, b"lidar_imu")
        imu_sock.connect(f"tcp://{args.host}:{args.imu_port}")
        print(f"[sub] levelling with MID-360 IMU from tcp://{args.host}:{args.imu_port}")

    saved = []
    last_hdr = None
    n = 0
    t0 = None
    t_stat = time.time()
    fps_count = 0
    first_wait = args.timeout

    while True:
        if not poller.poll(first_wait * 1000):
            print(f"[sub] no frame within {first_wait:.0f}s — is the publisher running "
                  f"on the robot, and is port {args.port} reachable?")
            break
        first_wait = 5.0

        while imu_sock is not None:
            try:
                a = parse_lidar_imu_accel(imu_sock.recv(zmq.NOBLOCK))
            except zmq.Again:
                break
            if a is not None:
                accel = a if accel is None else 0.95 * accel + 0.05 * a
                accel_log.append(a)

        _topic, msg = sock.recv_multipart()
        hdr, img = decode(msg)
        last_hdr = hdr
        now = time.time()
        if t0 is None:
            t0 = now
            k = hdr["intrinsics"]
            print(f"[sub] first frame: {img.shape} {img.dtype} "
                  f"depth_scale={hdr['depth_scale']:.6f} m/unit")
            print(f"[sub] intrinsics: fx={k['fx']:.2f} fy={k['fy']:.2f} "
                  f"ppx={k['ppx']:.2f} ppy={k['ppy']:.2f} model={k['model']}")

        n += 1
        fps_count += 1
        latency_ms = (now - hdr["stamp_host"]) * 1e3

        if args.save is not None:
            saved.append(img.copy())

        if now - t_stat >= 2.0:
            valid = float((img > 0).mean()) * 100.0
            d = img[img > 0]
            rng = (f"{d.min() * hdr['depth_scale']:.2f}-"
                   f"{d.max() * hdr['depth_scale']:.2f} m" if d.size else "no returns")
            print(f"[sub] {fps_count / (now - t_stat):5.1f} fps  "
                  f"latency {latency_ms:6.1f} ms  valid {valid:5.1f}%  {rng}")
            fps_count = 0
            t_stat = now
            if args.height_map:
                grid, frame = depth_height_map(img, hdr, accel, args.max_range)
                print(f"[sub] height map ({frame}), z [m], +x fwd to the right, +y up:")
                print_height_map(grid)

        if args.visualize:
            import cv2
            vis = cv2.applyColorMap(
                cv2.convertScaleAbs(img, alpha=0.03), cv2.COLORMAP_JET)
            cv2.putText(vis, f"seq={hdr['seq']} lat={latency_ms:.0f}ms",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("G1 D435i depth", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        if args.frames and n >= args.frames:
            break

    if t0 is not None:
        print(f"[sub] received {n} frames in {time.time() - t0:.1f}s")

    if args.save and saved and last_hdr is not None:
        np.savez_compressed(
            args.save,
            depth=np.stack(saved),
            depth_scale=last_hdr["depth_scale"],
            intrinsics=json.dumps(last_hdr["intrinsics"]),
            T_torso_optical=g1_frames.T_TORSO_OPTICAL,
            imu_accel=np.array(accel_log) if accel_log else np.zeros((0, 3)),
        )
        print(f"[sub] wrote {len(saved)} frames -> {args.save}")

    sock.close(linger=0)


if __name__ == "__main__":
    raise SystemExit(main())
