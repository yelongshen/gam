#!/usr/bin/env python3
"""Publish live D435i depth (and optionally color) from the G1's onboard Jetson over ZMQ.

RUNS ON THE ROBOT. Pair with `g1_depth_subscriber.py` on the workstation.

Wire format — a 2-part ZMQ multipart message per frame, on TWO topics
published from the same PUB socket:

    topic "depth"  (raw depth image):
        part 0: topic bytes (default b"depth")
        part 1: json_header_len (4-byte LE uint32) + json_header + payload_bytes

    topic "points" (organized point cloud, see below):
        part 0: topic bytes (default b"points")
        part 1: json_header_len (4-byte LE uint32) + json_header + payload_bytes

This mirrors the standard realsense-ros two-stage pipeline:

    #   Stage                    Artifact                       Where
    0   Raw depth image          /camera/depth/image_rect_raw    RealSense driver
    1   Organized point cloud    /camera/depth/color/points      RealSense driver

...except this robot does NOT have realsense-ros (realsense2_camera) or a
depth-capable driver installed -- only the raw pyrealsense2 SDK is
available, and the onboard `videohub_pc4` service holds the camera device
for its own (color-only) video streaming, unrelated to depth. So both
stages here are produced directly with pyrealsense2's own SDK primitives:
  - Stage 0 (depth image): `pipeline.wait_for_frames().get_depth_frame()`
  - Stage 1 (point cloud): RealSense SDK's built-in `rs.pointcloud()`
    deprojection (`pc.calculate(depth_frame)` + `get_vertices()`), which
    is the same math realsense-ros itself uses internally to build
    `/camera/depth/color/points` -- just computed and shipped ourselves
    instead of via a ROS2 node.

The JSON header carries everything needed to reconstruct and geometrically
interpret each frame (shape, dtype, depth_scale, intrinsics, timestamps),
so the subscriber needs no prior configuration and can be restarted
independently.

Point cloud shape: the vertices from `rs.pointcloud()` are already
"organized" -- i.e. they come out in the same row-major (height, width)
order as the depth frame they were computed from (post-decimation/
filtering), matching the traditional "organized point cloud" concept in
`/camera/depth/color/points` (one (x, y, z) triple per depth pixel, with
invalid/out-of-range pixels reported as (0, 0, 0)).

Bandwidth note: 640x480 uint16 @ 30 FPS is ~18 MB/s raw, which will not fit over
the robot's Wi-Fi (wlan0). Defaults here use decimation=2 (-> 320x240) + LZ4,
giving roughly 1-3 MB/s for depth. The point cloud is 3x float32 per pixel
(~6x the bytes of the raw depth image before compression), so consider
--no-points or a coarser --decimate if bandwidth constrained.

Usage (on robot):
    python3 g1_depth_publisher.py --port 5557 --decimate 2 --fps 15

    # Disable point cloud publishing (depth image only, lower bandwidth):
    python3 g1_depth_publisher.py --port 5557 --no-points
"""
import argparse
import json
import struct
import sys
import time

import numpy as np
import pyrealsense2 as rs
import zmq

try:
    import lz4.frame as lz4frame
except ImportError:
    lz4frame = None


def build_header(depth_frame, intr, depth_scale, shape, dtype, compress, nbytes, seq):
    return {
        "seq": seq,
        "stamp_host": time.time(),                  # Jetson wall clock, epoch s
        "stamp_frame_ms": depth_frame.get_timestamp(),  # camera clock, ms
        "shape": list(shape),
        "dtype": dtype,
        "compress": compress,
        "nbytes": nbytes,
        "depth_scale": depth_scale,                 # multiply raw units -> metres
        "intrinsics": {
            "width": intr.width, "height": intr.height,
            "fx": intr.fx, "fy": intr.fy,
            "ppx": intr.ppx, "ppy": intr.ppy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        },
    }


def build_points_header(depth_frame, intr, shape, dtype, compress, nbytes, seq):
    """Header for the organized point-cloud message.

    Unlike the depth header, values are already in metres (float32 XYZ),
    so no depth_scale field is needed here.
    """
    return {
        "seq": seq,
        "stamp_host": time.time(),
        "stamp_frame_ms": depth_frame.get_timestamp(),
        "shape": list(shape),          # (height, width, 3) -- organized
        "dtype": dtype,                # "float32"
        "compress": compress,
        "nbytes": nbytes,
        "units": "meters",
        "organized": True,
        "invalid_value": [0.0, 0.0, 0.0],  # RealSense's sentinel for no-return pixels
        "intrinsics": {
            "width": intr.width, "height": intr.height,
            "fx": intr.fx, "fy": intr.fy,
            "ppx": intr.ppx, "ppy": intr.ppy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--bind", default="tcp://0.0.0.0")
    ap.add_argument("--topic", default="depth")
    ap.add_argument("--points-topic", default="points",
                    help="ZMQ topic for the organized point cloud (default: points)")
    ap.add_argument("--no-points", action="store_true",
                    help="Disable point cloud computation/publishing (depth image only)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30, help="camera FPS")
    ap.add_argument("--send-hz", type=float, default=0.0,
                    help="throttle publishing to this rate (0 = every frame)")
    ap.add_argument("--decimate", type=int, default=2,
                    help="RealSense decimation magnitude (1 = off). 2 halves each axis.")
    ap.add_argument("--no-compress", action="store_true", help="send raw uint16")
    ap.add_argument("--hole-filling", action="store_true",
                    help="apply RealSense hole-filling filter (denser, but invents data)")
    ap.add_argument("--stats-every", type=float, default=5.0)
    args = ap.parse_args()

    compress = "none" if (args.no_compress or lz4frame is None) else "lz4"
    if not args.no_compress and lz4frame is None:
        print("[warn] lz4 not available, falling back to raw", file=sys.stderr)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height,
                         rs.format.z16, args.fps)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Post-processing chain. Decimation first (cheapest bandwidth win).
    filters = []
    if args.decimate > 1:
        dec = rs.decimation_filter()
        dec.set_option(rs.option.filter_magnitude, float(args.decimate))
        filters.append(dec)
    if args.hole_filling:
        filters.append(rs.hole_filling_filter())

    # RealSense SDK's built-in deprojection helper -- this is the same
    # depth-to-XYZ math realsense-ros uses internally to build its
    # /camera/depth/color/points topic.
    pointcloud = rs.pointcloud() if not args.no_points else None

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2)          # drop old frames rather than queue
    sock.bind(f"{args.bind}:{args.port}")
    print(f"[pub] bound {args.bind}:{args.port} topic={args.topic!r} "
          f"compress={compress} decimate={args.decimate}", flush=True)
    if not args.no_points:
        print(f"[pub] point cloud enabled, topic={args.points_topic!r} "
              f"(organized, float32 XYZ in meters)", flush=True)

    topic_b = args.topic.encode()
    points_topic_b = args.points_topic.encode()
    min_dt = (1.0 / args.send_hz) if args.send_hz > 0 else 0.0
    seq = 0
    sent = 0
    bytes_sent = 0
    points_bytes_sent = 0
    t_stat = time.time()
    t_last = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if not depth:
                continue

            now = time.time()
            if min_dt and (now - t_last) < min_dt:
                continue
            t_last = now

            for f in filters:
                depth = f.process(depth)

            img = np.asanyarray(depth.as_frame().get_data())  # uint16, raw units
            intr = depth.as_frame().profile.as_video_stream_profile().intrinsics

            raw = img.tobytes()
            payload = lz4frame.compress(raw) if compress == "lz4" else raw

            hdr = build_header(depth, intr, depth_scale, img.shape,
                               str(img.dtype), compress, len(raw), seq)
            hdr_b = json.dumps(hdr).encode()
            sock.send_multipart(
                [topic_b, struct.pack("<I", len(hdr_b)) + hdr_b + payload]
            )

            if pointcloud is not None:
                # Stage 1: organized point cloud via RealSense SDK deprojection.
                # calculate() uses the depth frame's own intrinsics, matching
                # the current (possibly decimated) resolution.
                points = pointcloud.calculate(depth)
                # get_vertices() returns a flat structured array of (x, y, z)
                # float32 triples, in the same row-major pixel order as the
                # depth frame -- i.e. already "organized".
                vtx = np.asanyarray(points.get_vertices()).view(np.float32)
                vtx = vtx.reshape(img.shape[0], img.shape[1], 3)

                pts_raw = vtx.tobytes()
                pts_payload = lz4frame.compress(pts_raw) if compress == "lz4" else pts_raw

                pts_hdr = build_points_header(
                    depth, intr, vtx.shape, str(vtx.dtype), compress, len(pts_raw), seq
                )
                pts_hdr_b = json.dumps(pts_hdr).encode()
                sock.send_multipart(
                    [points_topic_b, struct.pack("<I", len(pts_hdr_b)) + pts_hdr_b + pts_payload]
                )
                points_bytes_sent += len(pts_payload)

            seq += 1
            sent += 1
            bytes_sent += len(payload)

            if args.stats_every and (now - t_stat) >= args.stats_every:
                dt = now - t_stat
                msg = (f"[pub] {sent/dt:5.1f} fps  depth={bytes_sent/dt/1e6:5.2f} MB/s  "
                       f"shape={img.shape} seq={seq}")
                if pointcloud is not None:
                    msg += f"  points={points_bytes_sent/dt/1e6:5.2f} MB/s"
                print(msg, flush=True)
                sent = 0
                bytes_sent = 0
                points_bytes_sent = 0
                t_stat = now
    except KeyboardInterrupt:
        print("\n[pub] stopping", flush=True)
    finally:
        pipeline.stop()
        sock.close(linger=0)


if __name__ == "__main__":
    raise SystemExit(main())

