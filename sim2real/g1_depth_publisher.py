#!/usr/bin/env python3
"""Publish live D435i depth (and optionally color) from the G1's onboard Jetson over ZMQ.

RUNS ON THE ROBOT. Pair with `g1_depth_subscriber.py` on the workstation.

Wire format — a 2-part ZMQ multipart message per frame:
    part 0: topic bytes (default b"depth")
    part 1: json_header_len (4-byte LE uint32) + json_header + payload_bytes

The JSON header carries everything needed to reconstruct and geometrically
interpret the frame (shape, dtype, depth_scale, intrinsics, timestamps), so the
subscriber needs no prior configuration and can be restarted independently.

Bandwidth note: 640x480 uint16 @ 30 FPS is ~18 MB/s raw, which will not fit over
the robot's Wi-Fi (wlan0). Defaults here use decimation=2 (-> 320x240) + LZ4,
giving roughly 1-3 MB/s. Prefer the wired 192.168.123.x interface for full rate.

Usage (on robot):
    python3 g1_depth_publisher.py --port 5557 --decimate 2 --fps 15
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--bind", default="tcp://0.0.0.0")
    ap.add_argument("--topic", default="depth")
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

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2)          # drop old frames rather than queue
    sock.bind(f"{args.bind}:{args.port}")
    print(f"[pub] bound {args.bind}:{args.port} topic={args.topic!r} "
          f"compress={compress} decimate={args.decimate}", flush=True)

    topic_b = args.topic.encode()
    min_dt = (1.0 / args.send_hz) if args.send_hz > 0 else 0.0
    seq = 0
    sent = 0
    bytes_sent = 0
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

            seq += 1
            sent += 1
            bytes_sent += len(payload)

            if args.stats_every and (now - t_stat) >= args.stats_every:
                dt = now - t_stat
                print(f"[pub] {sent/dt:5.1f} fps  {bytes_sent/dt/1e6:5.2f} MB/s  "
                      f"shape={img.shape} seq={seq}", flush=True)
                sent = 0
                bytes_sent = 0
                t_stat = now
    except KeyboardInterrupt:
        print("\n[pub] stopping", flush=True)
    finally:
        pipeline.stop()
        sock.close(linger=0)


if __name__ == "__main__":
    raise SystemExit(main())
