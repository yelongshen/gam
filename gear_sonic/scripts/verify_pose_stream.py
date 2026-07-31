#!/usr/bin/env python3
"""
verify_pose_stream.py
=======================
Subscribes directly to the 'pose' topic published by
pico_manager_thread_server.py (default port 5556) to confirm SMPL/pose data is
actually arriving — independent of whether g1_deploy_onnx_ref is running.

This is the primary tool for debugging the ZMQ link between the teleop PC
(publisher) and the robot (subscriber).

Wire format (must match zmq_planner_sender.pack_pose_message):
    [topic bytes][1024-byte JSON header][concatenated binary field data]

Usage
-----
    # On the teleop PC itself (is the publisher alive at all?)
    python verify_pose_stream.py

    # From the ROBOT, pointing at the teleop PC (does traffic cross the network?)
    python verify_pose_stream.py --host 192.168.123.164 --seconds 15
"""

import argparse
import json
import time

import numpy as np
import zmq

# Must match gear_sonic.utils.teleop.zmq.zmq_planner_sender.HEADER_SIZE.
# NOTE: that module's docstring says 1024, but the actual constant is 1280.
HEADER_SIZE = 1280

DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "bool": np.bool_,
}


def parse_pose_message(raw: bytes, topic: str):
    """Parse the [topic][1024B JSON header][binary payload] pose message.

    Returns (fields_dict, error_string). Exactly one will be non-None.
    """
    topic_bytes = topic.encode("utf-8")
    if not raw.startswith(topic_bytes):
        return None, f"topic prefix mismatch (expected '{topic}')"

    offset = len(topic_bytes)
    header_bytes = raw[offset : offset + HEADER_SIZE]
    offset += HEADER_SIZE

    try:
        header = json.loads(header_bytes.rstrip(b"\x00").decode("utf-8"))
    except Exception as e:
        return None, f"header decode failed: {e}"

    data = {}
    pos = offset
    for f in header.get("fields", []):
        dtype = DTYPE_MAP.get(f["dtype"], np.float32)
        shape = tuple(f["shape"])
        n_elem = int(np.prod(shape)) if shape else 1
        n_bytes = n_elem * np.dtype(dtype).itemsize
        buf = raw[pos : pos + n_bytes]
        if len(buf) < n_bytes:
            return None, f"truncated payload on field '{f['name']}'"
        data[f["name"]] = np.frombuffer(buf, dtype=dtype).reshape(shape)
        pos += n_bytes

    return data, None


def main():
    ap = argparse.ArgumentParser(description="Verify pico_manager_thread_server pose stream")
    ap.add_argument("--host", type=str, default="localhost",
                    help="Publisher IP (use the teleop PC's IP when running from the robot)")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--topic", type=str, default="pose")
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    url = f"tcp://{args.host}:{args.port}"
    socket.connect(url)
    socket.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    print("=" * 70)
    print("Verifying pico_manager_thread_server 'pose' stream")
    print(f"  Connecting to: {url}  topic='{args.topic}'")
    print("=" * 70)

    n_msgs = 0
    n_timeouts = 0
    n_parse_errors = 0
    first_frame_index = None
    last_frame_index = None
    prev_wall = None
    intervals = []
    start_t = time.time()

    try:
        while time.time() - start_t < args.seconds:
            try:
                raw = socket.recv()
            except zmq.Again:
                n_timeouts += 1
                print(f"  [timeout] no message in last 1s (total: {n_timeouts})")
                continue

            now = time.time()
            n_msgs += 1
            if prev_wall is not None:
                intervals.append((now - prev_wall) * 1000.0)
            prev_wall = now

            data, err = parse_pose_message(raw, args.topic)
            if err is not None:
                n_parse_errors += 1
                if n_parse_errors <= 3:
                    print(f"  [msg {n_msgs}] PARSE ERROR: {err} (raw len={len(raw)})")
                continue

            if "frame_index" in data:
                fi = np.asarray(data["frame_index"]).flatten()
                if fi.size:
                    if first_frame_index is None:
                        first_frame_index = int(fi[0])
                    last_frame_index = int(fi[-1])

            elapsed = now - start_t
            if n_msgs <= 3 or n_msgs % 50 == 0:
                shapes = " | ".join(
                    f"{k}{tuple(np.asarray(v).shape)}"
                    for k, v in list(data.items())[:5]
                )
                fi_str = f" frame_index={last_frame_index}" if last_frame_index is not None else ""
                print(f"  [{elapsed:6.2f}s] msg #{n_msgs}{fi_str} | {shapes}")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    total_elapsed = time.time() - start_t
    print("\n" + "=" * 70)
    print(f"SUMMARY over {total_elapsed:.1f}s")
    print(f"  messages received : {n_msgs}")
    print(f"  recv timeouts     : {n_timeouts}")
    print(f"  parse errors      : {n_parse_errors}")

    if n_msgs == 0:
        print("\n❌ NO DATA RECEIVED.")
        print("   Checklist:")
        print(f"     1. Is pico_manager running and streaming on {args.host}?")
        print("        (it must be in POSE mode — use --auto_pose)")
        print(f"     2. Is port {args.port} reachable?  nc -vz {args.host} {args.port}")
        print("     3. Firewall on the publisher?  sudo ufw allow 5556/tcp")
        print("     4. Same subnet / correct IP?    ip -4 addr show")
    elif n_parse_errors == n_msgs:
        print("\n⚠️  Messages ARRIVE but NONE parsed — publisher/subscriber format mismatch.")
    else:
        rate = n_msgs / total_elapsed if total_elapsed > 0 else 0.0
        print(f"\n✅ Receiving pose data at ~{rate:.1f} msg/s")
        if intervals:
            arr = np.array(intervals)
            print(f"   inter-message interval: mean {arr.mean():.1f} ms | "
                  f"p95 {np.percentile(arr, 95):.1f} ms | max {arr.max():.1f} ms")
            if arr.max() > 100:
                print("   ⚠️  Large gaps detected (>100ms) — network jitter or publisher stalls.")
        if first_frame_index is not None and last_frame_index is not None:
            span = last_frame_index - first_frame_index
            print(f"   frame_index advanced {first_frame_index} → {last_frame_index} ({span} frames)")
            if span <= 0:
                print("   ⚠️  frame_index is NOT advancing — publisher is sending stale/frozen data.")
    print("=" * 70)


if __name__ == "__main__":
    main()
