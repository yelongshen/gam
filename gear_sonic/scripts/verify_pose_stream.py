#!/usr/bin/env python3
"""
verify_pose_stream.py
=======================
Subscribes directly to the 'pose' topic published by
pico_manager_thread_server.py on port 5556, to confirm SMPL/pose data is
actively being streamed (independent of whether g1_deploy_onnx_ref is
running or not).

Usage
-----
    python verify_pose_stream.py --port 5556 --topic pose --seconds 15
"""

import argparse
import time

import msgpack
import numpy as np
import zmq


def main():
    ap = argparse.ArgumentParser(description="Verify pico_manager_thread_server pose stream")
    ap.add_argument("--host", type=str, default="localhost")
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
    last_summary = None
    start_t = time.time()

    try:
        while time.time() - start_t < args.seconds:
            try:
                raw = socket.recv()
            except zmq.Again:
                n_timeouts += 1
                print(f"  [timeout] no message in last 1s (total: {n_timeouts})")
                continue

            n_msgs += 1
            payload = raw
            topic_bytes = args.topic.encode()
            if payload.startswith(topic_bytes):
                payload = payload[len(topic_bytes):]

            try:
                data = msgpack.unpackb(payload, raw=False)
            except Exception as e:
                print(f"  [msg {n_msgs}] unpack failed: {e} (len={len(raw)})")
                continue

            elapsed = time.time() - start_t

            # Try to summarize whatever fields are present
            summary_parts = []
            if isinstance(data, dict):
                for key in ("smpl_pose", "smpl_joints", "vr_position",
                            "body_quat_w", "joint_pos", "frame_index"):
                    if key in data:
                        val = np.array(data[key])
                        summary_parts.append(f"{key} shape={val.shape}")

            changed_note = ""
            summary_str = " | ".join(summary_parts) if summary_parts else str(list(data.keys())[:6] if isinstance(data, dict) else type(data))
            if last_summary is not None and summary_str == last_summary:
                changed_note = " (same shape as last msg)"

            if n_msgs <= 5 or n_msgs % 20 == 0:
                print(f"  [{elapsed:6.2f}s] msg #{n_msgs} | {summary_str}{changed_note}")

            last_summary = summary_str

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_msgs} messages received, {n_timeouts} timeouts, "
          f"over {time.time()-start_t:.1f}s")
    if n_msgs == 0:
        print("❌ NO POSE DATA — pico_manager_thread_server is not publishing "
              "on this topic/port, or it hasn't started streaming yet "
              "(needs A+X on PICO controllers).")
    else:
        rate = n_msgs / (time.time() - start_t)
        print(f"✅ Receiving pose data at ~{rate:.1f} msg/s")
    print("=" * 70)


if __name__ == "__main__":
    main()
