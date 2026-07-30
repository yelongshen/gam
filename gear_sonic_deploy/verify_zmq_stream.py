#!/usr/bin/env python3
"""
verify_zmq_stream.py
=====================
Minimal verification tool: subscribes to the g1_deploy debug output socket
(port 5557 by default) and prints raw message stats to confirm data is
actually being received and changing over time.

This is the simplest possible check that g1_deploy_onnx_ref is receiving
live SMPL/pose data (from PICO via pico_manager_thread_server.py) rather
than just running with stale/zero data.

Usage
-----
    python verify_zmq_stream.py --port 5557 --topic g1_debug --seconds 15
"""

import argparse
import time

import msgpack
import numpy as np
import zmq


def main():
    ap = argparse.ArgumentParser(description="Verify g1_deploy ZMQ debug output stream")
    ap.add_argument("--host", type=str, default="localhost")
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--topic", type=str, default="g1_debug")
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    url = f"tcp://{args.host}:{args.port}"
    socket.connect(url)
    socket.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout per recv

    print("=" * 70)
    print(f"Verifying ZMQ stream from g1_deploy_onnx_ref")
    print(f"  Connecting to: {url}")
    print(f"  Topic: {args.topic}")
    print("=" * 70)

    n_msgs = 0
    n_timeouts = 0
    last_root_pos = None
    start_t = time.time()

    try:
        while time.time() - start_t < args.seconds:
            try:
                raw = socket.recv()
            except zmq.Again:
                n_timeouts += 1
                print(f"  [timeout] no message received in last 1s "
                      f"(total timeouts: {n_timeouts})")
                continue

            n_msgs += 1

            # Message format: "<topic>" + msgpack-encoded payload,
            # strip topic prefix if present
            payload = raw
            topic_bytes = args.topic.encode()
            if payload.startswith(topic_bytes):
                payload = payload[len(topic_bytes):]

            try:
                data = msgpack.unpackb(payload, raw=False)
            except Exception as e:
                print(f"  [msg {n_msgs}] Failed to unpack: {e} "
                      f"(raw len={len(raw)})")
                continue

            # Try to find a root/base position field to track changes
            root_pos = None
            if isinstance(data, dict):
                for key in ("root_pos", "base_pos", "target_root_pos", "qpos"):
                    if key in data:
                        root_pos = np.array(data[key][:3])
                        break

            elapsed = time.time() - start_t
            if n_msgs % 30 == 0 or n_msgs <= 5:
                keys_preview = list(data.keys())[:8] if isinstance(data, dict) else type(data)
                changed = ""
                if root_pos is not None and last_root_pos is not None:
                    delta = np.linalg.norm(root_pos - last_root_pos)
                    changed = f" | root_pos delta={delta:.5f}"
                print(f"  [{elapsed:6.2f}s] msg #{n_msgs} | keys={keys_preview}{changed}")

            if root_pos is not None:
                last_root_pos = root_pos

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("\n" + "=" * 70)
    print(f"SUMMARY: received {n_msgs} messages, {n_timeouts} timeouts, "
          f"over {time.time()-start_t:.1f}s")
    if n_msgs == 0:
        print("❌ NO DATA RECEIVED — g1_deploy_onnx_ref debug socket is not "
              "publishing, or wrong host/port/topic.")
    elif n_msgs > 0:
        rate = n_msgs / (time.time() - start_t)
        print(f"✅ Receiving data at ~{rate:.1f} msg/s — the control loop IS "
              f"actively running and publishing state.")
        print("   (This confirms the deploy binary is alive and looping, but")
        print("    does NOT by itself confirm PICO data specifically — check")
        print("    root_pos delta values above: if they change while you")
        print("    move, that's PICO data driving the robot.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
