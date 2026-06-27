#!/usr/bin/env python3
"""
Replay a reference SMPL motion sequence to the deployed Sonic model via ZMQ.

Usage:
    python replay_smpl_sequence.py [--motion-dir PATH] [--port PORT] [--host HOST]
                                   [--chunk-size N] [--fps N] [--loop]

Example (sim):
    python replay_smpl_sequence.py \
        --motion-dir ../../../reference/real_example/squat_001__A359

Example (real robot, deploy running on Jetson):
    python replay_smpl_sequence.py \
        --motion-dir ../../../reference/real_example/squat_001__A359 \
        --host 192.168.123.xx

Prerequisites:
    - deploy.sh (or just run g1_deploy_onnx_ref ...) must already be running
      with --input-type zmq_manager (the default)
    - pip install pyzmq numpy
"""

import argparse
import json
import struct
import sys
import time

import numpy as np

try:
    import zmq
except ImportError:
    print("ERROR: pyzmq not installed.  Run: pip install pyzmq")
    sys.exit(1)

HEADER_SIZE = 1280  # Must match ZMQPackedMessageSubscriber::HEADER_SIZE
ROBOT_HZ = 50       # Policy runs at 50 Hz
CONTROL_DT = 1.0 / ROBOT_HZ


# ---------------------------------------------------------------------------
# ZMQ publisher helpers
# ---------------------------------------------------------------------------

def _pack_message(topic: bytes, header: dict, data: bytes) -> bytes:
    header_json = json.dumps(header).encode("utf-8")
    assert len(header_json) <= HEADER_SIZE, "Header too large — increase HEADER_SIZE"
    header_bytes = header_json + b"\x00" * (HEADER_SIZE - len(header_json))
    return topic + header_bytes + data


def send_command(pub, *, start: bool, stop: bool, planner: bool,
                 delta_heading: float | None = None, verbose: bool = True):
    fields = [
        {"name": "start",   "dtype": "u8",  "shape": [1]},
        {"name": "stop",    "dtype": "u8",  "shape": [1]},
        {"name": "planner", "dtype": "u8",  "shape": [1]},
    ]
    data = struct.pack("BBB", int(start), int(stop), int(planner))
    if delta_heading is not None:
        fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        data += struct.pack("<f", delta_heading)

    header = {"v": 1, "endian": "le", "count": 1, "fields": fields}
    pub.send(_pack_message(b"command", header, data))
    if verbose:
        print(f"[command] start={start} stop={stop} planner={planner}")


def send_pose(pub, joint_pos: np.ndarray, joint_vel: np.ndarray,
              body_quat: np.ndarray, frame_indices: np.ndarray,
              catch_up: bool = False, verbose: bool = False):
    """Send a chunk of N frames over the 'pose' ZMQ topic."""
    N, num_joints = joint_pos.shape
    header = {
        "v": 1, "endian": "le", "count": N,
        "fields": [
            {"name": "joint_pos",    "dtype": "f32", "shape": [N, num_joints]},
            {"name": "joint_vel",    "dtype": "f32", "shape": [N, num_joints]},
            {"name": "body_quat_w",  "dtype": "f32", "shape": [N, 4]},
            {"name": "frame_index",  "dtype": "i64", "shape": [N]},
            {"name": "catch_up",     "dtype": "u8",  "shape": [1]},
        ],
    }
    data = (
        joint_pos.astype(np.float32).tobytes()
        + joint_vel.astype(np.float32).tobytes()
        + body_quat.astype(np.float32).tobytes()
        + frame_indices.astype(np.int64).tobytes()
        + struct.pack("B", 1 if catch_up else 0)
    )
    pub.send(_pack_message(b"pose", header, data))
    if verbose:
        print(f"[pose]  frames {frame_indices[0]}..{frame_indices[-1]}  "
              f"joints={num_joints}  catch_up={catch_up}")


# ---------------------------------------------------------------------------
# Motion data loader
# ---------------------------------------------------------------------------

def load_motion(motion_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load joint_pos, joint_vel, and root body quaternion from a reference
    motion directory.

    Returns:
        joint_pos  – (T, 29) float32  radians
        joint_vel  – (T, 29) float32  rad/s
        root_quat  – (T, 4)  float32  (w, x, y, z)
    """
    import os

    def load_csv(name):
        path = os.path.join(motion_dir, name)
        return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)

    joint_pos = load_csv("joint_pos.csv")   # (T, 29)
    joint_vel = load_csv("joint_vel.csv")   # (T, 29)

    # body_quat.csv has 14 bodies × 4 (w,x,y,z) = 56 columns.
    # We only need body_0 (root) for the pose message: columns 0..3.
    body_quat_all = load_csv("body_quat.csv")   # (T, 56)
    root_quat = body_quat_all[:, :4]            # (T, 4) root w,x,y,z

    T = joint_pos.shape[0]
    print(f"[loader] Loaded {T} frames, {joint_pos.shape[1]} joints from {motion_dir}")
    return joint_pos, joint_vel, root_quat


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

def replay(motion_dir: str, host: str, port: int,
           chunk_size: int, loop: bool, verbose: bool):

    joint_pos, joint_vel, root_quat = load_motion(motion_dir)
    T = joint_pos.shape[0]

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    endpoint = f"tcp://{host}:{port}"
    pub.bind(endpoint)
    print(f"[zmq] Bound publisher to {endpoint}")
    time.sleep(0.5)   # let subscribers connect

    try:
        print("\nPress Ctrl+C to stop.\n")

        pass_num = 0
        while True:
            pass_num += 1
            print(f"=== Pass {pass_num}: replaying {T} frames at {ROBOT_HZ} Hz ===")

            # --- start in streamed-motion mode ---
            send_command(pub, start=True, stop=False, planner=False, verbose=True)
            time.sleep(0.3)

            frame = 0
            while frame < T:
                chunk_end = min(frame + chunk_size, T)
                n = chunk_end - frame

                indices = np.arange(frame, chunk_end, dtype=np.int64)
                send_pose(
                    pub,
                    joint_pos[frame:chunk_end],
                    joint_vel[frame:chunk_end],
                    root_quat[frame:chunk_end],
                    indices,
                    catch_up=False,
                    verbose=verbose,
                )
                frame = chunk_end

                # Sleep for the duration of the chunk so we feed at ~real-time.
                time.sleep(n * CONTROL_DT)

            print(f"[replay] Sequence complete ({T} frames).")

            if not loop:
                break

            print("[replay] Looping...")
            # small gap before next loop
            time.sleep(0.5)

        # --- stop ---
        send_command(pub, start=False, stop=True, planner=False, verbose=True)
        time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[replay] Interrupted — sending stop.")
        send_command(pub, start=False, stop=True, planner=False, verbose=True)
        time.sleep(0.2)

    finally:
        pub.close()
        ctx.term()
        print("[zmq] Closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    REPO_ROOT = "/Users/yelongshen/gam/gear_sonic_deploy"
    DEFAULT_MOTION = f"{REPO_ROOT}/reference/real_example/squat_001__A359"

    parser = argparse.ArgumentParser(
        description="Replay an SMPL reference sequence to the deployed Sonic model via ZMQ."
    )
    parser.add_argument(
        "--motion-dir", default=DEFAULT_MOTION,
        help=f"Path to motion directory (default: {DEFAULT_MOTION})"
    )
    parser.add_argument("--host", default="*",
                        help="ZMQ host to bind (default: * for all interfaces). "
                             "Use 'localhost' if deploy runs on the same machine.")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ port (default: 5556)")
    parser.add_argument("--chunk-size", type=int, default=10,
                        help="Frames per ZMQ message (default: 10 = 0.2 s)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the sequence continuously")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every pose message")
    args = parser.parse_args()

    replay(
        motion_dir=args.motion_dir,
        host=args.host,
        port=args.port,
        chunk_size=args.chunk_size,
        loop=args.loop,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
