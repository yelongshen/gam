#!/usr/bin/env python3
"""record_terrain.py - save the policy-facing `terrain` topic (terrain_publisher.py) to an npz.

For offline checks of the onboard map, e.g. the box test (humanoid-foundation-model
gear_sonic/scripts/compare_box_map.py). Keeps every `--every`-th message for `--seconds`:

  height_grid, height_grid_valid   (T, rows, cols)  terrain_z - torso_z, torso heading frame, [y][x]
  torso_pos, torso_quat            (T, 3), (T, 4)   odometry frame, quat x y z w
  timestamp, map_stamp, pose_stamp (T,)             publisher clock (pose_stamp NaN if not sent)
  recv_time                        (T,)             this machine's clock when received
  grid_origin, grid_resolution     (2,), ()         from the first message

Plain zmq + numpy (runs on the robot's host Python or on a workstation):

    python3 gear_sonic_deploy/perception/record_terrain.py --seconds 60 --out box_run1.npz
    python3 .../record_terrain.py --host 192.168.8.227 ...   # needs TERRAIN_BIND=tcp://0.0.0.0:5559
"""

import argparse
import os
import sys
import time

import numpy as np
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import zmq_packed  # noqa: E402

PER_FRAME = ("height_grid", "height_grid_valid", "torso_pos", "torso_quat")
STAMPS = ("timestamp", "map_stamp", "pose_stamp")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5559)
    ap.add_argument("--topic", default="terrain")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--every", type=int, default=5, help="keep every N-th message (5: 10 Hz of the 50 Hz topic)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, args.topic.encode())
    sock.connect(f"tcp://{args.host}:{args.port}")
    frames = {k: [] for k in PER_FRAME + STAMPS + ("recv_time",)}
    meta, n_seen = {}, 0
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        if not sock.poll(500):
            continue
        raw = sock.recv()
        now = time.time()
        n_seen += 1
        if (n_seen - 1) % args.every:
            continue
        _, f = zmq_packed.unpack(raw, args.topic)
        if not meta:
            meta = {"grid_origin": np.array(f["grid_origin"], np.float32).ravel(),
                    "grid_resolution": np.float32(np.ravel(f["grid_resolution"])[0])}
        for k in PER_FRAME:
            frames[k].append(np.array(f[k]))
        for k in STAMPS:
            frames[k].append(float(np.ravel(f[k])[0]) if k in f else np.nan)
        frames["recv_time"].append(now)
    if not frames["timestamp"]:
        sys.exit(f"no '{args.topic}' messages on tcp://{args.host}:{args.port} in {args.seconds:.0f} s")
    out = {k: np.stack(v) if k in PER_FRAME else np.array(v) for k, v in frames.items()}
    np.savez_compressed(args.out, **out, **meta)
    t = out["timestamp"]
    print(f"{len(t)} of {n_seen} messages over {t[-1] - t[0]:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
