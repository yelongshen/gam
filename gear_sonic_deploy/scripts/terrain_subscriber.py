#!/usr/bin/env python3
"""Test subscriber for perception/terrain_publisher.py (numpy + pyzmq only, no ROS).

Prints rate, map staleness and coverage, and checks the two terrain fields against each other:
height_grid resampled at the VideoMimic window's cell centres must reproduce terrain_height
(negated) wherever both are valid.

    python3 gear_sonic_deploy/scripts/terrain_subscriber.py                 # on the robot
    python3 gear_sonic_deploy/scripts/terrain_subscriber.py --show          # also print the 11x11 window
    python3 gear_sonic_deploy/scripts/terrain_subscriber.py --seconds 600 --record walk_terrain.npz
                                         # save every message (for the walking test; Ctrl-C saves too)
"""
import argparse
import time

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # else numpy's OpenBLAS spins a busy thread per core
import numpy as np  # noqa: E402
import zmq

import zmq_packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp://127.0.0.1:5559")
    ap.add_argument("--topic", default="terrain")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--show", action="store_true", help="print the last VideoMimic window")
    ap.add_argument("--record", help="save every message's fields (plus receive time) to this .npz")
    a = ap.parse_args()

    sock = zmq.Context().socket(zmq.SUB)
    sock.connect(a.connect)
    sock.setsockopt(zmq.SUBSCRIBE, a.topic.encode())
    sock.setsockopt(zmq.RCVTIMEO, 3000)
    msgs, t0 = [], time.time()
    try:
        while time.time() - t0 < a.seconds:
            try:
                raw = sock.recv()
                msgs.append((time.time(), zmq_packed.unpack(raw, a.topic)))   # receive time taken after recv
            except zmq.Again:
                if not msgs:
                    raise SystemExit(f"no '{a.topic}' messages on {a.connect} - is terrain_publisher running?")
                print("[terrain_subscriber] stream paused >3 s", flush=True)
    except KeyboardInterrupt:
        print(f"[terrain_subscriber] interrupted after {len(msgs)} msgs", flush=True)
    if not msgs:
        raise SystemExit("no messages")
    if a.record:
        names = [f["name"] for f in msgs[0][1][0]["fields"]]
        np.savez_compressed(a.record, recv_time=np.array([t for t, _ in msgs]),
                            **{n: np.stack([m[n] for _, (_, m) in msgs]) for n in names})
        print(f"[terrain_subscriber] saved {len(msgs)} msgs -> {a.record}", flush=True)
    t_recv = np.array([t for t, _ in msgs])
    last_h, d = msgs[-1][1]
    print(f"{len(msgs)} msgs in {t_recv[-1]-t_recv[0]:.1f} s -> {(len(msgs)-1)/(t_recv[-1]-t_recv[0]):.1f} Hz; "
          f"fields: {[(f['name'], f['dtype'], f['shape']) for f in last_h['fields']]}")
    stale = np.array([m["timestamp"][0] - m["map_stamp"][0] for _, (_, m) in msgs])
    lat = np.array([tr - m["timestamp"][0] for tr, (_, m) in msgs])
    print(f"map staleness (timestamp - map_stamp): median {np.median(stale)*1000:.0f} ms, max {stale.max()*1000:.0f} ms; "
          f"publish->receive {np.median(lat)*1000:.1f} ms")

    g, gv = d["height_grid"], d["height_grid_valid"]
    w, wv = d["terrain_height"], d["terrain_height_valid"]
    res, org = float(d["grid_resolution"][0]), d["grid_origin"]
    print(f"torso at {np.round(d['torso_pos'], 3)} (robot/odom); grid {gv.mean()*100:.0f}% valid, "
          f"window {wv.mean()*100:.0f}% seen")
    if gv.any():
        print(f"height_grid (terrain z - torso z): median {np.nanmedian(g):+.3f} m "
              f"-> torso ~{-np.nanmedian(g):.3f} m above the dominant surface")
    # cross-check: VideoMimic cell centres -> nearest grid cell
    c = (np.arange(11) - 5) * 0.1
    ix = np.round((c - org[0]) / res).astype(int)
    iy = np.round((c - org[1]) / res).astype(int)
    gs = g[np.ix_(iy, ix)]
    both = wv & gv[np.ix_(iy, ix)]
    if both.any():
        diff = (-gs[both]) - w[both]
        print(f"cross-check on {both.sum()} cells: -height_grid vs terrain_height |diff| median "
              f"{np.median(np.abs(diff))*1000:.1f} mm, max {np.abs(diff).max()*1000:.1f} mm")
    if a.show:
        for yi in range(10, -1, -1):
            print(" ".join(f"{w[yi, xi]:5.2f}" + (" " if wv[yi, xi] else "*") for xi in range(11)))


if __name__ == "__main__":
    main()
