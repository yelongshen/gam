"""record_zmq_raw.py - save a ZMQ topic's raw messages with their receive times, undecoded.

For streams whose decoder is not on the robot, e.g. the deploy runner's `g1_debug` state stream
(msgpack; the Jetson's system Python has no msgpack): decode them offline. Pairs with
record_terrain.py recordings made at the same time through `recv_time` (both this machine's clock).

  python3 perception/record_zmq_raw.py --port 5560 --topic g1_debug --seconds 120 --out ~/g1_debug_run.npz

Saved: data (uint8, all messages concatenated, topic prefix included), offsets (N+1,), recv_time (N,).
"""
import argparse
import time

import numpy as np
import zmq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--topic", default="")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, a.topic.encode())
    sock.setsockopt(zmq.RCVHWM, 100000)
    sock.connect(f"tcp://{a.host}:{a.port}")
    chunks, recv = [], []
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        if not sock.poll(200):
            continue
        while True:
            try:
                raw = sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            chunks.append(raw)
            recv.append(time.time())
    if not chunks:
        raise SystemExit(f"no messages on tcp://{a.host}:{a.port} topic '{a.topic}' in {a.seconds:.0f} s")
    offsets = np.cumsum([0] + [len(c) for c in chunks])
    np.savez(a.out, data=np.frombuffer(b"".join(chunks), np.uint8), offsets=offsets, recv_time=np.array(recv))
    print(f"{len(chunks)} messages over {recv[-1] - recv[0]:.1f} s ({(len(chunks) - 1) / max(recv[-1] - recv[0], 1e-9):.0f} Hz) -> {a.out}")


if __name__ == "__main__":
    main()
