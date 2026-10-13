#!/usr/bin/env python3
"""Quick probe: is anything actually being published on a ZMQ PUB port?"""
import sys
import time

import zmq

port = int(sys.argv[1]) if len(sys.argv) > 1 else 5558
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.connect(f"tcp://localhost:{port}")
s.setsockopt_string(zmq.SUBSCRIBE, "")  # all topics
s.setsockopt(zmq.RCVTIMEO, 1000)

print(f"probing tcp://localhost:{port} for {seconds}s ...", flush=True)
n = 0
t0 = time.time()
while time.time() - t0 < seconds:
    try:
        raw = s.recv()
        n += 1
        print(f"  GOT msg #{n}  len={len(raw)}  prefix={raw[:24]!r}", flush=True)
    except zmq.Again:
        print("  [timeout] no data in last 1s", flush=True)

print(f"TOTAL RECEIVED: {n}", flush=True)
