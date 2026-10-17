#!/usr/bin/env python3
"""
xrt_reader_test.py -- can several processes read the XRoboToolkit PC service at the same time?

Each instance calls xrt.init(), polls the controllers / headset at --hz for --seconds, optionally calls xrt.close()
early (--close-at), and writes a small JSON summary + every sample to --out. Run two or three of them in parallel.

    python xrt_reader_test.py --name A --seconds 10 --out /tmp/xrt_A.json &
    python xrt_reader_test.py --name B --seconds 10 --out /tmp/xrt_B.json &
"""
import argparse
import json
import time

import numpy as np
import xrobotoolkit_sdk as xrt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--hz", type=float, default=90.0)
    ap.add_argument("--close-at", type=float, default=0.0, help="call xrt.close() at this time and stop (0 = never)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t_init = time.time()
    xrt.init()
    init_s = time.time() - t_init
    samples, errors = [], 0
    t0 = time.time()
    dt = 1.0 / a.hz
    while time.time() - t0 < a.seconds:
        t = time.time()
        if a.close_at and t - t0 >= a.close_at:
            break
        try:
            samples.append({
                "t": t - t0,
                "ts_ns": int(xrt.get_time_stamp_ns()),
                "lt": float(xrt.get_left_trigger()), "rt": float(xrt.get_right_trigger()),
                "lg": float(xrt.get_left_grip()), "rg": float(xrt.get_right_grip()),
                "head": [float(v) for v in xrt.get_headset_pose()],
            })
        except Exception:
            errors += 1
        time.sleep(max(0.0, dt - (time.time() - t)))
    closed_early = bool(a.close_at and time.time() - t0 >= a.close_at - 0.05)
    xrt.close()

    ts = np.array([s["ts_ns"] for s in samples], dtype=np.int64)
    head = np.array([s["head"] for s in samples]) if samples else np.zeros((0, 7))
    summ = {
        "name": a.name, "init_seconds": round(init_s, 3), "samples": len(samples), "errors": errors,
        "closed_early": closed_early, "run_seconds": round(time.time() - t0, 2),
        "distinct_timestamps": int(len(np.unique(ts))) if len(ts) else 0,
        "ts_first": int(ts[0]) if len(ts) else None, "ts_last": int(ts[-1]) if len(ts) else None,
        "head_moved_m": float(np.linalg.norm(head[:, :3].max(0) - head[:, :3].min(0))) if len(head) else 0.0,
        "trigger_left_max": max((s["lt"] for s in samples), default=0.0),
        "trigger_right_max": max((s["rt"] for s in samples), default=0.0),
        "grip_left_max": max((s["lg"] for s in samples), default=0.0),
        "grip_right_max": max((s["rg"] for s in samples), default=0.0),
    }
    json.dump({"summary": summ, "samples": samples}, open(a.out, "w"))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
