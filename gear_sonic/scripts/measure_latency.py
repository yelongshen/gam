#!/usr/bin/env python3
"""
measure_latency.py
=====================
End-to-end latency probe for the PICO SMPL -> pico_manager -> g1_deploy_onnx_ref
pipeline.

Subscribes to BOTH:
  - port 5556 topic "pose"     (pico_manager's ZMQ PUB output — the INPUT
                                 the deploy binary consumes)
  - port 5557 topic "g1_debug" (g1_deploy_onnx_ref's ZMQ PUB output — the
                                 OUTPUT reflecting what the policy is tracking)

Both streams carry a comparable physical signal: the calibrated VR 3-point
position (left wrist, right wrist, head), so we can align a moving feature
(e.g. you waving your hand) between the two streams via cross-correlation
to estimate the true end-to-end latency, in milliseconds.

Usage
-----
    conda run -n xr python measure_latency.py --seconds 15

While it's recording, move one hand distinctly (e.g. wave up-down a few
times) so there's a clear, correlatable signal in both streams.

Requirements
------------
  pip install zmq msgpack numpy scipy
"""

import argparse
import threading
import time

import msgpack
import numpy as np
import zmq

# Must match gear_sonic.utils.teleop.zmq.zmq_planner_sender.HEADER_SIZE.
# NOTE: that module's docstring says 1024, but the actual constant is 1280.
HEADER_SIZE = 1280


def _parse_pose_message(raw: bytes, topic: str):
    """Parse pico_manager's custom [topic][1024B JSON header][binary data] format."""
    topic_bytes = topic.encode("utf-8")
    if not raw.startswith(topic_bytes):
        return None
    offset = len(topic_bytes)
    header_bytes = raw[offset : offset + HEADER_SIZE]
    offset += HEADER_SIZE
    try:
        import json

        header_str = header_bytes.rstrip(b"\x00").decode("utf-8")
        header = json.loads(header_str)
    except Exception:
        return None

    fields = header.get("fields", [])
    dtype_map = {"f32": np.float32, "f64": np.float64, "i32": np.int32,
                 "i64": np.int64, "bool": np.bool_}
    data = {}
    pos = offset
    for f in fields:
        name = f["name"]
        dtype = dtype_map.get(f["dtype"], np.float32)
        shape = tuple(f["shape"])
        n_elements = int(np.prod(shape)) if shape else 1
        n_bytes = n_elements * np.dtype(dtype).itemsize
        arr = np.frombuffer(raw[pos : pos + n_bytes], dtype=dtype).reshape(shape)
        data[name] = arr
        pos += n_bytes
    return data


class StreamRecorder:
    """Subscribes to one ZMQ PUB topic and records (wall_time, value) samples."""

    def __init__(self, port: int, topic: str, extract_fn, label: str):
        self.port = port
        self.topic = topic
        self.extract_fn = extract_fn
        self.label = label
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://localhost:{self.port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        socket.setsockopt(zmq.RCVTIMEO, 200)

        while not self._stop.is_set():
            try:
                raw = socket.recv()
            except zmq.Again:
                continue
            except Exception:
                continue
            t = time.time()
            val = self.extract_fn(raw, self.topic)
            if val is not None:
                self.samples.append((t, val))
        socket.close()
        context.term()


def extract_pose_wrist_x(raw: bytes, topic: str):
    """Extract left-wrist x-position from pico_manager's 'pose' message."""
    data = _parse_pose_message(raw, topic)
    if data is None or "vr_position" not in data:
        return None
    vr_pos = np.asarray(data["vr_position"]).flatten()  # (9,) = 3 points * xyz
    if vr_pos.size < 3:
        return None
    return float(vr_pos[0])  # left-wrist x


def extract_debug_wrist_x(raw: bytes, topic: str):
    """Extract left-wrist x-position (vr_3point_position[0]) from g1_deploy's g1_debug message."""
    topic_bytes = topic.encode("utf-8")
    if not raw.startswith(topic_bytes):
        return None
    payload = raw[len(topic_bytes):]
    try:
        data = msgpack.unpackb(payload, raw=False)
    except Exception:
        return None
    if not isinstance(data, dict) or "vr_3point_position" not in data:
        return None
    vr_pos = np.asarray(data["vr_3point_position"]).flatten()
    if vr_pos.size < 3:
        return None
    return float(vr_pos[0])


def resample_and_correlate(
    input_samples: list[tuple[float, float]],
    output_samples: list[tuple[float, float]],
    dt: float = 0.005,
):
    """
    Resample both irregular time-series onto a common uniform time grid,
    then cross-correlate to find the lag (output relative to input) that
    maximizes correlation. Returns lag in seconds (positive = output is
    delayed relative to input).
    """
    if len(input_samples) < 10 or len(output_samples) < 10:
        raise ValueError("Not enough samples collected for correlation.")

    t0 = min(input_samples[0][0], output_samples[0][0])
    t1 = max(input_samples[-1][0], output_samples[-1][0])
    grid = np.arange(t0, t1, dt)

    in_t = np.array([s[0] for s in input_samples])
    in_v = np.array([s[1] for s in input_samples])
    out_t = np.array([s[0] for s in output_samples])
    out_v = np.array([s[1] for s in output_samples])

    in_resampled = np.interp(grid, in_t, in_v)
    out_resampled = np.interp(grid, out_t, out_v)

    # Remove DC offset for correlation
    in_resampled = in_resampled - in_resampled.mean()
    out_resampled = out_resampled - out_resampled.mean()

    corr = np.correlate(out_resampled, in_resampled, mode="full")
    lags = np.arange(-len(in_resampled) + 1, len(out_resampled))
    best_idx = np.argmax(corr)
    best_lag_samples = lags[best_idx]
    best_lag_s = best_lag_samples * dt

    return best_lag_s, corr, lags, dt


def main():
    ap = argparse.ArgumentParser(description="Measure PICO->G1 policy latency")
    ap.add_argument("--pose-port", type=int, default=5556)
    ap.add_argument("--pose-topic", type=str, default="pose")
    ap.add_argument("--debug-port", type=int, default=5557)
    ap.add_argument("--debug-topic", type=str, default="g1_debug")
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    print("=" * 70)
    print("PICO -> G1 Policy Latency Probe")
    print("=" * 70)
    print(f"Input  (pose):     tcp://localhost:{args.pose_port} topic='{args.pose_topic}'")
    print(f"Output (g1_debug): tcp://localhost:{args.debug_port} topic='{args.debug_topic}'")
    print()
    print(f"Recording for {args.seconds:.0f}s.")
    print(">>> MOVE ONE HAND DISTINCTLY (wave up-down a few times) during this window! <<<")
    print()

    input_rec = StreamRecorder(args.pose_port, args.pose_topic, extract_pose_wrist_x, "input")
    output_rec = StreamRecorder(args.debug_port, args.debug_topic, extract_debug_wrist_x, "output")

    input_rec.start()
    output_rec.start()

    start_t = time.time()
    while time.time() - start_t < args.seconds:
        remaining = args.seconds - (time.time() - start_t)
        print(f"\r  Recording... {remaining:4.1f}s remaining "
              f"(input samples: {len(input_rec.samples)}, "
              f"output samples: {len(output_rec.samples)})", end="", flush=True)
        time.sleep(0.2)
    print()

    input_rec.stop()
    output_rec.stop()

    print(f"\nCollected {len(input_rec.samples)} input samples, "
          f"{len(output_rec.samples)} output samples.")

    if len(input_rec.samples) < 10:
        print("❌ No/insufficient 'pose' data — is pico_manager_thread_server.py running "
              "and streaming (headset worn, body tracking active)?")
        return
    if len(output_rec.samples) < 10:
        print("❌ No/insufficient 'g1_debug' data — is g1_deploy_onnx_ref running "
              "with ZMQ streaming mode ENABLED (press ENTER after ']')?")
        return

    try:
        lag_s, corr, lags, dt = resample_and_correlate(input_rec.samples, output_rec.samples)
    except ValueError as e:
        print(f"❌ {e}")
        return

    lag_ms = lag_s * 1000.0
    print()
    print("=" * 70)
    if lag_ms < 0:
        print(f"⚠️  Estimated lag is NEGATIVE ({lag_ms:.1f} ms) — this usually means the "
              f"movement signal wasn't distinct enough, or the two streams didn't overlap "
              f"in time. Try again with a clearer, more isolated hand movement.")
    else:
        print(f"✅ ESTIMATED END-TO-END LATENCY: {lag_ms:.1f} ms")
        print(f"   (time from PICO pose message being published to it being reflected")
        print(f"    in the G1 policy's tracked VR target)")
    print("=" * 70)


if __name__ == "__main__":
    main()
