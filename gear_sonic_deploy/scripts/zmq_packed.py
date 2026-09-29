#!/usr/bin/env python3
"""zmq_packed.py
================
Pack / unpack gear_sonic's ZMQ "packed message" format (numpy only):

    [topic prefix] [1280-byte JSON header, null-padded] [fields' raw bytes, in header order]

    header = {"v": 1, "endian": "le", "count": 1,
              "fields": [{"name": "height_grid", "dtype": "f32", "shape": [50, 50]}, ...]}

This is what gear_sonic_deploy's ZMQPackedMessageSubscriber (C++) parses
(src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_packed_message_subscriber.hpp)
and what g1_lidar_publisher.py / view_lidar_client.py use.
"""
import json

import numpy as np

HEADER_SIZE = 1280
_DTYPE_NAMES = {
    np.dtype(np.float32): "f32", np.dtype(np.float64): "f64",
    np.dtype(np.int32): "i32", np.dtype(np.int64): "i64",
    np.dtype(np.int16): "i16", np.dtype(np.int8): "i8", np.dtype(np.uint8): "u8",
    np.dtype(np.bool_): "bool",
}
_NAME_DTYPES = {v: k for k, v in _DTYPE_NAMES.items()}


def pack(topic, fields, count=1):
    """topic: str or bytes; fields: ordered dict name -> numpy array (little-endian on this platform)."""
    topic_b = topic.encode() if isinstance(topic, str) else topic
    header_fields, payload = [], []
    for name, arr in fields.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in _DTYPE_NAMES:
            raise TypeError(f"field {name!r}: unsupported dtype {arr.dtype}")
        header_fields.append({"name": name, "dtype": _DTYPE_NAMES[arr.dtype], "shape": list(arr.shape)})
        payload.append(arr.tobytes())
    header = json.dumps({"v": 1, "endian": "le", "count": count, "fields": header_fields}).encode()
    if len(header) > HEADER_SIZE:
        raise ValueError(f"header is {len(header)} bytes > {HEADER_SIZE}")
    return topic_b + header.ljust(HEADER_SIZE, b"\x00") + b"".join(payload)


def unpack(msg, topic=b""):
    """Inverse of pack(): returns (header dict, {name: numpy array}). Arrays are read-only views."""
    topic_b = topic.encode() if isinstance(topic, str) else topic
    if not msg.startswith(topic_b):
        raise ValueError(f"message does not start with topic {topic_b!r}")
    off = len(topic_b)
    header = json.loads(msg[off:off + HEADER_SIZE].rstrip(b"\x00"))
    if header.get("endian", "le") != "le":
        raise ValueError("big-endian payloads are not supported")
    pos, out = off + HEADER_SIZE, {}
    for f in header["fields"]:
        dt = _NAME_DTYPES[f["dtype"]]
        n = int(np.prod(f["shape"])) * dt.itemsize
        if pos + n > len(msg):
            raise ValueError(f"truncated payload at field {f['name']!r}")
        out[f["name"]] = np.frombuffer(msg, dtype=dt, count=int(np.prod(f["shape"])), offset=pos).reshape(f["shape"])
        pos += n
    return header, out


if __name__ == "__main__":
    a = {"g": np.random.rand(50, 50).astype(np.float32), "m": np.random.rand(50, 50) > 0.5,
         "p": np.arange(3, dtype=np.float64), "t": np.array([1.5])}
    h, b = unpack(pack("terrain", a), "terrain")
    assert all(np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a), "roundtrip mismatch"
    print("roundtrip OK:", [(f["name"], f["dtype"], f["shape"]) for f in h["fields"]])
