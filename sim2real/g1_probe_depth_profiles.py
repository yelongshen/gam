#!/usr/bin/env python3
"""List supported RealSense depth stream profiles. Run ON the robot."""
import pyrealsense2 as rs

ctx = rs.context()
devs = ctx.query_devices()
print("devices:", len(devs))
if not devs:
    raise SystemExit("no RealSense device found")

d = devs[0]
print("name:", d.get_info(rs.camera_info.name))
for s in d.sensors:
    nm = s.get_info(rs.camera_info.name)
    seen = set()
    for p in s.get_stream_profiles():
        if p.stream_type() != rs.stream.depth or p.format() != rs.format.z16:
            continue
        v = p.as_video_stream_profile()
        seen.add((v.width(), v.height(), p.fps()))
    if not seen:
        continue
    print("=== sensor:", nm)
    for wh in sorted({(w, h) for w, h, _ in seen}):
        fps = sorted({f for w, h, f in seen if (w, h) == wh})
        print(f"  {wh[0]}x{wh[1]}: fps={fps}")
