#!/usr/bin/env python3
"""Grab the next GridMap from each elevation-mapping publisher and save npz + png."""
import sys
import numpy as np
import rclpy
from grid_map_msgs.msg import GridMap
from rclpy.node import Node
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from elevation_mapping_cupy.gridmap_utils import decode_multiarray_to_rows_cols

TOPICS = ["/elevation_mapping_node/elevation_map_raw", "/elevation_mapping_node/filtered_elevation_map"]
out_prefix = sys.argv[1] if len(sys.argv) > 1 else "emc"


class Dumper(Node):
    def __init__(self):
        super().__init__("gridmap_dumper")
        self.maps = {}
        for t in TOPICS:
            self.create_subscription(GridMap, t, lambda m, t=t: self.maps.setdefault(t, m), 2)


rclpy.init()
node = Dumper()
end = node.get_clock().now().nanoseconds + 30e9
while len(node.maps) < len(TOPICS) and node.get_clock().now().nanoseconds < end:
    rclpy.spin_once(node, timeout_sec=0.5)

layers = {}
for t, m in node.maps.items():
    for name, arr in zip(m.layers, m.data):
        layers[name] = decode_multiarray_to_rows_cols(name, arr)
    info = m.info
if not layers:
    sys.exit("no GridMap received")

L, cx, cy = info.length_x, info.pose.position.x, info.pose.position.y
np.savez(f"{out_prefix}.npz", resolution=info.resolution, length=L, center=[cx, cy], **layers)
# grid_map convention: row index runs along -x, column index along -y.
extent = [cy + L / 2, cy - L / 2, cx - L / 2, cx + L / 2]  # left=+y, right=-y, bottom=-x, top=+x
names = [n for n in ["elevation", "variance", "inpaint", "min_smooth_filter"] if n in layers]
fig, axs = plt.subplots(1, len(names), figsize=(5.5 * len(names), 5.5), layout="constrained")
for ax, n in zip(np.atleast_1d(axs), names):
    kw = dict(cmap="magma", vmin=0, vmax=0.02) if n == "variance" else dict(cmap="viridis", vmin=-0.1, vmax=1.2)
    im = ax.imshow(layers[n], extent=extent, origin="upper", **kw)
    ax.plot(cy, cx, "r^", ms=8)
    ax.set_title(n); ax.set_xlabel("y (left +) [m]"); ax.set_ylabel("x (fwd) [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.savefig(f"{out_prefix}.png", dpi=90)
e = layers["elevation"]
print(f"saved {out_prefix}.npz/.png  layers={list(layers)}  res={info.resolution} len={L:.2f} center=({cx:.2f},{cy:.2f})")
print(f"elevation: valid {np.isfinite(e).mean()*100:.1f}%  min {np.nanmin(e):.3f}  median {np.nanmedian(e):.3f}  max {np.nanmax(e):.3f}")
node.destroy_node()
rclpy.shutdown()
