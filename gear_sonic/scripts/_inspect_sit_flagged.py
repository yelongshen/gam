import sys
import numpy as np
sys.path.insert(0, "/home/grease/gam/gear_sonic/scripts")
import chunk_pico_sit_session as C
d = C.load("/home/grease/g1_robot_data/pico_raw/20261002_151150_session.npz")
T = len(d["_t"]); fps = T / d["_t"][-1]
z_raw, v = C.signals(d, fps)
z = C.movmed(z_raw, int(0.5 * fps))
for a, b in ((1100, 1160), (1246, 1262), (1344, 1352), (1366, 1398)):
    print(f"--- {a}-{b} s   (t, z, speed)")
    for s in np.arange(a, b, 1.0):
        i = int(s * fps)
        print(f"  {s:6.0f} z={z[i]:.2f} v={v[i]:.2f}", end="")
        if int(s - a) % 4 == 3:
            print()
    print()
