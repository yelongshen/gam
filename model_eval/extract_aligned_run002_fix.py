import numpy as np
import pandas as pd
import glob
import os
import shutil

csv_path = '/home/grease/g1_deploy_run002/action.csv'
print(f"Loading {csv_path}...")
df_actions = pd.read_csv(csv_path)

robot_times = df_actions['time_realtime_ms'].values / 1000.0
start_time = robot_times[0]
end_time = robot_times[-1]

print(f"Robot action log range: {start_time:.3f} to {end_time:.3f} (length: {len(robot_times)})")

# We checked globally earlier, and no SMPL tracking file matches this timestamp roughly representing 38+ minutes gap.
# But just in case, we will search EVERYTHING recursively.

print("Searching all possible NPZ paths for anything matching this timestamp...")
all_npz = glob.glob('/home/grease/gam/**/*.npz', recursive=True) + glob.glob('/home/grease/g1_deploy_run002/**/*.npz', recursive=True)

print(f"Found {len(all_npz)} total NPZ files system-wide.")

def find_times():
    found = 0
    stride = max(1, len(all_npz) // 20000) # probe 20k limit max to avoid full disk read
    for i in range(0, len(all_npz), stride):
        f = all_npz[i]
        try:
            dat = np.load(f)
            t = dat['timestamp_realtime'][0]
            if start_time - 100.0 <= t <= end_time + 100.0:
                found += 1
                print(f"Found match: {t} in {f}")
        except Exception as e:
            pass
    return found

num_found = find_times()
if num_found == 0:
    print("FATAL: No NPZ files globally across the repo matched the timestamps logged by g1_deploy_run002.")
