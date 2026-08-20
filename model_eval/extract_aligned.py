import numpy as np
import pandas as pd
import glob
import os
import shutil
import concurrent.futures

csv_path = 'gear_sonic_deploy/logs/g1_deploy_run/action.csv'
df_actions = pd.read_csv(csv_path)

robot_times = df_actions['time_realtime_ms'].values / 1000.0
start_time = robot_times[0]
end_time = robot_times[-1]
buffer_s = 2.0  # Allow 2 seconds of buffer on either end

print(f"Robot action log range: {start_time:.3f} to {end_time:.3f} (length: {len(robot_times)})")

smpl_dir = 'logs/smpl_raw'
npz_files = sorted(glob.glob(os.path.join(smpl_dir, '*.npz')))

def check_file(f):
    t = np.load(f)['timestamp_realtime'][0]
    if start_time - buffer_s <= t <= end_time + buffer_s:
        return f
    return None

print(f"Filtering {len(npz_files)} NPZ files using process pool...")
valid_files = []
# It's fast to load NPZ files concurrently to find the timestamp
with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
    results = executor.map(check_file, npz_files[:150000]) # The first half seems to contain the times we need based on probe
    for res in results:
        if res is not None:
            valid_files.append(res)

print(f"Found {len(valid_files)} matching NPZ files.")

target_dir = 'paired_smpl_raw'
os.makedirs(target_dir, exist_ok=True)
for f in valid_files:
    shutil.copy(f, target_dir)

print(f"Copied {len(valid_files)} to {target_dir}")
