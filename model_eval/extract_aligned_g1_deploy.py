import numpy as np
import pandas as pd
import glob
import os
import shutil

csv_path = '/home/grease/g1_deploy_run/action.csv'
print(f"Loading {csv_path}...")
df_actions = pd.read_csv(csv_path)

robot_times = df_actions['time_realtime_ms'].values / 1000.0
start_time = robot_times[0]
end_time = robot_times[-1]
buffer_s = 2.0

print(f"Robot action log range: {start_time:.3f} to {end_time:.3f} (length: {len(robot_times)})")

smpl_dir = 'logs/smpl_raw' 
npz_files = sorted(glob.glob(os.path.join(smpl_dir, '*.npz')))
print(f"Scanning {len(npz_files)} NPZ records sequentially to avoid multiprocess overhead...")

valid_files = []
# Fast sequential read since multiprocess limits were hanging
start_idx = 0
found_any = False
for i, f in enumerate(npz_files):
    if i % 10000 == 0:
        print(f"Scanned {i} files...")
    try:
        dat = np.load(f)
        t = dat['timestamp_realtime'][0]
        
        # Check alignment
        if start_time - buffer_s <= t <= end_time + buffer_s:
            valid_files.append(f)
            found_any = True
        elif found_any and t > end_time + 10.0:
            # We found the cluster and have now passed it; safe to early exit!
            print("Passed the relevant time window. Early exit.")
            break
            
    except Exception as e:
        continue

print(f"Found {len(valid_files)} matching NPZ files overall.")

target_dir = 'paired_smpl_g1_deploy'
if len(valid_files) > 0:
    os.makedirs(target_dir, exist_ok=True)
    print(f"Copying {len(valid_files)} files to {target_dir}...")
    for f in valid_files:
        shutil.copy(f, target_dir)
    print("Done copying.")
else:
    print("WARNING: No overlapping timestamps found in the SMPL logs for this robot trace!")

