import numpy as np
import pandas as pd
import glob
import os
import shutil
import concurrent.futures

csv_path = '/home/grease/g1_deploy_run002/action.csv'
print(f"Loading {csv_path}...")
df_actions = pd.read_csv(csv_path)

robot_times = df_actions['time_realtime_ms'].values / 1000.0
start_time = robot_times[0]
end_time = robot_times[-1]
buffer_s = 2.0

print(f"Robot action log range: {start_time:.3f} to {end_time:.3f} (length: {len(robot_times)})")

smpl_dir = 'logs/raw_smpl' # check if there are matches in both directories
npz_files = sorted(glob.glob(os.path.join('logs/smpl_raw', '*.npz')) + glob.glob(os.path.join('logs/raw_smpl', '*.npz')))
print(f"Scanning {len(npz_files)} total available NPZ records...")

def check_file(f):
    dat = np.load(f)
    t = dat['timestamp_realtime'][0]
    if start_time - buffer_s <= t <= end_time + buffer_s:
        return f
    return None

valid_files = []
# Fast load check using pool
with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
    # Use chunking to speed up mapping over 295k files
    chunk_size = 50000
    for chunk_idx in range(0, len(npz_files), chunk_size):
        subset = npz_files[chunk_idx:chunk_idx+chunk_size]
        results = executor.map(check_file, subset)
        for res in results:
            if res is not None:
                valid_files.append(res)
        if len(valid_files) > 0 and chunk_idx > 0:
             # we probably grabbed what we needed
             print(f"Found matches around chunk {chunk_idx}")

print(f"Found {len(valid_files)} matching NPZ files.")

target_dir = 'paired_smpl_run002'
os.makedirs(target_dir, exist_ok=True)

if len(valid_files) > 0:
    print(f"Copying {len(valid_files)} files to {target_dir}...")
    for f in valid_files:
        shutil.copy(f, target_dir)
    print("Done copying.")
else:
    print("WARNING: No overlapping timestamps found in the SMPL logs for this robot trace!")
