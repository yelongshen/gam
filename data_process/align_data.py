import numpy as np
import pandas as pd
import glob
import os

csv_path = 'gear_sonic_deploy/logs/g1_deploy_run/action.csv'
print(f"Loading {csv_path}...")
df_actions = pd.read_csv(csv_path)

robot_times = df_actions['time_realtime_ms'].values / 1000.0
start_time = robot_times[0]
end_time = robot_times[-1]
print(f"Robot action log range: {start_time:.3f} to {end_time:.3f} (length: {len(robot_times)})")

smpl_dir = 'logs/smpl_raw'
npz_files = sorted(glob.glob(os.path.join(smpl_dir, '*.npz')))
print(f"Found {len(npz_files)} NPZ files in {smpl_dir}.")

# Let's binary search for the start and end NPZ files to match the robot action log
def get_time(idx):
    return np.load(npz_files[idx])['timestamp_realtime'][0]

print("Scanning bounds...")
left, right = 0, len(npz_files) - 1
best_start_idx = 0
while left <= right:
    mid = (left + right) // 2
    if get_time(mid) < start_time:
        left = mid + 1
    else:
        best_start_idx = mid
        right = mid - 1

left, right = 0, len(npz_files) - 1
best_end_idx = len(npz_files) - 1
while left <= right:
    mid = (left + right) // 2
    if get_time(mid) > end_time:
        best_end_idx = mid
        right = mid - 1
    else:
        left = mid + 1

print(f"Start NPZ index: {best_start_idx} (time: {get_time(best_start_idx):.3f})")
print(f"End NPZ index: {best_end_idx} (time: {get_time(best_end_idx):.3f})")

target_dir = 'paired_smpl_raw'
os.makedirs(target_dir, exist_ok=True)
subset_files = npz_files[max(0, best_start_idx-10) : min(len(npz_files), best_end_idx+10)]
print(f"Found {len(subset_files)} corresponding NPZ files. Need to copy to {target_dir}...")
