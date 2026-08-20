import numpy as np
import joblib

# Load offline training data
offline = joblib.load('/home/grease/GR00T-WholeBodyControl/data/smpl_filtered/ab_bicycle_001__A359.pkl')
offline_joints = offline['smpl_joints']  # shape (N, 24, 3)

# Load online live stream data
online = np.load('logs/smpl_raw/pose_000000.npz')
online_joints = online['smpl_joints']    # shape (4, 24, 3)

print("Offline (Training) - First frame, first 3 joints (Pelvis, L-Hip, R-Hip):")
print(offline_joints[0, :3, :])
print("\nOnline (Pico Stream) - First frame, first 3 joints (Pelvis, L-Hip, R-Hip):")
print(online_joints[0, :3, :])

offline_dist = np.linalg.norm(offline_joints[0, 0] - offline_joints[0, 1])
online_dist = np.linalg.norm(online_joints[0, 0] - online_joints[0, 1])
print(f"\nDistance between Pelvis and L-Hip:")
print(f"Offline: {offline_dist:.4f} meters")
print(f"Online:  {online_dist:.4f} meters")
