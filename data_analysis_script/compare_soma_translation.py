import numpy as np
import joblib

# Load offline training data
offline = joblib.load('/home/grease/GR00T-WholeBodyControl/data/smpl_filtered/ab_bicycle_001__A359.pkl')
offline_joints = offline['smpl_joints']  # shape (N, 24, 3)

# Load online live stream data
online = np.load('logs/smpl_raw/pose_000000.npz')
online_joints = online['smpl_joints']    # shape (4, 24, 3)

print("Offline (Training) - Head to L-Foot translation vector (joint 15 to 10):")
offline_diff = offline_joints[0, 15] - offline_joints[0, 10]
print(offline_diff)
print(f"Offline Magnitude: {np.linalg.norm(offline_diff):.4f} m")

print("\nOnline (Pico Stream) - Head to L-Foot translation vector (joint 15 to 10):")
online_diff = online_joints[0, 15] - online_joints[0, 10]
print(online_diff)
print(f"Online Magnitude: {np.linalg.norm(online_diff):.4f} m")

print("\nOffline (Training) - Shoulder Span (L-Shoulder (16) to R-Shoulder (17)):")
off_shoulder = offline_joints[0, 16] - offline_joints[0, 17]
print(off_shoulder)
print(f"Offline Magnitude: {np.linalg.norm(off_shoulder):.4f} m")

print("\nOnline (Pico Stream) - Shoulder Span (L-Shoulder (16) to R-Shoulder (17)):")
on_shoulder = online_joints[0, 16] - online_joints[0, 17]
print(on_shoulder)
print(f"Online Magnitude: {np.linalg.norm(on_shoulder):.4f} m")
