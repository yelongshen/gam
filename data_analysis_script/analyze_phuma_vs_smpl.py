import os
import glob
import numpy as np
import joblib

phuma_dir = "/home/grease/egodata/downloads/testset/PHUMA_extracted/data/g1/animation"
smpl_dir = "/home/grease/GR00T-WholeBodyControl/data/smpl_filtered"

print("Scanning PHUMA dataset...")
phuma_files = glob.glob(os.path.join(phuma_dir, "*.npy"))
phuma_count = len(phuma_files)

# Randomly sample some PHUMA files to calculate length statistics
phuma_lengths = []
for f in phuma_files[:500]:
    try:
        dat = np.load(f)
        # NPY files structure for PHUMA is likely [frames, dims] 
        phuma_lengths.append(dat.shape[0])
    except:
        pass

phuma_mean_len = np.mean(phuma_lengths) if phuma_lengths else 0
phuma_min_len = np.min(phuma_lengths) if phuma_lengths else 0
phuma_max_len = np.max(phuma_lengths) if phuma_lengths else 0

print("Scanning GR00T smpl_filtered dataset...")
smpl_files = glob.glob(os.path.join(smpl_dir, "*.pkl"))
smpl_count = len(smpl_files)

smpl_lengths = []
for f in smpl_files[:500]:
    try:
        dat = joblib.load(f)
        # BONES/AMASS smpl_filtered format is dict with 'joint_pos' usually [frames, 29]
        smpl_lengths.append(dat['joint_pos'].shape[0]) 
    except:
        pass

smpl_mean_len = np.mean(smpl_lengths) if smpl_lengths else 0
smpl_min_len = np.min(smpl_lengths) if smpl_lengths else 0
smpl_max_len = np.max(smpl_lengths) if smpl_lengths else 0

print("=========================================")
print("📊 EVALUATION DATASETS COMPARISON")
print("=========================================")
print(f"1) PHUMA Benchmark Dataset")
print(f"   -> Total Sequences: {phuma_count:,}")
print(f"   -> Format: .npy arrays (Robot Specific Retargeted)")
if phuma_lengths:
    print(f"   -> Sequence Length (Steps/Frames):")
    print(f"        Mean: {phuma_mean_len:.1f} frames")
    print(f"        Range: {phuma_min_len} to {phuma_max_len} frames")
print("")
print(f"2) GR00T-WBC smpl_filtered (Training Set)")
print(f"   -> Total Sequences: {smpl_count:,}")
print(f"   -> Format: .pkl joblib dictionaries")
if smpl_lengths:
    print(f"   -> Sequence Length (Steps/Frames):")
    print(f"        Mean: {smpl_mean_len:.1f} frames")
    print(f"        Range: {smpl_min_len} to {smpl_max_len} frames")
print("=========================================")

