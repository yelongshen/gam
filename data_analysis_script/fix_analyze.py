import os
import glob
import numpy as np
import joblib

phuma_dir = "/home/grease/egodata/downloads/testset/PHUMA_extracted/data/g1/animation"
smpl_dir = "/home/grease/GR00T-WholeBodyControl/data/smpl_filtered"

phuma_files = glob.glob(os.path.join(phuma_dir, "*.npy"))

phuma_lengths = []
for f in phuma_files:
    try:
        dat = np.load(f, allow_pickle=True).item()
        phuma_lengths.append(dat['dof_pos'].shape[0])
    except:
        pass

smpl_files = glob.glob(os.path.join(smpl_dir, "*.pkl"))
smpl_lengths = []
for f in smpl_files[:len(phuma_files)]:  # Just sample a matching amount for speed
    try:
        dat = joblib.load(f)
        smpl_lengths.append(dat['joint_pos'].shape[0]) 
    except:
        pass

print("=========================================")
print("📊 EVALUATION DATASETS COMPARISON")
print("=========================================")
print(f"1) PHUMA Benchmark Dataset")
print(f"   -> Total Sequences: {len(phuma_files):,}")
print(f"   -> Format: .npy dictionaries (root_trans, root_ori, dof_pos)")
if phuma_lengths:
    print(f"   -> Sequence Length Limit (Steps/Frames):")
    print(f"        Mean: {np.mean(phuma_lengths):.1f} frames")
    print(f"        Range: {np.min(phuma_lengths)} to {np.max(phuma_lengths)} frames")
print("")
print(f"2) GR00T-WBC smpl_filtered (Training Set)")
print(f"   -> Total Sequences: {len(smpl_files):,}")
print(f"   -> Format: .pkl joblib dictionaries (pose_aa, transl, smpl_joints)")
if smpl_lengths:
    print(f"   -> Sequence Length Limit (Steps/Frames):")
    print(f"        Mean: {np.mean(smpl_lengths):.1f} frames")
    print(f"        Range: {np.min(smpl_lengths)} to {np.max(smpl_lengths)} frames")
print("=========================================")
