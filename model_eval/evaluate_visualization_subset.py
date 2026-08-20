import os
import glob
import random
import shutil

source_dir = "/home/grease/gam/data/evaluation_test_set"
vis_dir = "/home/grease/gam/data/evaluation_visualization_set"

# Ensure clean directory
if os.path.exists(vis_dir):
    shutil.rmtree(vis_dir)
os.makedirs(vis_dir)

all_files = sorted(glob.glob(os.path.join(source_dir, "*.pkl")))

if not all_files:
    print(f"Error: No .pkl files found in {source_dir}!")
    exit(1)

# Pick exactly 10 random sequences
random.seed(42)  # For reproducibility
selected_files = random.sample(all_files, min(10, len(all_files)))

print(f"Randomly selected the following {len(selected_files)} motion sequences for visualization:")
for f in selected_files:
    print(f" - {os.path.basename(f)}")
    shutil.copy(f, vis_dir)

print(f"\nSaved visualization subset to {vis_dir}.")
