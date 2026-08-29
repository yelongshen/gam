import os
import csv
import shutil
import random
from pathlib import Path

# Paths
source_dir = '/home/grease/ego_dataset/smpl_filtered_to_bvh_csv_robot_filtered/smpl_filtered_to_bvh_csv'
test_dir = '/home/grease/ego_dataset/smpl_filtered_to_bvh_csv_robot_filtered_test'
train_dir = '/home/grease/ego_dataset/smpl_filtered_to_bvh_csv_robot_filtered_train'
csv_path = '/home/grease/gam/model_eval/dataset_categories.csv'

# Set up random seed for reproducibility
random.seed(42)

# Read categories mapping
categories = {}
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    for row in reader:
        if len(row) == 2:
            cat = row[1]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(row[0])

# Select 50 for test, rest to train
test_files = set()
train_files = set()

for cat, files in categories.items():
    # Sort for deterministic selection before shuffle
    files = sorted(files)
    random.shuffle(files)
    
    # Check if category has 50 files
    split_idx = min(50, len(files))
    
    for f in files[:split_idx]:
        test_files.add(f)
        
    for f in files[split_idx:]:
        train_files.add(f)

# Create output directories
os.makedirs(test_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)

# Copy files
print(f"Copying {len(test_files)} files to {test_dir}...")
total_test = 0
for f in test_files:
    src = os.path.join(source_dir, f)
    dst = os.path.join(test_dir, f)
    if os.path.exists(src):
        # use symlinks to save space and time
        if not os.path.exists(dst):
            os.symlink(src, dst)
        total_test += 1

print(f"Copying {len(train_files)} files to {train_dir}...")
total_train = 0
for f in train_files:
    src = os.path.join(source_dir, f)
    dst = os.path.join(train_dir, f)
    if os.path.exists(src):
        # use symlinks to save space and time
        if not os.path.exists(dst):
            os.symlink(src, dst)
        total_train += 1

print(f"Done! Test dataset size: {total_test}, Train dataset size: {total_train}")
