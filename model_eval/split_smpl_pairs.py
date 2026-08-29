import os

test_robot_dir = '/home/grease/ego_dataset/smpl_filtered_to_bvh_csv_robot_filtered_test'
train_robot_dir = '/home/grease/ego_dataset/smpl_filtered_to_bvh_csv_robot_filtered_train'

amass_dir = '/home/grease/ego_dataset/amass_smpl_filtered'
lafan1_dir = '/home/grease/ego_dataset/lafan1_smpl_filtered'

out_test_dir = '/home/grease/ego_dataset/amass_lafan1_smpl_filtered_test'
out_train_dir = '/home/grease/ego_dataset/amass_lafan1_smpl_filtered_train'

os.makedirs(out_test_dir, exist_ok=True)
os.makedirs(out_train_dir, exist_ok=True)

test_files = os.listdir(test_robot_dir)
train_files = os.listdir(train_robot_dir)

def resolve_source(f):
    if f.startswith('amass__'):
        return os.path.join(amass_dir, f[len('amass__'):])
    elif f.startswith('lafan1__'):
        return os.path.join(lafan1_dir, f[len('lafan1__'):])
    return None

import logging
missing = 0
found = 0

def process(files, out_dir):
    global missing, found
    for f in files:
        src = resolve_source(f)
        if src and os.path.exists(src):
            dst = os.path.join(out_dir, f)
            if not os.path.exists(dst):
                os.symlink(src, dst)
            found += 1
        else:
            missing += 1

print(f"Processing {len(test_files)} test files...")
process(test_files, out_test_dir)

print(f"Processing {len(train_files)} train files...")
process(train_files, out_train_dir)

print(f"Done! Created symlinks: {found}, Missing source: {missing}")

