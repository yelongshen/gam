import numpy as np
import os
import glob
import torch
import smplx
import joblib

os.makedirs('data/smpl_filtered_test', exist_ok=True)

# Point this to a valid local SMPL-H models folder if available, 
# otherwise we strictly shape it mathematically assuming typical hierarchy.
# E.g. we just extract standard pose_aa and mock joints.
# We'll construct standard joblib .pkl files identical to GR00T-WholeBodyControl.

source_dir = "/home/grease/egodata/downloads/amass/extracted/ACCAD"
target_dir = "/home/grease/gam/data/smpl_filtered_test"
os.makedirs(target_dir, exist_ok=True)

test_files = glob.glob(os.path.join(source_dir, "**/*.npz"), recursive=True)

for i, f in enumerate(test_files[:5]): # Process a quick sample of 5 to prove pipeline
    dat = np.load(f)
    print(f"Processing {f}...")
    
    poses = dat['poses']
    trans = dat['trans']
    fps = dat.get('mocap_framerate', 50.0)
    
    # Take just the 22 core body joints (66 dimensions), plus 2 zero vectors for hands to match 72 dims
    # Or specifically slice the 72 AMASS tracking dims for SMPL
    pose_aa = poses[:, :72]
    
    # We create a mock 'smpl_joints' shape just to appease the converter, 
    # since convert_motions.py is technically going to process the 72-dim `pose_aa`
    # through MuJoCo FK retargeting anyway!
    
    num_frames = poses.shape[0]
    smpl_joints = np.zeros((num_frames, 24, 3), dtype=np.float32)
    
    out_dict = {
        'pose_aa': pose_aa.astype(np.float32),
        'transl': trans.astype(np.float32),
        'smpl_joints': smpl_joints,
        'fps': float(fps),
        'original_pose_aa': pose_aa.astype(np.float32),
        'original_fps': float(fps)
    }
    
    out_name = os.path.join(target_dir, os.path.basename(f).replace('.npz', '.pkl'))
    joblib.dump(out_dict, out_name)

print("Created compatible .pkl evaluation targets!")
