import numpy as np
import glob
import os
import joblib
import shutil

source_amass = "/home/grease/egodata/downloads/amass/extracted"
target_dir = "/home/grease/gam/data/evaluation_test_set"

print(f"Scanning AMASS full dataset for Mode 2 Conversion at {source_amass}...")
npz_files = glob.glob(os.path.join(source_amass, "**/*.npz"), recursive=True)
print(f"Found {len(npz_files)} AMASS `.npz` sequences overall.")

success = 0
for i, f in enumerate(npz_files):
    try:
        dat = np.load(f, allow_pickle=True)
        poses = dat['poses']
        trans = dat['trans']
        fps = dat.get('mocap_framerate', 50.0)
        
        # 156-dim AMASS to 72-dim SMPL base body
        pose_aa = poses[:, :72]
        num_frames = poses.shape[0]
        smpl_joints = np.zeros((num_frames, 24, 3), dtype=np.float32)
        
        out_dict = {
            'pose_aa': pose_aa.astype(np.float32),
            'transl': trans.astype(np.float32),
            'smpl_joints': smpl_joints,
            'fps': float(fps)
        }
        
        # Unique filename construction
        base = os.path.basename(f).replace('.npz', '')
        parent = os.path.basename(os.path.dirname(f))
        
        # Avoid putting duplicate files (like re-adding ACCAD which is already there)
        unique_name = f"AMASS_{parent}_{base}.pkl"
        out_name = os.path.join(target_dir, unique_name)
        
        if not os.path.exists(out_name):
            joblib.dump(out_dict, out_name)
            success += 1
            
        if success > 0 and success % 5000 == 0:
            print(f"Exported {success} NEW sequences so far...")
            
    except Exception as e:
        pass

print(f"✅ Exported {success} NEW AMASS sequences to {target_dir}.")
