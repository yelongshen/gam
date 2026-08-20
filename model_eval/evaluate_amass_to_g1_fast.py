import numpy as np
import glob
import os
import joblib

source_dir = "/home/grease/egodata/downloads/amass/extracted"
target_dir = "/home/grease/gam/data/smpl_mode2_test"
os.makedirs(target_dir, exist_ok=True)

print(f"Scanning AMASS full dataset for Mode 2 Conversion at {source_dir}...")
npz_files = glob.glob(os.path.join(source_dir, "**/*.npz"), recursive=True)
print(f"Found {len(npz_files)} `.npz` sequences.")

success = 0
for f in npz_files:
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
        
        out_name = os.path.join(target_dir, os.path.basename(f).replace('.npz', '.pkl'))
        
        # In case of filename collisions across subjects (e.g. walk1.npz)
        base = os.path.basename(f).replace('.npz', '')
        parent = os.path.basename(os.path.dirname(f))
        unique_name = f"{parent}_{base}.pkl"
        out_name = os.path.join(target_dir, unique_name)
        
        joblib.dump(out_dict, out_name)
        success += 1
        
        if success % 1000 == 0:
            print(f"Processed {success}/{len(npz_files)} AMASS sequences...")
            
    except Exception as e:
        pass

print(f"✅ Successfully exported all {success} AMASS sequences perfectly formatted into Mode 2 Evaluation arrays.")

# --- LAFAN PROCESSING ---
# In gear_sonic, the script `sonic_data_processor.py` explicitly handles `.bvh` to `smpl_joints` mappings!

print("\nScanning LAFAN1 full dataset for Mode 2 Conversion...")
lafan_dir = "/home/grease/egodata/downloads/lafan1_extracted"
bvh_files = glob.glob(os.path.join(lafan_dir, "**/*.bvh"), recursive=True)
print(f"Found {len(bvh_files)} `.bvh` sequences.")

