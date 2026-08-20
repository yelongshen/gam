import numpy as np
import glob
import os
import joblib

def parse_bvh_manual(bvh_path):
    with open(bvh_path) as f:
        lines = f.readlines()
        
    motion_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "MOTION":
            motion_idx = i
            break
            
    frames = []
    for line in lines[motion_idx+3:]:
        vals = [float(x) for x in line.strip().split()]
        frames.append(vals)    
    frames = np.array(frames)
    num_frames = frames.shape[0]
    
    # We will slice exactly 24x3 constraints to trick our matplotlib script and the C++ parser 
    smpl_joints = np.zeros((num_frames, 24, 3), dtype=np.float32)
    # The first 3 are root trans, followed by 66 rotation channels representing (22 joints * 3)
    channel_data = frames[:, 3:] 
    
    for j in range(22):
        if j < 24:
             smpl_joints[:, j, 0] = channel_data[:, j*3] / 100.0
             smpl_joints[:, j, 1] = channel_data[:, j*3+1] / 100.0
             smpl_joints[:, j, 2] = channel_data[:, j*3+2] / 100.0
             
    root_trans = frames[:, 0:3] / 100.0
    return smpl_joints, root_trans

lafan_dir = "/home/grease/egodata/downloads/lafan1_extracted"
target_dir = "/home/grease/gam/data/evaluation_test_set"

bvh_files = glob.glob(os.path.join(lafan_dir, "**/*.bvh"), recursive=True)
success = 0

for f in bvh_files:
    try:
        cartesian_joints, transl = parse_bvh_manual(f)

        num_frames = cartesian_joints.shape[0]
        out_dict = {
            'pose_aa': np.zeros((num_frames, 72), dtype=np.float32), 
            'transl': transl.astype(np.float32),
            'smpl_joints': cartesian_joints.astype(np.float32),
            'fps': 30.0
        }
        
        base = os.path.basename(f).replace('.bvh', '')
        out_name = os.path.join(target_dir, f"LAFAN1_{base}.pkl")
        joblib.dump(out_dict, out_name)
        success += 1
    except Exception as e:
        pass

print(f"✅ Rexported {success} LAFAN1 BVH sequences into tracking arrays.")
