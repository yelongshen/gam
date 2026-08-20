import numpy as np

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
    print("Raw BVH frames shape:", frames.shape)
    if num_frames := frames.shape[0]:
        print("First 10 values from first frame:", frames[0, :10])
    return frames

f = '/home/grease/egodata/downloads/lafan1_extracted/fightAndSports1_subject4.bvh'
print(f"Testing {f}")
parse_bvh_manual(f)
