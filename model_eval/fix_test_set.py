import numpy as np
import glob
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random

vis_dir = "/home/grease/gam/data/evaluation_visualization_set"
npz_files = sorted(glob.glob(os.path.join("/home/grease/gam/data/evaluation_test_set", "LAFAN*.pkl")))

# Only include bones whose endpoints are actually populated by the BVH->SMPL mapping
# (joints 22/23 = hand tips are NOT mapped, so we drop (20,22) and (21,23) to avoid
#  spurious lines shooting to the world origin).
smpl_links = [
    (0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10),
    (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18),
    (17,19), (18,20), (19,21)
]

random.seed(42)
selected = random.sample(npz_files, min(12, len(npz_files)))

lafan_root = "/home/grease/egodata/downloads/lafan1_extracted"

# Porting the pure Forward Kinematics solver natively to avoid object attribute linking errors
def _parse_bvh_manual(bvh_path):
    with open(bvh_path) as f:
        lines = f.read().split('\n')
    joints, offsets, channels, parents = [], {}, {}, {}
    stack, ch_idx, i = [], 0, 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('ROOT ') or line.startswith('JOINT '):
            name = line.split()[1]
            joints.append(name)
            parents[name] = stack[-1] if stack else None
            stack.append(name)
        elif line.startswith('OFFSET') and stack:
            p = line.split()
            offsets[stack[-1]] = np.array([float(p[1]), float(p[2]), float(p[3])])
        elif line.startswith('CHANNELS') and stack:
            p = line.split(); n = int(p[1])
            channels[stack[-1]] = {'start': ch_idx, 'types': p[2:2+n]}
            ch_idx += n
        elif line.startswith('End Site'):  
            i += 1
            while i < len(lines):
                if lines[i].strip() == '}': break
                i += 1
        elif line == '}' and stack:
            stack.pop()
        elif line.strip() == 'MOTION':
            break
        i += 1

    motion_start_line = None
    motion_idx = None
    num_frames = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'MOTION':
            motion_start_line = i
        elif motion_start_line is not None and stripped.startswith('Frames:'):
            num_frames = int(stripped.split()[1])
        elif motion_start_line is not None and stripped.startswith('Frame Time:'):
            motion_idx = i + 1
            break
            
    frame_data = []
    for k in range(motion_idx, motion_idx + num_frames):
        vals = lines[k].split()
        if vals:
            frame_data.append([float(v) for v in vals])
    frame_data = np.array(frame_data, dtype=np.float64)

    def Rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def Ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    def Rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    result = np.zeros((num_frames, 72))
    # Approximation of SOMA BVH names to SMPL index limits
    to_smpl = {
        'Hips': 0, 'LeftUpLeg': 1, 'RightUpLeg': 2, 'Spine': 3, 
        'LeftLeg': 4, 'RightLeg': 5, 'Spine1': 6, 'LeftFoot': 7, 
        'RightFoot': 8, 'Spine2': 9, 'LeftToeBase': 10, 'RightToeBase': 11, 
        'Neck': 12, 'LeftShoulder': 13, 'RightShoulder': 14, 'Head': 15, 
        'LeftArm': 16, 'RightArm': 17, 'LeftForeArm': 18, 'RightForeArm': 19, 
        'LeftHand': 20, 'RightHand': 21
    }
    
    for fi, frame in enumerate(frame_data):
        wp, wr = {}, {}
        for j in joints:
            if j not in channels:
                wp[j] = wp.get(parents.get(j), np.zeros(3)).copy()
                wr[j] = wr.get(parents.get(j), np.eye(3)).copy()
                continue
            ch = channels[j]; start, types = ch['start'], ch['types']
            pos_v = [None, None, None]; rot_a, rot_t = [], []
            for k, t in enumerate(types):
                v = frame[start + k]
                if 'position' in t.lower():
                    pos_v['XYZ'.index(t[0].upper())] = v
                else:
                    rot_a.append(v); rot_t.append(t)
            R = np.eye(3)
            for ang, t in zip(rot_a, rot_t):
                a = np.radians(ang)
                if t == 'Xrotation': R = R @ Rx(a)
                elif t == 'Yrotation': R = R @ Ry(a)
                elif t == 'Zrotation': R = R @ Rz(a)
            parent = parents.get(j)
            if parent is None:
                wp[j] = np.array([v if v is not None else 0.0 for v in pos_v])
                wr[j] = R
            else:
                wp[j] = wp[parent] + wr[parent] @ offsets[j]
                wr[j] = wr[parent] @ R
                
        for bvh_name, smpl_idx in to_smpl.items():
             if bvh_name in wp:
                 result[fi, smpl_idx*3 : smpl_idx*3 + 3] = wp[bvh_name]
                 
    return result

for file_idx, f in enumerate(selected): 
    name = os.path.basename(f)
    print(f"[{file_idx+1}/{len(selected)}] Resolving Forward Kinematics for {name} to 3D MP4...")
    
    bvh_name = name.replace("LAFAN1_", "").replace(".pkl", ".bvh")
    bvh_path = glob.glob(os.path.join(lafan_root, "**", bvh_name), recursive=True)[0]
    
    joints_1d = _parse_bvh_manual(bvh_path)
    if joints_1d is None: continue
    
    joints_3d = joints_1d.reshape(-1, 24, 3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    stride = 4
    max_f = min(400, joints_3d.shape[0])
    render_frames = range(0, max_f, stride)
    
    def update(frame_idx):
        ax.clear()
        pts = joints_3d[frame_idx]

        # LAFAN1 BVH is Y-up; matplotlib 3d is Z-up.
        # Remap so the character stands upright: new_Z = +old_Y (up), new_Y = old_Z (depth)
        old_y = pts[:, 1].copy()
        old_z = pts[:, 2].copy()
        pts[:, 1] = old_z
        pts[:, 2] = old_y
        
        # only scatter populated joints (skip origin points at (0,0,0))
        mask = ~np.all(pts == 0, axis=1)
        ax.scatter(pts[mask,0], pts[mask,1], pts[mask,2], c='magenta', s=5)
        for bone in smpl_links:
            if bone[0] < pts.shape[0] and bone[1] < pts.shape[0]:
                pt1, pt2 = pts[bone[0]], pts[bone[1]]
                # skip if EITHER endpoint is an unpopulated (origin) joint
                if not (np.all(pt1==0) or np.all(pt2==0)):
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='purple', linewidth=1)
                
        center = pts[0]
        ran = 50.0 
        ax.set_xlim(center[0]-ran, center[0]+ran)
        ax.set_ylim(center[1]-ran, center[1]+ran)
        ax.set_zlim(center[2]-ran, center[2]+ran)
        ax.set_axis_off() 
        ax.set_title(f"{name.replace('.pkl', '')}")
        
    out_file = os.path.join(vis_dir, f"{name.replace('.pkl', '')}_skeleton.mp4")
    ani = animation.FuncAnimation(fig, update, frames=render_frames, interval=20)
    ani.save(out_file, writer='ffmpeg', fps=30//stride, dpi=100)
    plt.close()
    print(f"  Saved to {out_file}")
    
print("Evaluation Complete.")
