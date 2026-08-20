import numpy as np
import glob
import os
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Simple programmatic mapping to generate Cartesian skeleton lines 
# entirely mathematically from the standard 72D SMPL vector without relying on heavyweight
# Trimesh PyOpenGL model loading setups. We extract standard approximate bone vectors.

def get_approx_smpl_joints(pose_aa, trans):
    """
    Very crude/fast forward kinematics just for visual plotting. 
    It doesn't substitute real models, but gives us stick figures based on joint bounds.
    If 'pose_aa' isn't natively parsable by simple matrices, we fall back to plotting the root.
    Wait, in `evaluate_test_set_creator.py` we left `smpl_joints` as zeros for AMASS.
    Let's check if any clips here are LAFAN1 which DO have real 'smpl_joints' populated!
    """
    pass

vis_dir = "/home/grease/gam/data/evaluation_visualization_set"
npz_files = sorted(glob.glob(os.path.join(vis_dir, "*.pkl")))

smpl_links = [
    (0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10), 
    (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18), 
    (17,19), (18,20), (19,21), (20,22), (21,23)
]

for file_idx, f in enumerate(npz_files):
    name = os.path.basename(f)
    print(f"[{file_idx+1}/{len(npz_files)}] Rendering {name} to 3D MP4...")
    
    dat = joblib.load(f)
    
    # LAFAN1 has explicit real smpl_joints saved in our bypass script!
    if 'smpl_joints' in dat and np.abs(dat['smpl_joints']).max() > 0:
        joints_3d = dat['smpl_joints']
    else:
        # AMASS requires full bone transformation trees from pose_aa.
        # We process this using the local PyTorch module in `gear_sonic`.
        import torch
        from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import get_g1_key_frame_poses
        # Or using native smplx.
        import smplx
        try:
             model = smplx.create(ext='npz', model_type='smpl', gender='neutral', batch_size=1)
             poses = torch.tensor(dat['pose_aa'])
             output = model(global_orient=poses[:, :3], body_pose=poses[:, 3:72], transl=torch.tensor(dat['transl']))
             joints_3d = output.joints.detach().numpy()
        except:
             print(f"  -> Skipping {name}: Requires missing SMPL weight files to render pose_aa IK structure!")
             continue

    # Setup rendering
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    stride = 3
    render_frames = range(0, joints_3d.shape[0], stride)
    
    def update(frame_idx):
        ax.clear()
        
        pts = joints_3d[frame_idx]
        
        # Apply transformation if LAFAN vs SMPLx
        pts_z = pts[:, 2].copy()
        pts[:, 2] = -pts[:, 1]
        pts[:, 1] = pts_z
        
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='magenta', s=10)
        for bone in smpl_links:
            if bone[0] < pts.shape[0] and bone[1] < pts.shape[0]:
                pt1, pt2 = pts[bone[0]], pts[bone[1]]
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='purple', linewidth=2)
                
        # Center bounds around translation to keep stick figure stable
        center = pts[0]
        ran = 1.2
        ax.set_xlim(center[0]-ran, center[0]+ran)
        ax.set_ylim(center[1]-ran, center[1]+ran)
        ax.set_zlim(center[2]-ran, center[2]+ran)
        ax.set_axis_off() 
        ax.set_title(f"{name}\nFrame {frame_idx}")
        
    out_file = f.replace('.pkl', '_3d_skeleton.mp4')
    ani = animation.FuncAnimation(fig, update, frames=render_frames, interval=20)
    ani.save(out_file, writer='ffmpeg', fps=50//stride, dpi=120)
    plt.close()
    
print("Evaluation Complete.")
