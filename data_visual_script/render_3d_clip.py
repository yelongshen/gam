import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation

# Get ~4s clip of paired smpl files
smpl_dir = 'paired_smpl_raw'
npz_files = sorted(glob.glob(os.path.join(smpl_dir, '*.npz')))

render_files = npz_files[1000:1100] # ~ 2 seconds at 50fps

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame_idx):
    ax.clear()
    f = render_files[frame_idx]
    dat = np.load(f)
    # smpl_joints are typical [1, 24, 3] or [24, 3] layout
    joints_3d = dat['smpl_joints'].reshape(-1, 3) 
    
    # Simple kinematic tree plotting for SMPL (assuming standard 24 joint connections)
    # 0 is root (pelvis)
    bone_links = [
        (0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10), 
        (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18), 
        (17,19), (18,20), (19,21), (20,22), (21,23)
    ]
    ax.scatter(joints_3d[:,0], joints_3d[:,1], joints_3d[:,2], c='r', s=10)
    for bone in bone_links:
        pt1 = joints_3d[bone[0]]
        pt2 = joints_3d[bone[1]]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='b')
        
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title(f"SMPL 3D Human Mocap - Frame {frame_idx}")
    
ani = animation.FuncAnimation(fig, update, frames=len(render_files), interval=20)
ani.save('paired_smpl_clip_3d.gif', writer='imagemagick', fps=50)
print("Saved 3D animation to paired_smpl_clip_3d.gif")

