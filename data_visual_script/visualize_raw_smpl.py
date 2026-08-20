import numpy as np
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def main():
    smpl_dir = 'logs/raw_smpl'
    npz_files = sorted(glob.glob(os.path.join(smpl_dir, '*.npz')))
    
    if len(npz_files) == 0:
        print(f"No npz files found in {smpl_dir}")
        return
        
    print(f"Found {len(npz_files)} frames. Generating full visualization...")
    
    # We'll step by 2 for faster rendering and reasonable file size
    stride = 2
    
    render_files = npz_files[::stride]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    smpl_links = [
        (0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10), 
        (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18), 
        (17,19), (18,20), (19,21), (20,22), (21,23)
    ]
    
    # Cache SMPL so we don't spam disk reads
    smpl_cache = []
    actual_times = []
    
    print("Pre-loading all frames to cache...")
    for i, f in enumerate(render_files):
        try:
            d_smpl = np.load(f)
            joints = d_smpl['smpl_joints'].reshape(-1, 3)
            actual_times.append(d_smpl['timestamp_realtime'][0])
            # Reorient for typical 3D plot
            zs = joints[:,2].copy()
            joints[:,2] = -joints[:,1]
            joints[:,1] = zs
            smpl_cache.append(joints)
        except Exception as e:
            # Handle corrupted files
            actual_times.append(actual_times[-1] if actual_times else 0)
            smpl_cache.append(smpl_cache[-1] if smpl_cache else np.zeros((24, 3)))
        if i % 1000 == 0 and i > 0:
            print(f"Loaded {i}/{len(render_files)}...")
            
    def update(frame_idx):
        ax.clear()
        joints_3d = smpl_cache[frame_idx]
        t = actual_times[frame_idx]
        
        ax.scatter(joints_3d[:,0], joints_3d[:,1], joints_3d[:,2], c='magenta', s=10)
        for bone in smpl_links:
            pt1, pt2 = joints_3d[bone[0]], joints_3d[bone[1]]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='purple', linewidth=2)
                
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_axis_off() 
        ax.set_title(f"Human SMPL Mocap (logs/raw_smpl)\nFrame {frame_idx*stride} | Timestamp: {t:.3f}")

    out_file = 'raw_smpl_full.mp4'
    print(f"Initiating animation render. Output -> {out_file}")
    
    target_fps = 50 // stride
    ani = animation.FuncAnimation(fig, update, frames=len(render_files), interval=(1000.0/target_fps))
    ani.save(out_file, writer='ffmpeg', fps=target_fps, dpi=120)
    print("Done generating Video!")

if __name__ == "__main__":
    main()
