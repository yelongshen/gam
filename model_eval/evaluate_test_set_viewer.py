import numpy as np
import glob
import os
import joblib
import torch
import smplx

# Load SMPL framework to geometrically articulate the 72-D pose_aa array for 3D visualization.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

vis_dir = "/home/grease/gam/data/evaluation_visualization_set"
npz_files = sorted(glob.glob(os.path.join(vis_dir, "*.pkl")))

# Attempt to load an SMPL model if present locally, otherwise we can only plot the root translations.
# AMASS purely relies on SMPL parameters!
try:
    smpl = smplx.create(ext='npz', model_type='smpl', gender='neutral', batch_size=1)
except Exception as e:
    print("WARNING: Cannot instantiate smplx local model without the model weights file.")
    print("We will evaluate macro spatial constraints using `transl` paths instead.")
    smpl = None

def plot_transl_path(f, start=0, stride=5):
    dat = joblib.load(f)
    transl = dat['transl'][start::stride] # (N, 3) XYZ
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(transl[:,0], transl[:,1], transl[:,2], marker='o', markersize=2, linestyle='-', linewidth=1, color='b')
    
    # Mark start and end
    ax.scatter(transl[0,0], transl[0,1], transl[0,2], c='green', s=100, label='Start')
    # Use red for the end point
    if len(transl) > 1:
        ax.scatter(transl[-1,0], transl[-1,1], transl[-1,2], c='red', s=100, label='End')
        
    ax.set_title(f"Root Translation Trajectory\n{os.path.basename(f)}")
    ax.legend()
    # Normalize bounds around path
    mean_p = np.mean(transl, axis=0)
    max_range = np.array([transl[:,0].max()-transl[:,0].min(), transl[:,1].max()-transl[:,1].min(), transl[:,2].max()-transl[:,2].min()]).max() / 2.0
    ax.set_xlim(mean_p[0] - max_range, mean_p[0] + max_range)
    ax.set_ylim(mean_p[1] - max_range, mean_p[1] + max_range)
    ax.set_zlim(mean_p[2] - max_range, mean_p[2] + max_range)
    
    plt.savefig(f.replace('.pkl', '_trajectory.png'))
    plt.close()

if smpl is None:
    for file_idx, f in enumerate(npz_files):
        print(f"[{file_idx+1}/10] Plotting spatial origin path for {os.path.basename(f)}...")
        plot_transl_path(f)
