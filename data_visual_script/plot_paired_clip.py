import multiprocessing
import numpy as np
import pandas as pd
import glob
import os
import mujoco

import matplotlib
matplotlib.use('Agg') # Offline render backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def do_render():
    # Load MuJoCo Model for Robot FK
    # Need to find the mujoco XML for G1
    model_path = os.path.join("gear_sonic_deploy", "policy", "low_latency", "g1", "g1.xml")
    model_path = 'gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml'
    if not os.path.exists(model_path):
        # fallback search
        import fnmatch
        matches = []
        for root, dirnames, filenames in os.walk('gear_sonic_deploy'):
            for filename in fnmatch.filter(filenames, 'g1.xml'):
                matches.append(os.path.join(root, filename))
        if matches:
            model_path = matches[0]
        else:
            raise FileNotFoundError("Could not find g1.xml")
            
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    # 1. Load Robot CSV Data
    csv_q_path = 'gear_sonic_deploy/logs/g1_deploy_run/q.csv'
    csv_base_quat = 'gear_sonic_deploy/logs/g1_deploy_run/base_quat.csv'

    df_q = pd.read_csv(csv_q_path)
    df_base = pd.read_csv(csv_base_quat)
    
    # Check bounds
    if len(df_q) < 5:
        print("Not enough frames in robot logs!")
        return

    # 30 seconds at 50Hz = 1500 frames. 
    start_idx = 1000
    duration_frames = 1500
    end_idx = min(start_idx + duration_frames, len(df_q))
    
    rq_clip = df_q.iloc[start_idx:end_idx]
    rb_clip = df_base.iloc[start_idx:end_idx]
    clip_time = rq_clip['time_realtime_ms'].values / 1000.0

    print(f"Extracting {len(rq_clip)} frames (30s) starting at {clip_time[0]}")

    # 2. Extract corresponding SMPL frames
    smpl_dir = 'paired_smpl_raw'
    npz_files = sorted(glob.glob(os.path.join(smpl_dir, '*.npz')))
    
    # Build time array
    smpl_file_times = []
    print("Loading SMPL times...")
    for f in npz_files:
        dat = np.load(f)
        smpl_file_times.append(dat['timestamp_realtime'][0])
    smpl_file_times = np.array(smpl_file_times)

    valid_smpl = []
    for t_robot in clip_time:
        idx = np.searchsorted(smpl_file_times, t_robot)
        if idx >= len(npz_files): idx = len(npz_files)-1
        # take nearest 
        valid_smpl.append(npz_files[idx])

    # Pre-compute FK configurations for MuJoCo
    robot_cartesian_points = []
    for i in range(len(rq_clip)):
        # Apply Base Quat
        d.qpos[3:7] = [rb_clip.iloc[i]['base_qw'], rb_clip.iloc[i]['base_qx'], rb_clip.iloc[i]['base_qy'], rb_clip.iloc[i]['base_qz']]
        # Assuming Q starts at dof 7 in standard mujoco state
        q_keys = [col for col in rq_clip.columns if col.startswith('q_')]
        q_vals = rq_clip.iloc[i][q_keys].values
        d.qpos[7:7+len(q_vals)] = q_vals
        mujoco.mj_kinematics(m, d)
        
        # Keep joint Cartesian positions
        # site_xpos or xanchor 
        body_points = []
        for b_id in range(1, m.nbody):  # Skip worldbody 0
            body_points.append(d.xpos[b_id].copy())
        robot_cartesian_points.append(np.array(body_points))
        
    robot_cartesian_points = np.array(robot_cartesian_points)
    
    
    print("Preparing 3D Animation...")
    # 3. Setup Animation Side By Side
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Bone tree for standard SMPL
    smpl_links = [
        (0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10), 
        (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18), 
        (17,19), (18,20), (19,21), (20,22), (21,23)
    ]
    
    # G1 body parent relationships
    robot_links = []
    for j_id in range(m.nbody):
        p_id = m.body_parentid[j_id]
        if p_id != 0 and p_id != j_id:
            # account for skipped wordbody shift (body_id - 1)
            if j_id-1 >= 0 and p_id-1 >= 0:
                robot_links.append((p_id-1, j_id-1))
            
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        # SMPL Plot
        dat = np.load(valid_smpl[frame_idx])
        joints_3d = dat['smpl_joints'].reshape(-1, 3) 
        
        # Orient SMPL upright
        # Usually it needs some rotation (e.g. Z up or Y up). Assume Y-up is standard, let's match robot Z-up
        smpl_z = joints_3d[:,2].copy()
        joints_3d[:,2] = -joints_3d[:,1]
        joints_3d[:,1] = smpl_z
        
        ax1.scatter(joints_3d[:,0], joints_3d[:,1], joints_3d[:,2], c='magenta', s=10)
        for bone in smpl_links:
            pt1, pt2 = joints_3d[bone[0]], joints_3d[bone[1]]
            ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='purple')
            
        ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1); ax1.set_zlim(-1, 1)
        ax1.set_title(f"Human (SMPL) Frame {frame_idx}/{duration_frames}")

        # Robot Plot
        r_pts = robot_cartesian_points[frame_idx]
        ax2.scatter(r_pts[:,0], r_pts[:,1], r_pts[:,2], c='cyan', s=10)
        for bone in robot_links:
            pt1, pt2 = r_pts[bone[0]], r_pts[bone[1]]
            ax2.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='teal')
            
        ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(-1, 1)
        ax2.set_title("G1 Robot (MuJoCo FK)")

    ani = animation.FuncAnimation(fig, update, frames=duration_frames, interval=20)
    out_file = 'skeleton_3d_comparison.gif'
    # High framerates in large gifs can run out of memory inside MPL. Let's step every 2nd frame (25 fps instead of 50)
    frames_downsampled = range(0, duration_frames, 2)
    ani = animation.FuncAnimation(fig, update, frames=frames_downsampled, interval=40)
    
    print(f"Saving to {out_file}...")
    ani.save(out_file, writer='pillow', fps=25)
    print("Done!")

if __name__ == "__main__":
    do_render()
