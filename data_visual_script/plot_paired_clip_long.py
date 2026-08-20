import multiprocessing
import numpy as np
import pandas as pd
import glob
import os
import mujoco

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def do_render():
    model_path = 'gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml'
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    csv_q_path = '/home/grease/g1_deploy_run/q.csv'
    csv_base_quat = '/home/grease/g1_deploy_run/base_quat.csv'
    df_q = pd.read_csv(csv_q_path)
    df_base = pd.read_csv(csv_base_quat)
    
    start_idx = 0
    stride = 3
    max_frames_to_try = 30 * 50
    end_idx = min(start_idx + max_frames_to_try, len(df_q))
    
    rq_clip = df_q.iloc[start_idx:end_idx:stride]
    rb_clip = df_base.iloc[start_idx:end_idx:stride]
    clip_time = rq_clip['time_realtime_ms'].values / 1000.0
    actual_duration = len(rq_clip) * stride / 50.0

    print(f"Extracting {len(rq_clip)} downsampled frames ({actual_duration:.1f}s) starting at {clip_time[0]}")

    smpl_dir = 'paired_smpl_g1_deploy'
    npz_files = sorted(glob.glob(os.path.join(smpl_dir, '*.npz')))
    
    if len(npz_files) == 0:
        print("No paired SMPL data found in paired_smpl_g1_deploy")
        return
        
    print("Loading SMPL times...")
    smpl_file_times = np.zeros(len(npz_files))
    for i, f in enumerate(npz_files):
        dat = np.load(f)
        smpl_file_times[i] = dat['timestamp_realtime'][0]

    valid_smpl = []
    skipped = 0
    for t_robot in clip_time:
        idx = np.searchsorted(smpl_file_times, t_robot)
        if idx >= len(npz_files): idx = len(npz_files)-1
        
        if abs(smpl_file_times[idx] - t_robot) > 1.0:
            skipped += 1
            idx = -1 
        
        valid_smpl.append(idx)

    print(f"Finished mapping times. Frames missing active alignment: {skipped}")

    robot_cartesian_points = []
    print("Computing Robot MuJoCo Forward Kinematics over entire chunk...")
    for i in range(len(rq_clip)):
        d.qpos[3:7] = [rb_clip.iloc[i]['base_qw'], rb_clip.iloc[i]['base_qx'], rb_clip.iloc[i]['base_qy'], rb_clip.iloc[i]['base_qz']]
        q_keys = [col for col in rq_clip.columns if col.startswith('q_')]
        q_vals = rq_clip.iloc[i][q_keys].values
        d.qpos[7:7+len(q_vals)] = q_vals
        mujoco.mj_kinematics(m, d)
        
        body_points = []
        for b_id in range(1, m.nbody):
            body_points.append(d.xpos[b_id].copy())
        robot_cartesian_points.append(np.array(body_points))
        
    robot_cartesian_points = np.array(robot_cartesian_points)
    
    print("Preparing 3D Animation...")
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    smpl_links = [
        (0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10), 
        (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18), 
        (17,19), (18,20), (19,21), (20,22), (21,23)
    ]
    
    robot_links = []
    for j_id in range(m.nbody):
        p_id = m.body_parentid[j_id]
        if p_id != 0 and p_id != j_id:
            if j_id-1 >= 0 and p_id-1 >= 0:
                robot_links.append((p_id-1, j_id-1))
                
    smpl_cache = []
    for s_idx in valid_smpl:
        if s_idx != -1:
            d_smpl = np.load(npz_files[s_idx])
            joints = d_smpl['smpl_joints'].reshape(-1, 3)
            zs = joints[:,2].copy()
            joints[:,2] = -joints[:,1]
            joints[:,1] = zs
            smpl_cache.append(joints)
        else:
            smpl_cache.append(None)
            
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        joints_3d = smpl_cache[frame_idx]
        if joints_3d is not None:
            ax1.scatter(joints_3d[:,0], joints_3d[:,1], joints_3d[:,2], c='magenta', s=4)
            for bone in smpl_links:
                pt1, pt2 = joints_3d[bone[0]], joints_3d[bone[1]]
                ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='purple', linewidth=2)
                
        ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1); ax1.set_zlim(-1, 1)
        ax1.set_axis_off() 
        ax1.set_title(f"Human PICO Stream (SMPL)")

        r_pts = robot_cartesian_points[frame_idx]
        ax2.scatter(r_pts[:,0], r_pts[:,1], r_pts[:,2], c='cyan', s=4)
        for bone in robot_links:
            pt1, pt2 = r_pts[bone[0]], r_pts[bone[1]]
            ax2.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='teal', linewidth=2)
            
        ax2.set_xlim(-0.8, 0.8); ax2.set_ylim(-0.8, 0.8); ax2.set_zlim(-1, 1.2)
        ax2.set_axis_off()
        
        current_s = (frame_idx * stride)/50.0
        ax2.set_title(f"G1 Robot Output ({current_s:.1f}s / {actual_duration:.1f}s)")

    out_file = 'skeleton_3d_comparison_g1_deploy.mp4'
    print(f"Initiating animation render. Output -> {out_file}")
    
    target_fps = 50 // stride
    ani = animation.FuncAnimation(fig, update, frames=len(rq_clip), interval=(1000.0/target_fps))
    
    ani.save(out_file, writer='ffmpeg', fps=target_fps, dpi=120)
    print("Done generating Video!")

if __name__ == "__main__":
    do_render()
