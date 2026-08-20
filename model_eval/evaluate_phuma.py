import numpy as np

# Quick mock code representation to show the exact process needed
phuma_file = '/home/grease/egodata/downloads/testset/PHUMA_extracted/data/g1/animation/Ways_to_Stand_Winded_clip1_chunk_0000.npy'
dat = np.load(phuma_file, allow_pickle=True).item()

root_trans = dat['root_trans'] # 3D vector global position
root_ori = dat['root_ori']   # 4D quaternion
dof_pos = dat['dof_pos']     # 29 joints angles target
fps = dat['fps']           # e.g., 30 or 50 hz 

print(f"Number of frames: {dof_pos.shape[0]}, Target FPS: {fps}")
print(f"First frame root trans: {root_trans[0]}")
