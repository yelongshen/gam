"""Contract test for videomimic_obs.terrain_obs (needs numpy, scipy, torch).

Checks against (1) a line-by-line port of VideoMimic's videomimic_inference_real.cpp
heightmap code, including the Eigen column-major -> torch transpose(1, 2), and
(2) the simulation's grid_pattern(ordering="xy") layout.

    python3 gear_sonic_deploy/scripts/test_videomimic_obs.py
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import videomimic_obs as vo
from scipy.spatial import cKDTree
rng=np.random.default_rng(0)
P=rng.uniform(-3,3,(60000,2)); z=0.3*P[:,0]-0.1*P[:,1]+0.2*(np.hypot(P[:,0]-1.2,P[:,1]+0.4)<0.3)
pts=np.c_[P,z]; torso=np.array([0.7,-0.2,0.9]); yaw=0.6
NX=NY=11; res=0.1; half=NX/2*res
local=np.array([(-half+(i+0.5)*res, -half+(j+0.5)*res) for j in range(NY) for i in range(NX)])
R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]); glob=(R@local.T).T+torso[:2]
tree=cKDTree(pts[:,:2]); heights=np.zeros((NY,NX)); k=0
for j in range(NY):
    for i in range(NX):
        d,ix=tree.query(glob[k],k=3)
        heights[j,i]=0.85 if d[0]>0.15 else torso[2]-np.sum(pts[ix,2]/(d+1e-6))/np.sum(1/(d+1e-6)); k+=1
t=torch.from_numpy(heights.flatten(order="F").astype(np.float32)).reshape(1,NY,NX).transpose(1,2).contiguous()
ref=t[0].numpy(); ours=vo.terrain_obs(pts,torso,yaw)
print("max |ours - C++ port| =", np.abs(ours-ref).max())
flat=vo.terrain_obs(np.c_[P,0.3*P[:,0]],(0,0,0.9),0.0)
print("slope along +x -> varies across obs[5, x]:", np.round(flat[5,[0,5,10]],3), "| constant down obs[y, 5]:", np.round(flat[[0,5,10],5],3))
g=torch.meshgrid(torch.linspace(-0.5,0.5,11),torch.linspace(-0.5,0.5,11),indexing="xy")
sim=(0.9-0.3*g[0].flatten()).view(11,11).numpy()
print("max |ours - sim grid_pattern| =", np.abs(flat-sim).max())
far=vo.terrain_obs(np.c_[P+10,z],(0,0,0.9)); print("no points nearby -> all default:", np.unique(far))
assert np.abs(ours-ref).max()<1e-5 and np.abs(flat-sim).max()<1e-2 and np.all(far==0.85)
print("OK")
