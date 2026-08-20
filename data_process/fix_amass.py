"""
AMASS 3D skeleton visualizer.

AMASS `poses` are SMPL(-H) axis-angle parameters (156-dim = 52 joints x 3).
We use the first 72 dims = 24 SMPL body joints and run forward kinematics on
the SMPL kinematic tree with a canonical rest-pose skeleton to recover the
3D joint positions, then animate them to MP4.

Randomly picks 2 sequences from each of the 21 AMASS sub-datasets.
"""
import os
import glob
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

AMASS_ROOT = "/home/grease/egodata/downloads/amass/extracted"
OUT_DIR = "/home/grease/gam/data/evaluation_visualization_set/amass"
os.makedirs(OUT_DIR, exist_ok=True)

# ── SMPL 24-joint kinematic tree ────────────────────────────────────────────
SMPL_PARENTS = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
    13, 14, 16, 17, 18, 19, 20, 21
])

# Canonical rest-pose joint positions (approx neutral SMPL, meters, x=right y=up z=fwd)
SMPL_REST = np.array([
    [ 0.00,  0.00,  0.00],  # 0  pelvis
    [ 0.09, -0.06,  0.00],  # 1  L_hip
    [-0.09, -0.06,  0.00],  # 2  R_hip
    [ 0.00,  0.12,  0.00],  # 3  spine1
    [ 0.09, -0.48,  0.00],  # 4  L_knee
    [-0.09, -0.48,  0.00],  # 5  R_knee
    [ 0.00,  0.26,  0.00],  # 6  spine2
    [ 0.09, -0.88,  0.00],  # 7  L_ankle
    [-0.09, -0.88,  0.00],  # 8  R_ankle
    [ 0.00,  0.38,  0.00],  # 9  spine3
    [ 0.09, -0.92,  0.12],  # 10 L_foot
    [-0.09, -0.92,  0.12],  # 11 R_foot
    [ 0.00,  0.50,  0.00],  # 12 neck
    [ 0.07,  0.42,  0.00],  # 13 L_collar
    [-0.07,  0.42,  0.00],  # 14 R_collar
    [ 0.00,  0.62,  0.00],  # 15 head
    [ 0.17,  0.42,  0.00],  # 16 L_shoulder
    [-0.17,  0.42,  0.00],  # 17 R_shoulder
    [ 0.44,  0.42,  0.00],  # 18 L_elbow
    [-0.44,  0.42,  0.00],  # 19 R_elbow
    [ 0.70,  0.42,  0.00],  # 20 L_wrist
    [-0.70,  0.42,  0.00],  # 21 R_wrist
    [ 0.78,  0.42,  0.00],  # 22 L_hand
    [-0.78,  0.42,  0.00],  # 23 R_hand
], dtype=np.float64)

# Bone offset of each joint relative to its parent in the rest pose
SMPL_OFFSETS = SMPL_REST.copy()
for j in range(1, 24):
    SMPL_OFFSETS[j] = SMPL_REST[j] - SMPL_REST[SMPL_PARENTS[j]]

SMPL_LINKS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
    (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
    (20, 22), (21, 23),
]


def axis_angle_to_matrix(aa):
    """Rodrigues formula: (3,) axis-angle -> (3,3) rotation matrix."""
    theta = np.linalg.norm(aa)
    if theta < 1e-8:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def smpl_fk(pose_aa, trans):
    """pose_aa: (T, 72) axis-angle, trans: (T, 3) -> joints (T, 24, 3)."""
    T = pose_aa.shape[0]
    joints = np.zeros((T, 24, 3))
    for t in range(T):
        gr = [None] * 24  # global rotation matrices
        gp = np.zeros((24, 3))  # global positions
        for j in range(24):
            R = axis_angle_to_matrix(pose_aa[t, j * 3:j * 3 + 3])
            p = SMPL_PARENTS[j]
            if p == -1:
                gr[j] = R
                gp[j] = trans[t]
            else:
                gr[j] = gr[p] @ R
                gp[j] = gp[p] + gr[p] @ SMPL_OFFSETS[j]
        joints[t] = gp
    return joints


def render(joints, title, out_file):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    stride = max(1, joints.shape[0] // 200)
    max_f = min(joints.shape[0], 600)
    frames = range(0, max_f, stride)

    def update(fi):
        ax.clear()
        pts = joints[fi]
        # SMPL is Y-up; matplotlib is Z-up -> new_Z=+Y (up), new_Y=+Z (depth)
        p = pts.copy()
        p[:, 1], p[:, 2] = pts[:, 2], pts[:, 1]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='deepskyblue', s=6)
        for a, b in SMPL_LINKS:
            ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]],
                    [p[a, 2], p[b, 2]], c='navy', linewidth=1.2)
        c = p[0]
        r = 1.0
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_axis_off()
        ax.set_title(title, fontsize=9)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=33)
    fps = 30 // stride if stride < 30 else 20
    ani.save(out_file, writer='ffmpeg', fps=max(5, fps), dpi=100)
    plt.close()


def main():
    subdirs = sorted([d for d in os.listdir(AMASS_ROOT)
                      if os.path.isdir(os.path.join(AMASS_ROOT, d))])
    print(f"Found {len(subdirs)} AMASS sub-datasets.")
    random.seed(42)

    for sd in subdirs:
        npz_files = [f for f in glob.glob(os.path.join(AMASS_ROOT, sd, "**", "*.npz"),
                                          recursive=True)
                     if all(tok not in os.path.basename(f).lower()
                            for tok in ('shape', 'stagei.', 'neutral'))]
        if not npz_files:
            print(f"[{sd}] no npz sequences, skipping.")
            continue
        picks = random.sample(npz_files, min(2, len(npz_files)))
        for f in picks:
            try:
                d = np.load(f)
                if 'poses' not in d.files:
                    print(f"[{sd}] {os.path.basename(f)} has no 'poses', skip.")
                    continue
                poses = d['poses']
                trans = d['trans'] if 'trans' in d.files else np.zeros((poses.shape[0], 3))
                pose_aa = poses[:, :72].astype(np.float64)
                joints = smpl_fk(pose_aa, trans.astype(np.float64))
                rel = os.path.relpath(f, os.path.join(AMASS_ROOT, sd))
                seq = os.path.splitext(rel)[0].replace('_poses', '').replace(os.sep, '_')
                out = os.path.join(OUT_DIR, f"AMASS_{sd}_{seq}_skeleton.mp4")
                render(joints, f"{sd} / {seq}", out)
                print(f"[{sd}] saved {os.path.basename(out)}  ({joints.shape[0]} frames)")
            except Exception as e:
                print(f"[{sd}] FAILED {os.path.basename(f)}: {e}")

    print("AMASS visualization complete.")


if __name__ == "__main__":
    main()
