"""
Motion-content classifier for AMASS / LAFAN1 SMPL sequences.

Instead of relying on filenames, this reads the actual 3D joint trajectories
(via forward kinematics) and derives kinematic features to route each sequence
into one of four evaluation categories used in EVAL_METRICS_DRAFT.md:

    - Basic Locomotion          (walk / run / step / stand)
    - Agility / High-Dynamic    (jump / kick / punch / fight)
    - Upper-Body Manipulation   (arms active, legs mostly static)
    - Unstructured / OOD         (crawl / sit / lie / ground / fall)

The rule set operates on physically meaningful, filename-independent features:
    * root_speed        - horizontal pelvis speed (locomotion)
    * root_vspeed_peak  - peak vertical pelvis speed (jump / hop)
    * pelvis_low_frac   - fraction of time pelvis is near the ground (OOD)
    * limb_energy       - mean joint acceleration magnitude (dynamism)
    * upper_lower_ratio - upper-body vs lower-body motion energy (manipulation)
    * airborne_frac     - fraction of time both feet are off the ground (agility)

Usage:
    .venv_sim/bin/python classify_motions.py                 # classify all
    .venv_sim/bin/python classify_motions.py --limit 500     # sample subset
    .venv_sim/bin/python classify_motions.py --save labels.csv
"""
import os
import glob
import csv
import argparse
import numpy as np

AMASS_ROOT = "/home/grease/egodata/downloads/amass/extracted"
LAFAN_ROOT = "/home/grease/egodata/downloads/lafan1_extracted"
FPS = 30.0  # target feature sampling rate

CATEGORIES = [
    "Basic Locomotion",
    "Agility / High-Dynamic",
    "Upper-Body Manipulation",
    "Unstructured / OOD",
]

# ── SMPL kinematic tree (24 body joints) ────────────────────────────────────
SMPL_PARENTS = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
    13, 14, 16, 17, 18, 19, 20, 21
])
SMPL_REST = np.array([
    [0.00, 0.00, 0.00], [0.09, -0.06, 0.00], [-0.09, -0.06, 0.00],
    [0.00, 0.12, 0.00], [0.09, -0.48, 0.00], [-0.09, -0.48, 0.00],
    [0.00, 0.26, 0.00], [0.09, -0.88, 0.00], [-0.09, -0.88, 0.00],
    [0.00, 0.38, 0.00], [0.09, -0.92, 0.12], [-0.09, -0.92, 0.12],
    [0.00, 0.50, 0.00], [0.07, 0.42, 0.00], [-0.07, 0.42, 0.00],
    [0.00, 0.62, 0.00], [0.17, 0.42, 0.00], [-0.17, 0.42, 0.00],
    [0.44, 0.42, 0.00], [-0.44, 0.42, 0.00], [0.70, 0.42, 0.00],
    [-0.70, 0.42, 0.00], [0.78, 0.42, 0.00], [-0.78, 0.42, 0.00],
], dtype=np.float64)
SMPL_OFFSETS = SMPL_REST.copy()
for _j in range(1, 24):
    SMPL_OFFSETS[_j] = SMPL_REST[_j] - SMPL_REST[SMPL_PARENTS[_j]]

# joint groups (SMPL indices)
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]          # hips/knees/ankles/feet
UPPER_JOINTS = [16, 17, 18, 19, 20, 21, 22, 23]     # shoulders/elbows/wrists/hands
FOOT_JOINTS = [10, 11]                               # toe joints (for airborne)
PELVIS = 0


def axis_angle_to_matrix(aa):
    theta = np.linalg.norm(aa)
    if theta < 1e-8:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def smpl_fk(pose_aa, trans):
    T = pose_aa.shape[0]
    joints = np.zeros((T, 24, 3))
    for t in range(T):
        gr = [None] * 24
        gp = np.zeros((24, 3))
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


# ── LAFAN1 BVH -> SMPL-indexed joints ───────────────────────────────────────
BVH_TO_SMPL = {
    'Hips': 0, 'LeftUpLeg': 1, 'RightUpLeg': 2, 'Spine': 3, 'LeftLeg': 4,
    'RightLeg': 5, 'Spine1': 6, 'LeftFoot': 7, 'RightFoot': 8, 'Spine2': 9,
    # NOTE: real LAFAN1 files name these joints 'LeftToe'/'RightToe' (no
    # "Base" suffix) - verified against dance2_subject1.bvh's HIERARCHY
    # section (see RAW_LAFAN1_DATA_FORMAT.md, section 5). The original dict
    # only had the 'Base' spelling, so joints 10/11 were NEVER populated and
    # stayed at [0,0,0] for every frame of every LAFAN1 clip. Keeping both
    # spellings preserves compatibility with any other BVH source that does
    # use the '...Base' naming convention.
    'LeftToeBase': 10, 'LeftToe': 10,
    'RightToeBase': 11, 'RightToe': 11,
    'Neck': 12, 'LeftShoulder': 13,
    'RightShoulder': 14, 'Head': 15, 'LeftArm': 16, 'RightArm': 17,
    'LeftForeArm': 18, 'RightForeArm': 19, 'LeftHand': 20, 'RightHand': 21,
}


def parse_bvh(bvh_path):
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
            channels[stack[-1]] = {'start': ch_idx, 'types': p[2:2 + n]}
            ch_idx += n
        elif line.startswith('End Site'):
            i += 1
            while i < len(lines) and lines[i].strip() != '}':
                i += 1
        elif line == '}' and stack:
            stack.pop()
        elif line == 'MOTION':
            break
        i += 1
    motion_idx = num_frames = None
    seen_motion = False
    for k, line in enumerate(lines):
        s = line.strip()
        if s == 'MOTION':
            seen_motion = True
        elif seen_motion and s.startswith('Frames:'):
            num_frames = int(s.split()[1])
        elif seen_motion and s.startswith('Frame Time:'):
            motion_idx = k + 1
            break
    if motion_idx is None:
        return None
    frame_data = []
    for k in range(motion_idx, min(motion_idx + num_frames, len(lines))):
        vals = lines[k].split()
        if vals:
            frame_data.append([float(v) for v in vals])
    frame_data = np.array(frame_data, dtype=np.float64)

    def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    T = frame_data.shape[0]
    out = np.zeros((T, 24, 3))
    root_name = joints[0]                 # 'Hips' - the ROOT line's own joint
    root_rot_yup = np.zeros((T, 3, 3))    # raw BVH-frame (cm, Y-up) rotation
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
        for bvh_name, smpl_idx in BVH_TO_SMPL.items():
            if bvh_name in wp:
                out[fi, smpl_idx] = wp[bvh_name]
        root_rot_yup[fi] = wr[root_name]
    # LAFAN1 is in cm & Y-up -> convert to meters, Z-up (x, -z, y)
    out = out / 100.0
    out_zup = out.copy()
    out_zup[..., 1] = -out[..., 2]
    out_zup[..., 2] = out[..., 1]
    out = out_zup
    # Re-express the root rotation matrix in the SAME Z-up world basis as
    # the joint positions above. Positions transform as new_vec = P @ old_vec
    # with P = swap(axis 1, axis 2) (no sign flip, see RAW_LAFAN1_DATA_FORMAT.md
    # section 4).
    #
    # IMPORTANT: this is a LEFT-MULTIPLICATION ONLY (R_new = P @ R_old), NOT
    # a similarity/conjugation (P @ R_old @ P^T). We are only RELABELING how
    # WORLD-frame quantities are written (Y-up -> Z-up); the LOCAL/body-rest
    # axes that R_old maps FROM are completely unchanged. A first version of
    # this used conjugation and was WRONG - verified by testing R_old =
    # identity (character with zero rotation, i.e. local axes exactly
    # aligned with the OLD Y-up world axes): conjugation gives
    # P @ I @ P.T = I (claims the character is STILL at zero rotation after
    # relabeling world axes - impossible, since its local axes are now
    # misaligned with the NEW Z-up world axes by exactly the permutation
    # itself). Left-multiply-only gives P @ I = P, correctly reporting that
    # non-identity relative rotation. This matches how the REAL codebase
    # converts SMPL's Y-up root quaternion to Z-up
    # (gear_sonic/isaac_utils/rotations.py::smpl_root_ytoz_up(), which is
    # `quat_mul(base_rot, root_quat)` - pure left multiplication, no
    # conjugation - confirmed by reading that function directly).
    # Fix determinant: the target Z-up system we actually want is
    # [x, -z, y], which is what normalize_split_test.to_local_zup() uses,
    # because that is a proper right-handed rotation (det=+1) from Y-up:
    #   [ 1  0  0]
    #   [ 0  0 -1]
    #   [ 0  1  0]
    # The previous permutation [x, z, y] was a reflection (det=-1), wrapping
    # the body in a mirrored universe, which crashed scipy's rotation parser.
    P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    root_rot_zup = np.einsum('ij,tjk->tik', P, root_rot_yup)
    return out, root_rot_zup


# ── Feature extraction (units: meters, Z-up) ────────────────────────────────
def extract_features(joints):
    """joints: (T, 24, 3) meters, Z-up. Returns dict of scalar features."""
    T = joints.shape[0]
    if T < 4:
        return None
    dt = 1.0 / FPS
    pelvis = joints[:, PELVIS]                        # (T,3)

    # Stable floor estimate: 5th percentile of foot-joint heights across the seq
    # (robust to occasional crouches/jumps corrupting a single global min).
    floor = np.percentile(joints[:, FOOT_JOINTS, 2], 5)
    z = joints[:, :, 2] - floor                       # heights above floor

    # root horizontal speed (m/s)
    root_vel = np.diff(pelvis, axis=0) / dt           # (T-1,3)
    root_speed = float(np.percentile(np.linalg.norm(root_vel[:, :2], axis=1), 75))
    root_vspeed_peak = float(np.abs(root_vel[:, 2]).max())

    # pelvis low fraction: standing pelvis ~0.9 m; crouch/sit/lie pulls it down
    pelvis_h = z[:, PELVIS]
    pelvis_low_frac = float((pelvis_h < 0.55).mean())

    # limb dynamic energy (mean joint acceleration magnitude, m/s^2)
    vel = np.diff(joints, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    limb_energy = float(np.linalg.norm(acc, axis=2).mean())

    # upper vs lower motion energy
    upper_e = np.linalg.norm(vel[:, UPPER_JOINTS], axis=2).mean()
    lower_e = np.linalg.norm(vel[:, LOWER_JOINTS], axis=2).mean() + 1e-6
    upper_lower_ratio = float(upper_e / lower_e)

    # airborne fraction: BOTH feet clearly above the floor at once (jump/hop)
    feet_h = z[:, FOOT_JOINTS].min(axis=1)
    airborne_frac = float((feet_h > 0.15).mean())

    return dict(
        frames=T,
        root_speed=root_speed,
        root_vspeed_peak=root_vspeed_peak,
        pelvis_low_frac=pelvis_low_frac,
        limb_energy=limb_energy,
        upper_lower_ratio=upper_lower_ratio,
        airborne_frac=airborne_frac,
    )


def classify(feat):
    """Rule-based routing on kinematic features -> category string."""
    if feat is None:
        return "Uncategorized"

    # 1) On/near the ground a large fraction of time -> unstructured / OOD
    #    (sit / lie / crawl / stay-down after fall)
    if feat["pelvis_low_frac"] > 0.40:
        return "Unstructured / OOD"

    # 2) Airborne phase or strong vertical impulse or very high limb energy
    #    -> agility / high-dynamic (jump / hop / kick / punch / fight)
    if (feat["airborne_frac"] > 0.06 or
            feat["root_vspeed_peak"] > 1.5 or
            feat["limb_energy"] > 8.0):
        return "Agility / High-Dynamic"

    # 3) Clear horizontal travel with active legs -> basic locomotion
    if feat["root_speed"] > 0.6:
        return "Basic Locomotion"

    # 4) Arms dominate, legs mostly static -> upper-body manipulation
    if feat["upper_lower_ratio"] > 2.0:
        return "Upper-Body Manipulation"

    # 5) Low travel, low dynamics -> gentle standing / stepping = locomotion
    return "Basic Locomotion"


def load_joints(path):
    if path.endswith('.npz'):
        d = np.load(path)
        if 'poses' not in d.files:
            return None
        poses = d['poses']
        trans = d['trans'] if 'trans' in d.files else np.zeros((poses.shape[0], 3))
        # subsample AMASS (usually 120/60 fps) down to ~FPS for speed
        src_fps = float(d['mocap_framerate']) if 'mocap_framerate' in d.files else 60.0
        step = max(1, int(round(src_fps / FPS)))
        pose_aa = poses[::step, :72].astype(np.float64)
        trans = trans[::step].astype(np.float64)
        if pose_aa.shape[0] < 4:
            return None
        j = smpl_fk(pose_aa, trans)
        # AMASS root orientation already yields a Z-up world frame -> no swap.
        return j
    elif path.endswith('.bvh'):
        j, _root_rot = parse_bvh(path)   # load_joints only needs positions
        return j
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='max files (0 = all)')
    ap.add_argument('--lafan_only', action='store_true')
    ap.add_argument('--save', type=str, default='motion_categories.csv')
    args = ap.parse_args()

    amass = [f for f in glob.glob(os.path.join(AMASS_ROOT, "**/*.npz"), recursive=True)
             if all(t not in os.path.basename(f).lower()
                    for t in ('shape', 'stagei.', 'neutral'))]
    lafan = glob.glob(os.path.join(LAFAN_ROOT, "**/*.bvh"), recursive=True)
    files = lafan if args.lafan_only else (amass + lafan)
    if args.limit:
        import random
        random.seed(0)
        files = random.sample(files, min(args.limit, len(files)))

    print(f"Classifying {len(files):,} sequences "
          f"({len(amass):,} AMASS + {len(lafan):,} LAFAN1)...")

    counts = {c: 0 for c in CATEGORIES}
    counts["Uncategorized"] = 0
    rows = []
    for i, f in enumerate(files):
        try:
            j = load_joints(f)
            feat = extract_features(j) if j is not None else None
            cat = classify(feat)
        except Exception as e:
            cat, feat = "Uncategorized", None
        counts[cat] += 1
        src = "AMASS" if f.endswith('.npz') else "LAFAN1"
        rows.append((src, os.path.relpath(f), cat,
                     None if feat is None else round(feat['root_speed'], 3),
                     None if feat is None else round(feat['airborne_frac'], 3),
                     None if feat is None else round(feat['pelvis_low_frac'], 3),
                     None if feat is None else round(feat['upper_lower_ratio'], 3),
                     None if feat is None else round(feat['limb_energy'], 2)))
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(files)}")

    # write CSV
    with open(args.save, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(["source", "path", "category", "root_speed",
                    "airborne_frac", "pelvis_low_frac",
                    "upper_lower_ratio", "limb_energy"])
        w.writerows(rows)

    total = len(files)
    print("\n" + "=" * 52)
    print("MOTION-CONTENT CLASSIFICATION (rule-based, filename-free)")
    print("=" * 52)
    for c in CATEGORIES + ["Uncategorized"]:
        n = counts[c]
        if n:
            print(f"  {c:28s} {n:5d}  ({100*n/total:5.1f}%)")
    print("=" * 52)
    print(f"Per-sequence labels + features saved to: {args.save}")


if __name__ == "__main__":
    main()
