"""
Normalize split_test.csv clips into the `smpl_filtered` representation and,
optionally, into built-in reference-motion directories for
`./target/release/g1_deploy_onnx_ref`.

WHY
---
The Mode-2 encoder expects SMPL joints in the SAME representation the training
data (`smpl_filtered`) uses:

  * ROOT-LOCAL  - the pelvis is fixed at the body-frame origin (its position is
                  constant over time); global motion lives in `transl` and the
                  root axis-angle, NOT in `smpl_joints`.
  * Z-UP        - smpl_filtered stores (x, -z, y) of the raw SMPL FK output.

Streaming raw world-frame Y-up FK joints (pelvis ~0.9 m and translating) is a
completely different distribution and is what produced the large tracking gap.

Verified against smpl_filtered by reconstructing its own clips:
    best axis map  perm=(0,2,1) signs=(1,-1,1)   residual ~0.066 m (body shape)

OUTPUTS
-------
--out_pkl_dir   <name>.pkl  {pose_aa (T,72), transl (T,3),
                             smpl_joints (T,24,3), fps}
--out_ref_dir   <name>/smpl_joint.csv, smpl_pose.csv, metadata.txt
                (loadable directly as reference motions; all other CSVs are
                 optional in motion_data_reader.hpp)

USAGE
  .venv_sim/bin/python normalize_split_test.py --limit 20 \
      --out_pkl_dir data/split_test_smpl \
      --out_ref_dir gear_sonic_deploy/reference/split_test
"""
import os
import csv
import argparse

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling imports
sys.path.insert(0, os.path.dirname(sys.path[0]))                # repo root (fix_amass.py)

import fix_amass as F           # smpl_fk (SMPL forward kinematics)
import classify_motions as C    # BVH parsing for LAFAN clips

TARGET_FPS = 50.0
# body-frame pelvis offset used by smpl_filtered (constant, shape dependent)
PELVIS_OFFSET = np.array([0.003, -0.351, 0.012])


def _rotmat_to_axis_angle(R):
    """Batched inverse-Rodrigues: (T,3,3) rotation matrices -> (T,3) axis-angle.
    
    Uses scipy.spatial.transform.Rotation which is numerically robust against
    boundary cases and reflections, unlike naive log-map implementations.
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def canonicalize_root_rotation(pose_aa, trans):
    """Re-anchor the SMPL root orientation (and translation) so the clip's
    FIRST frame is upright (spine +Z) and facing a canonical +X heading,
    matching the convention `smpl_filtered` already uses.

    WHY THIS IS NEEDED
    -------------------
    Raw AMASS/LAFAN `pose_aa[:, :3]` encodes the root joint's rotation in
    whatever absolute coordinate convention the ORIGINAL mocap rig used for
    that dataset/subject - not the identity rotation, even for a person who
    is simply standing upright facing forward. Measured root-rotation
    magnitudes:
        smpl_filtered (already canonicalized)  : ~0.12 rad average
        raw split_test AMASS clips (uncorrected): ~2-3 rad average
    Feeding that raw, large-magnitude root rotation straight into
    stream_clip_mode2.official_root_quat_w() (Y-up->Z-up, remove base rot)
    can produce a `body_quat_w` implying the torso is tilted 90+ degrees from
    vertical for a clip that is plainly standing upright - verified on
    123_123_07_stageii.npz: raw root rotation was 2.15 rad, and
    official_root_quat_w() reported a 90.7 deg tilt, while the TRUE spine
    tilt (independently measured via classify_motions FK on the neck-pelvis
    vector) was ~0 deg near frame 0. This mismatch causes the streamed
    `body_quat_w` to be wildly inconsistent with the (correct) `smpl_joints`
    geometry, and the robot falls immediately when tracking begins.

    HOW IT'S FIXED
    --------------
    We compute a single RIGID correction rotation `D` from the true skeleton
    geometry (not the root joint's own encoding) at frame 0:
        up    = normalize(neck(12)   - pelvis(0))
        right = normalize(L_hip(1)   - R_hip(2)), Gram-Schmidt vs `up`
        fwd   = up x right
    `F0 = [right | fwd | up]` is the world-frame basis implied by frame 0's
    pose. `D = F0^T` is the rotation that maps that basis back to the
    canonical (X=right, Y=fwd, Z=up) world axes.

    Because `classify_motions.smpl_fk` computes every joint's world rotation
    as `gr[parent] @ R_local(joint)`, and the root has no parent
    (`gr[0] = R_root`, `gp[0] = trans`), left-multiplying the root's rotation
    AND translation by the same constant `D` propagates through the ENTIRE
    kinematic chain by induction:
        new_joints(t) = D @ old_joints(t)   for every joint and every frame.
    This is exactly a rigid rotation of the whole recording -- it changes
    ONLY the arbitrary absolute anchor, and preserves 100% of the relative
    motion (turns, leans, gait, everything) frame-to-frame and joint-to-joint.
    Only the ROOT's own axis-angle needs to be rewritten (all other joints'
    axis-angles are parent-relative and are therefore untouched).

    Returns
    -------
    (pose_aa_fixed, trans_fixed) - same shapes as the inputs.
    """
    if len(pose_aa) == 0:
        return pose_aa, trans

    # IMPORTANT: correct the ROOT JOINT'S OWN rotation to near-identity at
    # frame 0 -- NOT the full-body visible orientation. `official_root_quat_w`
    # (used both by stream_clip_mode2.py and the live PICO pipeline) is built
    # ONLY from pose_aa[:, :3]; it has no knowledge of the spine chain
    # (joints 3, 6, 9, 12) that also contributes to how the torso visibly
    # looks oriented. An earlier version of this function aligned the FULL
    # BODY's neck-pelvis vector to world +Z, which reduced but did NOT
    # eliminate the bad tilt (92 deg -> 70 deg) precisely because the root
    # joint's own rotation still didn't start near-identity. Zeroing the root
    # joint's OWN rotation at frame 0 is also exactly what makes
    # smpl_filtered's pose_aa[:, :3] have such a small average magnitude
    # (~0.12 rad) compared to raw AMASS (~2-3 rad): that dataset's own
    # pipeline evidently applies the same kind of root-only canonicalization.
    R0 = F.axis_angle_to_matrix(pose_aa[0, :3])
    D = R0.T   # left-multiplying by D makes new_root_mats[0] = D @ R0 = Identity

    root_aa = pose_aa[:, :3]
    root_mats = np.stack([F.axis_angle_to_matrix(aa) for aa in root_aa], axis=0)
    new_root_mats = np.einsum('ij,tjk->tik', D, root_mats)
    new_root_aa = _rotmat_to_axis_angle(new_root_mats)

    pose_aa_fixed = pose_aa.copy()
    pose_aa_fixed[:, :3] = new_root_aa
    trans_fixed = trans @ D.T                          # D @ trans(t) for every t

    return pose_aa_fixed, trans_fixed


def to_local_zup(pose_aa, pelvis_offset=PELVIS_OFFSET):
    """Root-local (translation only), Z-up SMPL joints in the smpl_filtered
    convention.

    IMPORTANT #1: root ROTATION must be KEPT in the FK (not zeroed). The
    stored smpl_joints in real smpl_filtered files still carry the root
    rotation baked in - only the root TRANSLATION is pinned.
    stream_clip_mode2.py's official_encoder_joints() removes that rotation at
    stream time, mirroring gear_sonic/envs/manager_env/mdp/observations.py::
    smpl_joints_multi_future_local(). If we zeroed the rotation here AND it
    gets removed again there, every non-pelvis joint receives a spurious
    extra rotation (verified: ~0.57 m mean error on a turning clip).

    IMPORTANT #2: use classify_motions.smpl_fk's convention directly, with NO
    extra axis swap. `fix_amass.smpl_fk` and `classify_motions.smpl_fk` are
    two SEPARATE implementations with DIFFERENT internal axis conventions.
    classify_motions.load_joints() applies its FK with no swap and has been
    repeatedly verified against real-world standing heights (foot ~0m, head
    ~1.4-1.5m) throughout this project. Using fix_amass.smpl_fk + a hand-
    derived [0,2,1]*[1,-1,1] remap instead produced joint POSITIONS that
    happened to look reasonable (matched real smpl_filtered clips to ~0.05 m,
    a position-only check) but a WRONG rotational frame: the neck-pelvis
    "spine direction" came out 110 deg from vertical instead of the true
    31 deg (independently confirmed via classify_motions FK) - because a
    coordinate permutation can preserve relative distances between points
    while still representing a different orientation. This silently fed the
    encoder a badly wrong body_quat_w and caused immediate robot falls.
    """
    loc = C.smpl_fk(pose_aa, np.zeros((len(pose_aa), 3)))   # (T,24,3), Z-up, ROOT ROTATION KEPT
    return loc - loc[:, 0:1, :] + pelvis_offset             # pin TRANSLATION only


def load_clip(path):
    """Return (pose_aa (T,72), transl (T,3)) resampled to TARGET_FPS.

    AMASS (.npz) only - LAFAN (.bvh) uses the dedicated load_bvh_clip() below,
    since its smpl_joints must come from the REAL captured joint positions,
    not from re-running SMPL FK on an all-zero body pose.
    """
    if path.endswith('.npz'):
        d = np.load(path)
        poses = d['poses']
        src = float(d['mocap_framerate']) if 'mocap_framerate' in d.files else 60.0
        tr = d['trans'] if 'trans' in d.files else np.zeros((len(poses), 3))
        step = max(1, int(round(src / TARGET_FPS)))
        return poses[::step, :72].astype(np.float64), tr[::step].astype(np.float64)
    return None, None


def _interp_linear(x_src, y_src, x_tgt):
    """Linear interpolation along axis 0 for an (T, ...) array `y_src`
    sampled at times `x_src`, evaluated at `x_tgt`. Thin wrapper around
    np.interp applied independently to every trailing component."""
    flat = y_src.reshape(len(y_src), -1)
    out = np.stack([np.interp(x_tgt, x_src, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return out.reshape((len(x_tgt),) + y_src.shape[1:])


def load_bvh_clip(path):
    """LAFAN1 (.bvh) -> smpl_filtered-style (pose_aa, transl, smpl_joints).

    Fixes/implements, per RAW_LAFAN1_DATA_FORMAT.md:

    1. classify_motions.parse_bvh() now ALSO returns the real per-frame
       Hips (root) rotation matrix (already computed internally by its own
       FK loop, previously discarded). We convert it to axis-angle and use
       it as pose_aa[:, :3] -- giving body_quat_w a REAL, per-frame root
       orientation instead of the previous all-zero placeholder (which made
       official_root_quat_w() report a constant identity orientation
       regardless of the true pose on every LAFAN1 clip).
    2. LAFAN1 is native 30 fps (verified: `Frame Time: 0.033333`), but EVERY
       smpl_filtered clip is uniformly 50 fps (verified empirically: fps=50.0
       and original_fps=30.0 on 100% of a 3000-clip random sample). We
       therefore UPSAMPLE (30->50 fps) via linear interpolation of both the
       joint positions and the root axis-angle, onto the same target
       timeline used elsewhere (`TARGET_FPS`), rather than a stride-based
       decimation (which cannot upsample: round(30/50)=round(0.6)=1 would
       silently keep every frame unchanged, exactly the previous no-op bug).
    3. canonicalize_root_rotation() (already fixed/verified for AMASS) is
       reused as-is to re-anchor frame 0 to a near-identity root rotation,
       matching smpl_filtered's convention. The SAME rigid correction D is
       then applied directly to the interpolated joint positions (not just
       pose_aa/transl), since a rigid rotation of the whole recording's root
       propagates identically to every joint's world position for ANY
       skeleton (SMPL's or LAFAN's own) - proven for the SMPL FK chain in
       canonicalize_root_rotation()'s docstring, and equally true here
       because it only changes the absolute anchor, not any inter-joint
       relationship.

    Returns
    -------
    (pose_aa (T,72), transl (T,3), smpl_joints (T,24,3)) at 50 fps, or
    (None, None, None) if the file could not be parsed.
    """
    parsed = C.parse_bvh(path)
    if parsed is None:
        return None, None, None
    j30, root_rot30 = parsed                      # (T0,24,3), (T0,3,3) @ 30fps
    
    # LAFAN defaults missing 22 and 23 to [0,0,0], extrapolate them from forearms to wrists
    j30[:, 22, :] = j30[:, 20, :] + (j30[:, 20, :] - j30[:, 18, :]) * 0.3
    j30[:, 23, :] = j30[:, 21, :] + (j30[:, 21, :] - j30[:, 19, :]) * 0.3

    T0 = len(j30)
    src_fps = 30.0                                 # verified: Frame Time = 0.033333
    duration = (T0 - 1) / src_fps
    t_src = np.arange(T0) / src_fps
    t_tgt = np.arange(0, duration, 1.0 / TARGET_FPS)
    if len(t_tgt) < 4:
        return None, None, None

    root_aa30 = _rotmat_to_axis_angle(root_rot30)  # (T0,3)
    j50 = _interp_linear(t_src, j30, t_tgt)         # (T,24,3) @ 50fps
    root_aa50 = _interp_linear(t_src, root_aa30, t_tgt)  # (T,3) @ 50fps

    pose_aa = np.zeros((len(t_tgt), 72))
    pose_aa[:, :3] = root_aa50
    transl = j50[:, 0, :].copy()                    # pelvis world position

    pose_aa_fixed, transl_fixed = canonicalize_root_rotation(pose_aa, transl)

    # Apply the SAME rigid correction D to the joint positions themselves
    # (canonicalize_root_rotation() only touches pose_aa/transl; recompute D
    # here rather than changing that function's return signature, to keep
    # the already-verified AMASS code path untouched).
    D = F.axis_angle_to_matrix(pose_aa[0, :3]).T
    j50_fixed = j50 @ D.T

    smpl_joints = j50_fixed - j50_fixed[:, 0:1, :] + PELVIS_OFFSET
    return pose_aa_fixed, transl_fixed, smpl_joints


def aa_to_quat_zup(pose_aa):
    """Root axis-angle -> world quaternion (w,x,y,z) in the Z-up basis.

    `smpl_anchor_orientation_4frame_step1` is built from motion_body_quat, so a
    reference motion must ship `body_quat.csv`. The root rotation lives in
    pose_aa[:, :3] in the raw (Y-up) SMPL frame; we rebase it with the same
    (x, y, z) -> (x, -z, y) change of basis used for the joints.
    """
    P = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])
    out = np.zeros((len(pose_aa), 4))
    for t, aa in enumerate(pose_aa[:, :3]):
        R = F.axis_angle_to_matrix(aa)
        R = P @ R @ P.T
        w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
        if w > 1e-8:
            x = (R[2, 1] - R[1, 2]) / (4 * w)
            y = (R[0, 2] - R[2, 0]) / (4 * w)
            z = (R[1, 0] - R[0, 1]) / (4 * w)
        else:  # 180 deg fallback
            x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
            y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
            z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
        q = np.array([w, x, y, z])
        out[t] = q / (np.linalg.norm(q) + 1e-12)
    return out


def write_reference_dir(out_dir, name, smpl_joints, smpl_pose, root_quat):
    """Write a reference-motion directory the C++ reader can load in mode 2.

    Mode 2 ('smpl') requires FOUR observations:
        encoder_mode_4                          (internal)
        smpl_joints_4frame_step1                <- smpl_joint.csv
        smpl_anchor_orientation_4frame_step1    <- body_quat.csv
        motion_joint_positions_wrists_4frame_step1 <- joint_pos.csv

    NOTE: we deliberately do NOT write joint_vel.csv. Mode 0 ('g1') requires
    motion_joint_velocities_10frame_step1, so without that file mode 0 fails
    its check and the binary falls back to mode 2 - which is what we want for
    SMPL clips. (joint_pos is zero-filled: only the wrist entries are consumed
    by mode 2, exactly like the live ZMQ stream which also sends zeros.)
    """
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    T, J, _ = smpl_joints.shape
    P = smpl_pose.shape[1]

    with open(os.path.join(d, 'smpl_joint.csv'), 'w') as f:
        f.write(','.join(f"smpl_joint_{j}_{a}" for j in range(J) for a in 'xyz') + "\n")
        for t in range(T):
            f.write(','.join(f"{v:.9f}" for v in smpl_joints[t].reshape(-1)) + "\n")

    with open(os.path.join(d, 'smpl_pose.csv'), 'w') as f:
        f.write(','.join(f"smpl_pose_{p}_{a}" for p in range(P) for a in 'xyz') + "\n")
        for t in range(T):
            f.write(','.join(f"{v:.9f}" for v in smpl_pose[t].reshape(-1)) + "\n")

    # root orientation -> smpl_anchor_orientation
    with open(os.path.join(d, 'body_quat.csv'), 'w') as f:
        f.write("body_0_w,body_0_x,body_0_y,body_0_z\n")
        for t in range(T):
            f.write(','.join(f"{v:.9f}" for v in root_quat[t]) + "\n")

    # 29 DOF placeholders.
    # Mode 2 only ever reads the SIX wrist entries
    #   wrist_joint_isaaclab_order_in_isaaclab_index = {23,24,25,26,27,28}
    # (policy_parameters.hpp:88), so every other entry is unread and can stay
    # zero. joint_vel.csv must exist as well: if joint_pos.csv is present while
    # joint_vel.csv is missing, the velocity gather indexes an empty vector and
    # the binary segfaults instead of failing over to the next encoder mode.
    for fn in ('joint_pos.csv', 'joint_vel.csv'):
        with open(os.path.join(d, fn), 'w') as f:
            f.write(','.join(f"joint_{i}" for i in range(29)) + "\n")
            zero = ','.join(["0.000000000"] * 29) + "\n"
            for _ in range(T):
                f.write(zero)

    with open(os.path.join(d, 'metadata.txt'), 'w') as f:
        f.write(f"Metadata for: {name}\n{'='*30}\n\n")
        f.write("Body part indexes:\n[ 0  4 10 18  5 11 19  9 16 22 28 17 23 29]\n\n")
        f.write(f"Total timesteps: {T}\n\n")
        f.write(f"  smpl_joints: ({T}, {J}, 3) (float32)\n")
        f.write(f"  smpl_poses: ({T}, {P}, 3) (float32)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_csv', default='data_analysis/split/split_test.csv')
    ap.add_argument('--out_pkl_dir', default='data/split_test_smpl')
    ap.add_argument('--out_ref_dir', default='')
    ap.add_argument('--limit', type=int, default=0, help='0 = all')
    ap.add_argument('--per_category', type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.split_csv)))
    if args.per_category:
        from collections import defaultdict
        by = defaultdict(list)
        for r in rows:
            by[r['category']].append(r)
        rows = [r for v in by.values() for r in v[:args.per_category]]
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(args.out_pkl_dir, exist_ok=True)
    if args.out_ref_dir:
        os.makedirs(args.out_ref_dir, exist_ok=True)

    import joblib
    ok = 0
    for i, r in enumerate(rows):
        src = os.path.normpath(os.path.join(os.getcwd(), r['path']))
        # basenames are NOT unique across AMASS subjects (e.g. rub050 and
        # rub114 both contain 0015_sitting1_poses.npz), so qualify the name
        # with its parent directory to avoid silently overwriting clips.
        base = os.path.splitext(os.path.basename(src))[0]
        parent = os.path.basename(os.path.dirname(src))
        name = f"{parent}_{base}"
        try:
            if src.endswith('.bvh'):
                pose_aa, transl, smpl_joints = load_bvh_clip(src)
                if pose_aa is None or len(pose_aa) < 4:
                    print(f"  skip {name}: unreadable"); continue
            else:
                pose_aa, transl = load_clip(src)
                if pose_aa is None or len(pose_aa) < 4:
                    print(f"  skip {name}: unreadable"); continue
                # Re-anchor the raw AMASS root rotation to the smpl_filtered
                # convention BEFORE running FK, so smpl_joints/body_quat_w are
                # geometrically consistent with a standing-upright frame 0.
                pose_aa, transl = canonicalize_root_rotation(pose_aa, transl)
                smpl_joints = to_local_zup(pose_aa)

            smpl_pose = pose_aa[:, 3:3 + 63].reshape(-1, 21, 3)

            joblib.dump({
                'pose_aa': pose_aa.astype(np.float32),
                'transl': transl.astype(np.float32),
                'smpl_joints': smpl_joints.astype(np.float32),
                'fps': TARGET_FPS,
            }, os.path.join(args.out_pkl_dir, name + '.pkl'))

            if args.out_ref_dir:
                write_reference_dir(args.out_ref_dir, name,
                                    smpl_joints.astype(np.float32),
                                    smpl_pose.astype(np.float32),
                                    aa_to_quat_zup(pose_aa))
            ok += 1
            if (i + 1) % 25 == 0 or i == 0:
                print(f"  [{i+1}/{len(rows)}] {name}  T={len(pose_aa)} "
                      f"pelvis={smpl_joints[:,0,:].mean(0).round(3)}", flush=True)
        except Exception as e:
            print(f"  fail {name}: {e}")

    print(f"\nconverted {ok}/{len(rows)} clips -> {args.out_pkl_dir}")
    if args.out_ref_dir:
        print(f"reference motions -> {args.out_ref_dir}")


if __name__ == "__main__":
    main()
