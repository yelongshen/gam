"""Convert `smpl_filtered`-style `.pkl` clips into `.bvh` files on the SOMA
skeleton, so they can be fed into the external SOMA Retargeter
(https://github.com/NVIDIA/soma-retargeter).

Implements the pipeline documented in `CONVERT_SMPL_FILTERED_TO_BVH_PLAN.md`.

v2 design note (this replaced an earlier `pose_aa`-based approach): body
joint rotations are now derived directly from `smpl_joints` (3D positions),
NOT from `pose_aa[:, 3:72]`. This is required because `pose_aa` only carries
real per-joint rotation data for AMASS-derived `smpl_filtered` clips;
LAFAN1-derived clips (`convert_lafan_to_smpl_filtered.py`) only populate
`pose_aa[:, :3]` (root) -- body joints are left at zero, since LAFAN gives
joint POSITIONS via BVH FK, not SMPL-style per-joint axis-angle. Verified
directly: `pose_aa[:, 3:72]` for `data/lafan1_smpl_filtered/*.pkl` has
`std() == 0.0` everywhere. Using it produced a converted BVH where every
body joint (knee, elbow, ...) never bent at all.

The fix reuses the SAME geometric technique already proven in
`convert_lafan_to_smpl_filtered.py`'s root canonicalization (build an
orthonormal frame from a primary bone direction + a secondary lateral
reference, both derived from live 3D joint positions), generalized to
EVERY mapped joint instead of just the root. Since `smpl_joints` is always
populated (LAFAN and AMASS alike), this works universally and also
sidesteps the earlier REST-pose calibration approach's twist-ambiguity
severity (per-frame, geometry-only -- no separate rest-frame comparison
needed).

Usage:
    .venv_sim/bin/python convert_smpl_filtered_to_bvh.py \\
        --pkl data/lafan1_smpl_filtered/walk1_subject1.pkl \\
        --template ~/soma-retargeter/assets/motions/bvh/Neutral_walk_forward_002__A057.bvh \\
        --out /tmp/walk1_subject1_soma.bvh
"""
import argparse
import glob
import multiprocessing
import os
import sys
import time

import joblib
import numpy as np
import scipy.spatial.transform as sT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling imports
import classify_motions as C  # SMPL_PARENTS indexing reference (documentation only)


# SMPL-24 joint index -> SOMA joint name. See CONVERT_SMPL_FILTERED_TO_BVH_PLAN.md
# Sec.2.1 for the full anatomical justification. Root (index 0) is handled
# separately (translation + rotation on `Hips`); indices 22/23 (hand-tip
# end-effectors) have no BVH-rotatable SOMA equivalent and are left unmapped.
SMPL_TO_SOMA = {
    1: 'LeftLeg', 2: 'RightLeg',
    3: 'Spine1',
    4: 'LeftShin', 5: 'RightShin',
    6: 'Spine2',
    7: 'LeftFoot', 8: 'RightFoot',
    9: 'Chest',
    10: 'LeftToeBase', 11: 'RightToeBase',
    12: 'Neck1',
    13: 'LeftShoulder', 14: 'RightShoulder',
    15: 'Head',
    16: 'LeftArm', 17: 'RightArm',
    18: 'LeftForeArm', 19: 'RightForeArm',
    20: 'LeftHand', 21: 'RightHand',
}

# For each SMPL joint (parent, used as a rotation "carrier"), the ONE
# primary child used to derive that parent's bone direction (both in
# smpl_joints live data and in the SOMA template's rest OFFSETs). Chosen to
# match the main serial kinematic chain; a parent's OTHER children (e.g.
# Chest also has LeftShoulder/RightShoulder besides its primary Neck1) get
# their OWN rotation from THEIR OWN primary child independently -- no
# conflict, since each joint's rotation is computed independently and only
# consumed as "parent world rotation" by whichever child needs it.
PRIMARY_CHILD = {
    0: 3,    # Hips -> Spine1 (root's own "up" direction)
    1: 4, 2: 5,      # Left/RightLeg -> Shin
    3: 6,            # Spine1 -> Spine2
    4: 7, 5: 8,      # Left/RightShin -> Foot
    6: 9,            # Spine2 -> Chest
    7: 10, 8: 11,    # Left/RightFoot -> ToeBase
    9: 12,           # Chest -> Neck1
    12: 15,          # Neck1 -> Head... wait Head has no mapped child, see LEAF note below
    13: 16, 14: 17,  # Left/RightShoulder -> Arm
    16: 18, 17: 19,  # Left/RightArm -> ForeArm
    18: 20, 19: 21,  # Left/RightForeArm -> Hand
}
# Leaf joints in our mapped set (no further mapped child of their own to
# derive a bone direction from): get Identity local rotation (no data to
# derive it from -- reduced-24-joint SMPL has no hand-tip/beyond-toe/
# beyond-head landmark). Head(15) technically has a PRIMARY_CHILD entry
# above (mapped to itself via Neck1->Head) but Head itself has no further
# child, so its OWN local rotation is likewise identity; that's handled by
# it simply not appearing as a key needing ITS OWN world-rotation lookup
# beyond what its parent (Neck1) already computed.
LEAF_JOINTS = {10, 11, 15, 20, 21}

ROOT_SOMA_NAME = 'Hips'
LATERAL_PAIR_SMPL = (1, 2)  # L_hip -> R_hip: stable lateral reference, SMPL indices


def axis_angle_to_matrix(aa):
    theta = np.linalg.norm(aa)
    if theta < 1e-8:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _self_test_rotation_convention():
    """Verify our Euler extraction matches the retargeter's own composition
    order R = Rz(a) @ Ry(b) @ Rx(c) for channel order [Z,Y,X] (confirmed by
    reading soma_retargeter/assets/bvh.py::euler_to_quaternion directly).

    NOTE: this requires scipy's *uppercase* ('ZYX', extrinsic) convention,
    NOT lowercase ('zyx', intrinsic) -- verified empirically: only 'ZYX'
    round-trips exactly through a known R = Rz(20)@Ry(30)@Rx(40) test case;
    'zyx' silently returns a completely different (wrong) triple with no
    error raised, so this self-test exists specifically to catch that."""
    def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(20):
        aa = rng.normal(size=3) * 1.2
        R = axis_angle_to_matrix(aa)
        z, y, x = sT.Rotation.from_matrix(R).as_euler('ZYX', degrees=True)
        R_rebuilt = Rz(np.radians(z)) @ Ry(np.radians(y)) @ Rx(np.radians(x))
        max_err = max(max_err, np.abs(R - R_rebuilt).max())
    print(f"[self-test] Euler(ZYX) round-trip max error: {max_err:.2e} "
          f"({'OK' if max_err < 1e-8 else 'FAILED'})")
    return max_err < 1e-8


def parse_bvh_template(path):
    """Read a reference SOMA BVH: return (hierarchy_text_lines_before_MOTION,
    joints, channels, offsets, n_channels) so we can write a new MOTION
    section into the exact same hierarchy/channel layout."""
    with open(path) as f:
        text = f.read()
    lines = text.split('\n')
    mo = next(k for k, l in enumerate(lines) if l.strip() == 'MOTION')
    hierarchy_lines = lines[:mo]

    joints, channels, offsets = [], {}, {}
    stack, ch_idx = [], 0
    for line in hierarchy_lines:
        s = line.strip()
        if s.startswith('ROOT ') or s.startswith('JOINT '):
            name = s.split()[1]
            joints.append(name)
            stack.append(name)
        elif s.startswith('OFFSET') and stack:
            p = s.split()
            offsets[stack[-1]] = np.array([float(p[1]), float(p[2]), float(p[3])])
        elif s.startswith('CHANNELS') and stack:
            p = s.split(); n = int(p[1])
            channels[stack[-1]] = {'start': ch_idx, 'types': p[2:2 + n]}
            ch_idx += n
        elif s.startswith('End Site'):
            pass  # no CHANNELS, nothing to record
        elif s == '}' and stack:
            stack.pop()
    return hierarchy_lines, joints, channels, offsets, ch_idx


def _build_frames_batch(primary_dir, lateral_dir):
    """Vectorized orthonormal-frame construction from a primary direction
    and a secondary (lateral) reference, for a whole (T,3) batch at once.
    Same recipe as `canonicalize_root_rotation`'s `[right|fwd|up]` frame
    (Gram-Schmidt + cross product), generalized: axis1 = primary (bone
    direction), axis2 = lateral, orthogonalized against axis1, axis3 =
    axis1 x axis2. Returns (T,3,3) rotation matrices (columns = axes)."""
    a1 = primary_dir / np.linalg.norm(primary_dir, axis=-1, keepdims=True)
    proj = np.sum(lateral_dir * a1, axis=-1, keepdims=True)
    a2_raw = lateral_dir - proj * a1
    norm2 = np.linalg.norm(a2_raw, axis=-1, keepdims=True)
    # Guard against a1 (nearly) parallel to lateral_dir: fall back to a
    # fixed world reference for those (rare) frames/joints.
    bad = (norm2 < 1e-6).squeeze(-1)
    if np.any(bad):
        fallback = np.where(np.abs(a1[bad, 0:1]) < 0.9, np.array([1., 0., 0.]), np.array([0., 1., 0.]))
        a2_raw[bad] = fallback - np.sum(fallback * a1[bad], axis=-1, keepdims=True) * a1[bad]
        norm2 = np.linalg.norm(a2_raw, axis=-1, keepdims=True)
    a2 = a2_raw / norm2
    a3 = np.cross(a1, a2)
    return np.stack([a1, a2, a3], axis=-1)  # (T,3,3), columns are the 3 axes


def convert(pkl_path, template_path, out_path, verbose=True):
    ok = _self_test_rotation_convention() if verbose else True
    if verbose and not ok:
        print("[WARN] rotation-convention self-test failed; output angles "
              "may be wrong. Continuing anyway for inspection.")

    d = joblib.load(pkl_path)
    smpl_joints = np.asarray(d['smpl_joints'], dtype=np.float64)  # (T,24,3) Z-up meters, root-rotated
    transl = np.asarray(d['transl'], dtype=np.float64)            # (T, 3), Z-up meters
    fps = float(d.get('fps', 50.0))
    T = len(smpl_joints)
    if verbose:
        print(f"Loaded {pkl_path}: {T} frames @ {fps} fps")

    hierarchy_lines, joints, channels, soma_offsets, n_channels = parse_bvh_template(template_path)
    if verbose:
        print(f"Template hierarchy: {len(joints)} joints, {n_channels} channels total")

    # ── Per-joint REST frames (SOMA template, computed once) ──────────────
    lateral_smpl_rest = C.SMPL_REST[LATERAL_PAIR_SMPL[1]] - C.SMPL_REST[LATERAL_PAIR_SMPL[0]]
    lateral_soma_rest = (soma_offsets.get('RightLeg', np.zeros(3))
                         - soma_offsets.get('LeftLeg', np.zeros(3)))

    rest_frame_soma = {}  # smpl_idx (rotation carrier) -> (3,3)
    for p, c in PRIMARY_CHILD.items():
        soma_name = SMPL_TO_SOMA.get(c)
        if soma_name is None or soma_name not in soma_offsets:
            continue
        rest_dir = soma_offsets[soma_name][None, :]
        rest_lat = lateral_soma_rest[None, :]
        rest_frame_soma[p] = _build_frames_batch(rest_dir, rest_lat)[0]

    # ── Per-joint LIVE frames (from smpl_joints, every frame at once) ─────
    # IMPORTANT: `smpl_joints` is Z-up (SMPL_FILTERED_DATA_FORMAT.md Sec.5.3),
    # but `soma_offsets` (the template BVH's rest OFFSETs) are in the raw
    # BVH convention, Y-up. Bone directions derived from smpl_joints MUST be
    # converted into that same raw Y-up convention before being compared
    # against/aligned to soma_offsets -- otherwise the two are in mismatched
    # coordinate systems, which produces pose-DEPENDENT (not constant, so
    # much harder to spot) errors: sometimes a joint's direction happens to
    # still roughly line up despite the axis mismatch, other times it's
    # badly wrong, producing an intermittent "leg suddenly rises above hip
    # height" glitch rather than a uniform bug. This was found and fixed by
    # tracing one such glitch frame back to its raw bone-direction vectors.
    # Same inverse mapping used for root translation above:
    #   raw.x = zup.x, raw.y = zup.z, raw.z = -zup.y
    smpl_joints_raw = np.stack([
        smpl_joints[..., 0], smpl_joints[..., 2], -smpl_joints[..., 1],
    ], axis=-1)

    lateral_live = (smpl_joints_raw[:, LATERAL_PAIR_SMPL[1]] - smpl_joints_raw[:, LATERAL_PAIR_SMPL[0]])

    R_world_soma = {}  # smpl_idx -> (T,3,3)
    for p, c in PRIMARY_CHILD.items():
        if p not in rest_frame_soma:
            continue
        bone_dir_live = smpl_joints_raw[:, c] - smpl_joints_raw[:, p]
        target_frame = _build_frames_batch(bone_dir_live, lateral_live)          # (T,3,3)
        # R_world_soma(p) @ rest_frame_soma(p) = target_frame  =>  R = target @ rest^T
        R_world_soma[p] = np.einsum('tij,kj->tik', target_frame, rest_frame_soma[p])

    motion_data = np.zeros((T, n_channels), dtype=np.float64)
    letter_to_col = {'Zrotation': 0, 'Yrotation': 1, 'Xrotation': 2}

    def write_local_rotation(soma_name, R_local):
        ch = channels.get(soma_name)
        if ch is None:
            return False
        start, types = ch['start'], ch['types']
        if not all(t.endswith('rotation') or t.endswith('position') for t in types):
            return False
        euler = sT.Rotation.from_matrix(R_local).as_euler('ZYX', degrees=True)
        for k, t in enumerate(types):
            if t in letter_to_col:
                motion_data[:, start + k] = euler[:, letter_to_col[t]]
        return True

    # ── Calculate Scale Ratio ──────────────────────────────────────────────
    # AMASS/LAFAN subjects have true biological leg lengths, but this script
    # maps their rotations onto a fixed-geometry SOMA template. If the SMPL
    # subject's legs are shorter than SOMA, mapping their unscaled pelvis
    # height directly onto the SOMA rig drives the toes deep underground during
    # retargeting (since the fixed SOMA legs overreach the floor).
    # We must scale the root height (z) proportionately to leg length difference.
    
    # LAFAN: Hips(0)->LeftUpLeg(1)->LeftLeg(4)->LeftFoot(7)->LeftToeBase(10)
    def dist(idx_a, idx_b):
        return np.linalg.norm(smpl_joints[0, idx_a] - smpl_joints[0, idx_b])
    
    # SMPL biological length (cm)
    ll_smpl = (dist(0, 1) + dist(1, 4) + dist(4, 7) + dist(7, 10)) * 100.0
    
    # SOMA template length (cm)
    def dist_soma(name):
        return np.linalg.norm(soma_offsets.get(name, np.zeros(3)))
    
    # SOMA: Hips->LeftLeg->LeftShin->LeftFoot->LeftToeBase (LeftLeg is child of Hips)
    ll_soma = dist_soma("LeftLeg") + dist_soma("LeftShin") + dist_soma("LeftFoot") + dist_soma("LeftToeBase")
    
    scale_ratio = ll_soma / ll_smpl if ll_smpl > 0 else 1.0

    # ── Root (Hips): translation + rotation ────────────────────────────────
    # Translation: world, Z-up meters -> raw BVH Y-up cm. Inverting
    # RAW_LAFAN1_DATA_FORMAT.md Sec.4: parsed = (raw.X, -raw.Z, raw.Y)/100
    #   => raw.X = zup.x, raw.Y = zup.z, raw.Z = -zup.y   (all * 100 for cm)
    # Scale root Z and XY to match the template skeleton's dimensions
    # so the SOMA rig's feet land at z=0 without penetrating the floor.
    raw_x = transl[:, 0] * 100.0 * scale_ratio
    raw_y = transl[:, 2] * 100.0 * scale_ratio
    raw_z = -transl[:, 1] * 100.0 * scale_ratio

    root_ch = channels[ROOT_SOMA_NAME]
    start, types = root_ch['start'], root_ch['types']
    for k, t in enumerate(types):
        col = start + k
        if t == 'Xposition':
            motion_data[:, col] = raw_x
        elif t == 'Yposition':
            motion_data[:, col] = raw_y
        elif t == 'Zposition':
            motion_data[:, col] = raw_z
    # Root has no parent (the outer "Root" node is always identity), so
    # its LOCAL rotation IS its world rotation directly.
    root_euler = sT.Rotation.from_matrix(R_world_soma[0]).as_euler('ZYX', degrees=True)
    for k, t in enumerate(types):
        if t in letter_to_col:
            motion_data[:, start + k] = root_euler[:, letter_to_col[t]]

    # ── Body joints: R_local(j) = R_world(parent_j)^-1 @ R_world(j) ────────
    mapped, skipped = 0, []
    for smpl_idx, soma_name in SMPL_TO_SOMA.items():
        parent_idx = int(C.SMPL_PARENTS[smpl_idx])
        if smpl_idx in LEAF_JOINTS or smpl_idx not in R_world_soma:
            # No data to derive this joint's own rotation from (reduced
            # 24-joint SMPL has no landmark beyond hand/head/toe) -- leave
            # at rest (Identity local rotation, i.e. all-zero channels,
            # already the array's initialized value).
            if write_local_rotation(soma_name, np.tile(np.eye(3), (T, 1, 1))):
                mapped += 1
            else:
                skipped.append(soma_name)
            continue
        R_parent = R_world_soma.get(parent_idx)
        if R_parent is None:
            skipped.append(f"{soma_name} (no parent world-rotation available)")
            continue
        R_local = np.einsum('tji,tjk->tik', R_parent, R_world_soma[smpl_idx])
        if write_local_rotation(soma_name, R_local):
            mapped += 1
        else:
            skipped.append(soma_name)

    if verbose:
        print(f"Mapped {mapped}/{len(SMPL_TO_SOMA)} body joints "
              f"({len(skipped)} skipped: {skipped[:5]}{'...' if len(skipped) > 5 else ''})")

    # ── Write the .bvh: template hierarchy verbatim + our new MOTION data ──
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(hierarchy_lines))
        f.write('\nMOTION\n')
        f.write(f'Frames: {T}\n')
        f.write(f'Frame Time: {1.0 / fps:.6f}\n')
        for t in range(T):
            f.write(' '.join(f'{v:.6f}' for v in motion_data[t]))
            f.write('\n')

    if verbose:
        print(f"Saved -> {out_path}")
    return out_path


def verify_roundtrip(bvh_path, pkl_path):
    """Reload the generated BVH and compare the Hips (root) world position
    against the original smpl_filtered `transl` field -- should match almost
    exactly since root translation math is exact (no calibration ambiguity
    there, unlike body-joint rotations -- see plan doc Sec.2.2).

    IMPORTANT: this reads Hips's position channels DIRECTLY rather than
    running them through the shared `visualize_soma_retarget.fk()` helper.
    That helper (like the retargeter's OWN template files elsewhere) always
    ADDS `OFFSET + position_channel` for any joint with position channels,
    which is WRONG for a "true root with its own OFFSET" joint like this
    Hips (verified against soma_retargeter/assets/bvh.py's actual parser --
    see plan doc Sec.1.2: position channels REPLACE OFFSET, they don't add
    to it). Using the shared buggy FK here would report a false ~1.01 m
    constant error (exactly `Hips` OFFSET.y) even though the writer is
    correct."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    hierarchy_lines, joints, channels, soma_offsets, n_channels = parse_bvh_template(bvh_path)
    with open(bvh_path) as f:
        text = f.read()
    lines = text.split('\n')
    mo = next(k for k, l in enumerate(lines) if l.strip() == 'MOTION')
    nf = int(lines[mo + 1].split()[1])
    frame_data = np.array([[float(v) for v in lines[k].split()]
                            for k in range(mo + 3, mo + 3 + nf) if lines[k].strip()])

    d = joblib.load(pkl_path)
    transl = np.asarray(d['transl'], dtype=np.float64)

    root_ch = channels[ROOT_SOMA_NAME]
    start, types = root_ch['start'], root_ch['types']
    col = {t: start + k for k, t in enumerate(types)}

    n_check = min(len(frame_data), len(transl), 50)
    errs = []
    for t in range(0, n_check, max(1, n_check // 10)):
        raw_x = frame_data[t, col['Xposition']]
        raw_y = frame_data[t, col['Yposition']]
        raw_z = frame_data[t, col['Zposition']]
        hips_m_zup = np.array([raw_x / 100.0, -raw_z / 100.0, raw_y / 100.0])
        err = np.linalg.norm(hips_m_zup - transl[t])
        errs.append(err)
    errs = np.array(errs)
    print(f"[verify] Hips world-position round-trip error over {len(errs)} "
          f"sampled frames: mean={errs.mean():.6f} m, max={errs.max():.6f} m")


def _convert_one_task(args_tuple):
    pkl_path, template_path, out_path = args_tuple
    name = os.path.basename(out_path)
    try:
        convert(pkl_path, template_path, out_path, verbose=False)
        return (name, "ok", None)
    except Exception as e:
        return (name, "fail", str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', help='single input smpl_filtered .pkl clip')
    ap.add_argument('--out', help='single output .bvh path (with --pkl)')
    ap.add_argument('--input_dir', nargs='*',
                    help='one or more directories of smpl_filtered .pkl clips to batch-convert '
                         '(e.g. data/lafan1_smpl_filtered data/amass_smpl_filtered)')
    ap.add_argument('--output_dir', help='output directory for batch mode')
    ap.add_argument('--template', required=True,
                     help='reference SOMA .bvh file (hierarchy/offsets template), '
                          'e.g. from soma-retargeter/assets/motions/bvh/*.bvh')
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--skip_existing', action='store_true')
    args = ap.parse_args()

    if args.pkl:
        out_path = convert(args.pkl, args.template, args.out)
        verify_roundtrip(out_path, args.pkl)
        return

    assert args.input_dir and args.output_dir, "--input_dir and --output_dir required for batch mode"
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = []
    for in_dir in args.input_dir:
        prefix = os.path.basename(os.path.normpath(in_dir)).replace('_smpl_filtered', '')
        for pkl_path in sorted(glob.glob(os.path.join(in_dir, "*.pkl"))):
            name = os.path.splitext(os.path.basename(pkl_path))[0]
            out_path = os.path.join(args.output_dir, f"{prefix}__{name}.bvh")
            if args.skip_existing and os.path.exists(out_path):
                continue
            tasks.append((pkl_path, args.template, out_path))

    print(f"Converting {len(tasks):,} clips ({args.num_workers} workers)...")
    t0 = time.time()
    ok, fail = 0, 0
    with multiprocessing.Pool(args.num_workers) as pool:
        for i, (name, status, err) in enumerate(pool.imap_unordered(_convert_one_task, tasks, chunksize=8), 1):
            if status == "ok":
                ok += 1
            else:
                fail += 1
                print(f"[fail] {name}: {err}")
            if i % 200 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  {i}/{len(tasks)}  ok={ok} fail={fail}  [{rate:.1f}/s, {elapsed:.0f}s elapsed]", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {ok} converted, {fail} failed, {elapsed:.0f}s elapsed, output dir: {args.output_dir}")


if __name__ == '__main__':
    main()
