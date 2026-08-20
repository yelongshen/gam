"""Convert raw AMASS `.npz` clips into `smpl_filtered`-style `.pkl` files.

Implements the pipeline documented in `RAW_AMASS_DATA_FORMAT.md` /
`SMPL_FILTERED_DATA_FORMAT.md`, fixing the two schema-dependent bugs
identified there:

    1. Schema-aware parse: detect Schema A (SMPL-H, no `_stageii` suffix) vs
       Schema B (SMPL-X, `_stageii` suffix) and build `pose_aa[:, :72]`
       correctly for each (Bug #2 fix -- schema B's raw `poses[:, 66:72]`
       is jaw/eye data, NOT hand data).
    2. Schema-aware framerate: check both `mocap_frame_rate` (schema B) and
       `mocap_framerate` (schema A) key spellings (Bug #1 fix -- schema B's
       true key was previously never matched, silently defaulting to a
       wrong hardcoded 60.0 fps).
    3. Resample source fps -> 50 fps (linear interpolation).
    4. Canonicalize frame-0 root rotation via the SIMPLE, proven approach
       (`D = R0.T` directly from the root's own axis-angle) -- NOT the
       skeleton-geometry approach used for LAFAN/BVH, since AMASS pose_aa
       is already native SMPL convention (no BVH-vs-SMPL rest-frame
       mismatch to correct for; using neck-pelvis geometry here gets
       contaminated by spine posture and gives clip-dependent, non-zero
       results -- verified empirically).
    5. Derive `smpl_joints` via FK, KEEPING root rotation (per
       SMPL_FILTERED_DATA_FORMAT.md Sec 5.2 -- real smpl_filtered does NOT
       heading-canonicalize the body, only pins the pelvis translation),
       Z-up remap, pelvis-pinned at PELVIS_OFFSET.
    6. Save {pose_aa, transl, smpl_joints, fps, original_pose_aa,
       original_fps} as a .pkl.

Usage:
    .venv_sim/bin/python convert_amass_to_smpl_filtered.py \\
        --input /home/grease/egodata/downloads/amass/extracted \\
        --output data/amass_smpl_filtered \\
        --limit 5
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

import fix_amass as F

TARGET_FPS = 50.0
PELVIS_OFFSET = np.array([0.003, -0.351, 0.012])
# Raw SMPL FK is Y-up; smpl_filtered's smpl_joints convention is Z-up:
# (x, y, z) -> (x, -z, y)
ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def _interp_linear(x_src, y_src, x_tgt):
    """Linear interpolation along axis 0, applied independently per
    trailing component (thin wrapper around np.interp)."""
    flat = y_src.reshape(len(y_src), -1)
    out = np.stack([np.interp(x_tgt, x_src, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return out.reshape((len(x_tgt),) + y_src.shape[1:])


def load_amass_fixed(npz_path):
    """Schema-aware AMASS loader. Returns (pose72, trans, src_fps, schema)."""
    d = np.load(npz_path, allow_pickle=True)

    # Bug #1 fix: framerate key mismatch (schema B uses 'mocap_frame_rate',
    # schema A uses 'mocap_framerate' -- checking only one silently breaks
    # the other).
    if 'mocap_frame_rate' in d.files:
        src_fps = float(d['mocap_frame_rate'])
    elif 'mocap_framerate' in d.files:
        src_fps = float(d['mocap_framerate'])
    else:
        src_fps = 60.0  # last-resort fallback only

    # Bug #2 fix: schema-aware poses[:, :72] slice. Schema B's byte offsets
    # 66:72 are pose_jaw + half of pose_eye, NOT hand data.
    schema = 'B' if 'pose_hand' in d.files else 'A'
    if schema == 'B':
        pose72 = np.concatenate(
            [d['root_orient'], d['pose_body'], d['pose_hand'][:, :6]], axis=1)
    else:
        pose72 = d['poses'][:, :72]

    trans = d['trans'] if 'trans' in d.files else np.zeros((len(pose72), 3))
    return pose72.astype(np.float64), trans.astype(np.float64), src_fps, schema


def canonicalize_root_rotation(pose_aa, trans):
    """Zero the root joint's OWN rotation at frame 0. Simple + exact for
    AMASS since pose_aa is already native SMPL axis-angle (unlike LAFAN's
    BVH rest-frame, which needed a skeleton-geometry-derived correction)."""
    R0 = F.axis_angle_to_matrix(pose_aa[0, :3])
    D = R0.T  # D @ R0 = Identity

    root_mats = np.stack([F.axis_angle_to_matrix(aa) for aa in pose_aa[:, :3]], axis=0)
    new_root_mats = np.einsum('ij,tjk->tik', D, root_mats)
    new_root_aa = sT.Rotation.from_matrix(new_root_mats).as_rotvec()

    pose_aa_fixed = pose_aa.copy()
    pose_aa_fixed[:, :3] = new_root_aa
    trans_fixed = trans @ D.T
    return pose_aa_fixed, trans_fixed


def convert_one(npz_path):
    """Return dict(pose_aa, transl, smpl_joints, fps, original_pose_aa,
    original_fps) or None on failure."""
    pose72_raw, trans_raw, src_fps, schema = load_amass_fixed(npz_path)
    T0 = len(pose72_raw)
    if T0 < 2:
        return None
    duration = (T0 - 1) / src_fps
    t_src = np.arange(T0) / src_fps
    t_tgt = np.arange(0, duration, 1.0 / TARGET_FPS)
    if len(t_tgt) < 4:
        return None

    pose72 = _interp_linear(t_src, pose72_raw, t_tgt)
    trans = _interp_linear(t_src, trans_raw, t_tgt)

    pose_aa, transl = canonicalize_root_rotation(pose72, trans)

    # smpl_joints: KEEP root rotation (real smpl_filtered convention), Z-up,
    # pelvis-pinned.
    joints_raw = F.smpl_fk(pose_aa, np.zeros((len(pose_aa), 3)))
    joints_zup = joints_raw @ ZUP.T
    smpl_joints = joints_zup - joints_zup[:, 0:1, :] + PELVIS_OFFSET

    T = len(pose_aa)
    pose_aa_full = np.zeros((T, 72), dtype=np.float32)
    pose_aa_full[:, :pose_aa.shape[1]] = pose_aa

    return {
        "pose_aa": pose_aa_full,
        "transl": transl.astype(np.float32),
        "smpl_joints": smpl_joints.astype(np.float32),
        "fps": TARGET_FPS,
        "original_pose_aa": pose72_raw.astype(np.float32),
        "original_fps": src_fps,
    }


def _unique_name(npz_path, input_root):
    """Build a collision-free output basename from the path relative to
    --input (1904 of 17892 AMASS files share a basename across different
    sub-datasets/subjects, e.g. 'walk_poses.npz' -- using only
    os.path.basename() would silently overwrite files)."""
    rel = os.path.relpath(npz_path, input_root)
    rel = os.path.splitext(rel)[0]
    return rel.replace(os.sep, "__").replace(" ", "_")


def _convert_one_task(args_tuple):
    npz_path, out_path = args_tuple
    name = os.path.basename(out_path)
    try:
        d = convert_one(npz_path)
        if d is None:
            return (name, "skip", "too short / parse failed")
        joblib.dump(d, out_path)
        return (name, "ok", len(d["pose_aa"]))
    except Exception as e:
        return (name, "fail", str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/grease/egodata/downloads/amass/extracted")
    ap.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "amass_smpl_filtered"))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--files", nargs="*", default=None,
                    help="explicit list of .npz paths (absolute or relative "
                         "to --input) to convert (overrides --limit)")
    ap.add_argument("--num_workers", type=int, default=8,
                    help="parallel worker processes (default 8)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip files whose output .pkl already exists")
    args = ap.parse_args()

    if args.files:
        npz_files = [f if os.path.isabs(f) else os.path.join(args.input, f)
                     for f in args.files]
    else:
        all_files = glob.glob(os.path.join(args.input, "**/*.npz"), recursive=True)
        # skip shape/stagei-only files (no motion data), same filter used
        # elsewhere in this pipeline (classify_motions.py main()).
        npz_files = sorted(
            f for f in all_files
            if all(t not in os.path.basename(f).lower()
                   for t in ('shape', 'stagei.', 'neutral')))
        if args.limit > 0:
            npz_files = npz_files[:args.limit]

    os.makedirs(args.output, exist_ok=True)

    tasks = []
    for npz_path in npz_files:
        name = _unique_name(npz_path, args.input)
        out_path = os.path.join(args.output, f"{name}.pkl")
        if args.skip_existing and os.path.exists(out_path):
            continue
        tasks.append((npz_path, out_path))

    print(f"Converting {len(tasks):,} AMASS clips ({args.num_workers} workers)...")
    t0 = time.time()
    ok, skip, fail = 0, 0, 0
    with multiprocessing.Pool(args.num_workers) as pool:
        for i, (name, status, info) in enumerate(pool.imap_unordered(_convert_one_task, tasks, chunksize=16), 1):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
                print(f"[skip] {name}: {info}")
            else:
                fail += 1
                print(f"[fail] {name}: {info}")
            if i % 500 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  {i}/{len(tasks)}  ok={ok} skip={skip} fail={fail}  "
                      f"[{rate:.0f}/s, {elapsed:.0f}s elapsed]", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {ok} converted, {skip} skipped, {fail} failed, "
          f"{elapsed:.0f}s elapsed, output dir: {args.output}")


if __name__ == "__main__":
    main()
