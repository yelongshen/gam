"""Convert LAFAN1 `.bvh` clips into `smpl_filtered`-style `.pkl` files.

Implements the pipeline documented in `RAW_LAFAN1_DATA_FORMAT.md` §6:

    1. Parse BVH -> world-space joints (classify_motions.parse_bvh, meters/Z-up)
    2. Extrapolate hand-tip joints 22/23 (no LAFAN1 equivalent)
    3. Resample 30 fps -> 50 fps (linear interpolation)
    4. Canonicalize frame-0 orientation (skeleton-geometry-derived rigid rotation)
    5. Derive pose_aa[:, :3] (root axis-angle) from the canonicalized joints,
       per frame, via the official_root_quat_w() convention
    6. Save {pose_aa, transl, smpl_joints, fps} as a .pkl

Usage:
    .venv_sim/bin/python convert_lafan_to_smpl_filtered.py \\
        --input /home/grease/egodata/downloads/lafan1_extracted \\
        --output data/split_test_smpl \\
        --limit 5
"""
import argparse
import glob
import os
import sys

import joblib
import numpy as np
import scipy.spatial.transform as sT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling imports

import classify_motions as C
import normalize_split_test as N
import stream_clip_mode2 as S


def convert_one(bvh_path):
    """Return dict(pose_aa, transl, smpl_joints, fps) or None on failure."""
    j_raw, root_rot_raw = C.parse_bvh(bvh_path)
    if j_raw is None:
        return None

    # Step 2: extrapolate hand-tips (joints 22/23) from forearm->hand vector.
    j_raw = j_raw.copy()
    j_raw[:, 22, :] = j_raw[:, 20, :] + (j_raw[:, 20, :] - j_raw[:, 18, :]) * 0.3
    j_raw[:, 23, :] = j_raw[:, 21, :] + (j_raw[:, 21, :] - j_raw[:, 19, :]) * 0.3

    # Step 3: resample 30 -> 50 fps.
    T0 = len(j_raw)
    src_fps = 30.0
    duration = (T0 - 1) / src_fps
    t_src = np.arange(T0) / src_fps
    t_tgt = np.arange(0, duration, 1.0 / N.TARGET_FPS)
    if len(t_tgt) < 4:
        return None
    j50 = N._interp_linear(t_src, j_raw, t_tgt)
    T = len(j50)

    # Step 4: canonicalize frame-0 orientation using skeleton geometry.
    up0 = j50[0, 12] - j50[0, 0]
    up0 = up0 / np.linalg.norm(up0)
    right0 = j50[0, 1] - j50[0, 2]
    right0 = right0 - up0 * np.dot(right0, up0)
    right0 = right0 / np.linalg.norm(right0)
    fwd0 = np.cross(up0, right0)
    F0 = np.column_stack([right0, fwd0, up0])
    F0_tgt = np.eye(3)
    R_align = F0_tgt @ F0.T
    aligned_joints = j50 @ R_align.T

    # Step 5: derive pose_aa[:, :3] from the canonicalized joints, per frame.
    up = aligned_joints[:, 12] - aligned_joints[:, 0]
    up = up / np.linalg.norm(up, axis=-1, keepdims=True)
    right = aligned_joints[:, 1] - aligned_joints[:, 2]
    right = right - up * np.sum(right * up, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    fwd = np.cross(up, right)

    R_pol = np.stack([-fwd, right, up], axis=-1)
    root_q = sT.Rotation.from_matrix(R_pol).as_quat()  # x,y,z,w
    root_q = np.stack([root_q[:, 3], root_q[:, 0], root_q[:, 1], root_q[:, 2]], axis=-1)

    y_inv = S._qconj(S._YTOZ)[None, :]
    b_inv = S._qconj(S._BASE_CONJ)[None, :]
    q1 = S._qmul(np.broadcast_to(y_inv, (T, 4)), root_q)
    aa_quat = S._qmul(q1, np.broadcast_to(b_inv, (T, 4)))

    th = 2 * np.arccos(np.clip(aa_quat[:, 0:1], -1.0, 1.0))
    ax = aa_quat[:, 1:] / (np.linalg.norm(aa_quat[:, 1:], axis=-1, keepdims=True) + 1e-12)
    pose_aa = th * ax

    pose_aa_full = np.zeros((T, 72), dtype=np.float32)
    pose_aa_full[:, :3] = pose_aa

    transl = aligned_joints[:, 0, :].copy()
    smpl_joints = aligned_joints - aligned_joints[:, 0:1, :] + N.PELVIS_OFFSET

    return {
        "pose_aa": pose_aa_full.astype(np.float32),
        "transl": transl.astype(np.float32),
        "smpl_joints": smpl_joints.astype(np.float32),
        "fps": N.TARGET_FPS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/grease/egodata/downloads/lafan1_extracted")
    ap.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "lafan1_smpl_filtered"))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--files", nargs="*", default=None,
                    help="explicit list of .bvh basenames to convert (overrides --limit)")
    ap.add_argument("--prefix", default="",
                    help="optional filename prefix for outputs, e.g. 'TEST_' "
                         "(default: none, since train/test split happens afterward)")
    args = ap.parse_args()

    if args.files:
        bvh_files = [os.path.join(args.input, f) for f in args.files]
    else:
        bvh_files = sorted(glob.glob(os.path.join(args.input, "*.bvh")))
        if args.limit > 0:
            bvh_files = bvh_files[:args.limit]

    os.makedirs(args.output, exist_ok=True)
    ok, failed = 0, 0
    for bvh_path in bvh_files:
        name = os.path.splitext(os.path.basename(bvh_path))[0]
        out_path = os.path.join(args.output, f"{args.prefix}{name}.pkl")
        try:
            d = convert_one(bvh_path)
            if d is None:
                print(f"[skip] {name}: parse/resample failed")
                failed += 1
                continue
            joblib.dump(d, out_path)
            print(f"[ok]   {name}: {len(d['pose_aa'])} frames -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"[fail] {name}: {e}")
            failed += 1

    print(f"\nDone: {ok} converted, {failed} failed, output dir: {args.output}")


if __name__ == "__main__":
    main()
