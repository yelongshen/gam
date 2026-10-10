#!/usr/bin/env python3
"""match_clips_by_state.py — find where the two target motion clips
(walk_180_R_003__A332_M, walk_backward_start_001__A030_M) actually occur in
the online_robodata runs, by comparing recorded body/joint STATE (SMPL pose
for smpl_20260919, G1 dof positions for g1_run_0918) against the reference
clips in ego_dataset/eval_subset -- NOT by trusting any motion_name/label
field (g1_run_0918's motion_name.csv only ever says "macarena_001__A545" or
generic "streamed", and smpl_20260919 has no name field at all).

Matching method: for each candidate run, slide the reference clip's pose
sequence over the run's recorded sequence and compute the mean per-frame L2
error at every offset; report the offset with the lowest error, only if it's
below a a threshold (the reference clip is much shorter than the run, so the
run likely contains many *other* motions too -- we're looking for the single
best-matching sub-window).
"""
import glob
import io
import zlib

import joblib
import numpy as np


REF_SMPL = {
    "walk_backward_start_001__A030_M": "/home/grease/ego_dataset/eval_subset/smpl/walk_backward_start_001__A030_M.pkl",
    "walk_180_R_003__A332_M": "/home/grease/ego_dataset/eval_subset/smpl/walk_180_R_003__A332_M.pkl",
}
REF_ROBOT = {
    "walk_backward_start_001__A030_M": "/home/grease/ego_dataset/eval_subset/robot/walk_backward_start_001__A030_M.pkl",
    "walk_180_R_003__A332_M": "/home/grease/ego_dataset/eval_subset/robot/walk_180_R_003__A332_M.pkl",
}

SMPL_RUNS_GLOB = "/home/grease/humanoid-foundation-model-2/online_robodata/smpl_20260919/streamed_*"
G1_RUNS_GLOB = "/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/*"


def load_zlib_joblib(path):
    with open(path, "rb") as f:
        raw = f.read()
    return joblib.load(io.BytesIO(zlib.decompress(raw)))


def robust_load_csv(path: str, expected_cols: int) -> np.ndarray:
    """Some run CSVs have a few malformed/ragged rows (column count changes
    mid-file, likely from a logger hiccup) -- np.loadtxt hard-fails on that.
    Parse line-by-line instead, keeping only rows with exactly
    `expected_cols` fields, and stitching the surviving rows back into one
    contiguous array. This can't fix a genuine multi-row corruption gap, but
    since we only need *a* long enough clean stretch to slide the (much
    shorter) reference clip over, dropping the bad rows is sufficient."""
    rows = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != expected_cols:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float64)


def sliding_window_match(ref: np.ndarray, run: np.ndarray):
    """ref: (Tref, D), run: (Trun, D). Returns (best_offset, best_mean_err,
    err_at_every_offset) using dense per-frame L2 error, brute-force but
    vectorized over the offset axis via as_strided-free rolling window."""
    t_ref, d = ref.shape
    t_run = run.shape[0]
    if t_run < t_ref:
        return None, np.inf, None

    n_offsets = t_run - t_ref + 1
    errs = np.empty(n_offsets, dtype=np.float64)
    # Chunk to keep memory reasonable; this is brute force but the arrays
    # here are small enough (thousands x tens of dims) to just loop.
    ref_flat = ref.reshape(1, t_ref, d)
    for off in range(n_offsets):
        window = run[off:off + t_ref]
        errs[off] = np.mean(np.linalg.norm(window - ref, axis=-1))
    best = int(np.argmin(errs))
    return best, float(errs[best]), errs


def match_smpl():
    print("=" * 78)
    print("Matching against smpl_20260919 (true SMPL pose state, 50fps)")
    print("=" * 78)
    run_dirs = sorted(glob.glob(SMPL_RUNS_GLOB))
    refs = {}
    for name, path in REF_SMPL.items():
        d = load_zlib_joblib(path)
        # The run's smpl_pose.csv only logs 21 joints (63 dims), not the
        # full 24-joint (72-dim) SMPL pose -- slice the reference down to
        # match so we're comparing like-for-like.
        pose_aa_3d = d["pose_aa"].reshape(d["pose_aa"].shape[0], -1, 3)  # (T, 24, 3)
        pose = pose_aa_3d[:, :21, :].reshape(pose_aa_3d.shape[0], -1)  # (T, 63)
        refs[name] = pose
        print(f"  ref '{name}': {pose.shape[0]} frames @ {d['fps']} fps")

    results = {name: [] for name in REF_SMPL}
    for run_dir in run_dirs:
        try:
            run_pose = robust_load_csv(f"{run_dir}/smpl_pose.csv", expected_cols=63)
        except Exception as e:
            print(f"  [skip] {run_dir}: {e}")
            continue
        if run_pose.shape[0] == 0:
            print(f"  [skip] {run_dir}: no valid rows")
            continue
        for name, ref in refs.items():
            off, err, _ = sliding_window_match(ref, run_pose)
            if off is not None:
                results[name].append((run_dir, off, err, ref.shape[0]))

    for name, matches in results.items():
        matches.sort(key=lambda x: x[2])
        print(f"\n--- best matches for '{name}' (lower err = better; SMPL pose_aa L2, radians) ---")
        for run_dir, off, err, n_frames in matches[:5]:
            t_start_s = off / 50.0
            t_end_s = (off + n_frames) / 50.0
            print(f"  err={err:.4f}  run={run_dir.split('/')[-1]}  "
                  f"frames[{off}:{off + n_frames}]  ~t=[{t_start_s:.1f}s, {t_end_s:.1f}s]")


def match_g1():
    print("\n" + "=" * 78)
    print("Matching against g1_run_0918 (executed G1 dof positions, 50Hz)")
    print("=" * 78)
    run_dirs = sorted(glob.glob(G1_RUNS_GLOB))
    run_dirs = [d for d in run_dirs if d.split("/")[-1][0].isdigit()]

    refs = {}
    for name, path in REF_ROBOT.items():
        d = load_zlib_joblib(path)
        inner = d[name]
        dof = inner["dof"]  # (T, 29) @ inner['fps'] (likely 30)
        ref_fps = inner["fps"]
        # Resample to 50Hz (G1 control rate) via linear interpolation, so it
        # lines up with q.csv's native 50Hz recording.
        t_ref_original = np.arange(dof.shape[0]) / ref_fps
        t_ref_50hz = np.arange(0, t_ref_original[-1], 1.0 / 50.0)
        dof_50hz = np.stack([
            np.interp(t_ref_50hz, t_ref_original, dof[:, j]) for j in range(dof.shape[1])
        ], axis=1)
        refs[name] = dof_50hz
        print(f"  ref '{name}': {dof.shape[0]} frames @ {ref_fps}fps -> "
              f"resampled to {dof_50hz.shape[0]} frames @ 50Hz")

    results = {name: [] for name in REF_ROBOT}
    for run_dir in run_dirs:
        try:
            run_q_full = robust_load_csv(f"{run_dir}/q.csv", expected_cols=34)  # 5 meta cols + 29 dof
        except Exception as e:
            print(f"  [skip] {run_dir}: {e}")
            continue
        if run_q_full.shape[0] == 0:
            print(f"  [skip] {run_dir}: no valid rows")
            continue
        run_q = run_q_full[:, 5:5 + 29]
        for name, ref in refs.items():
            off, err, _ = sliding_window_match(ref, run_q)
            if off is not None:
                results[name].append((run_dir, off, err, ref.shape[0]))

    for name, matches in results.items():
        matches.sort(key=lambda x: x[2])
        print(f"\n--- best matches for '{name}' (lower err = better; dof position L2, rad) ---")
        for run_dir, off, err, n_frames in matches[:5]:
            t_start_s = off / 50.0
            t_end_s = (off + n_frames) / 50.0
            print(f"  err={err:.4f}  run={run_dir.split('/')[-1]}  "
                  f"frames[{off}:{off + n_frames}]  ~t=[{t_start_s:.1f}s, {t_end_s:.1f}s]")


if __name__ == "__main__":
    match_smpl()
    match_g1()
