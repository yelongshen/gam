#!/usr/bin/env python3
"""detect_clips_and_eval.py — for the 2026-09-22 online robot data
(g1_robot_data/g1_run_0922: executed G1 dof state; g1_robot_data/20260922:
streamed SMPL state), detect which eval_subset motion clip was actually
running during each "streamed" segment by matching recorded body/joint
STATE against the ~139 reference clips in ego_dataset/eval_subset (not by
trusting motion_name.csv, which only ever says "macarena_001__A545" or
generic "streamed" -- see prior analysis in this session), and then compute
basic online tracking/health metrics for the policy over each matched
segment.

Matching is done via FFT cross-correlation (O(D*(N+T)log(N+T)) per
clip/run pair) instead of a brute-force per-offset Python loop, since we now
search ~139 candidate clips x 5 runs x 2 datasets.
"""
import glob
import io
import json
import zlib

import joblib
import numpy as np
from scipy.signal import fftconvolve


EVAL_ROBOT_DIR = "/home/grease/ego_dataset/eval_subset/robot"
EVAL_SMPL_DIR = "/home/grease/ego_dataset/eval_subset/smpl"

G1_RUNS_GLOB = "/home/grease/g1_robot_data/g1_run_0922/*"
SMPL_RUNS_GLOB = "/home/grease/g1_robot_data/20260922/20260922/streamed_*"


def load_zlib_joblib(path):
    with open(path, "rb") as f:
        raw = f.read()
    return joblib.load(io.BytesIO(zlib.decompress(raw)))


def robust_load_csv(path: str, expected_cols: int) -> np.ndarray:
    rows = []
    with open(path) as f:
        header = next(f).strip().split(",")
        if expected_cols is None:
            expected_cols = len(header)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != expected_cols:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float64)


def fast_sliding_match(ref: np.ndarray, run: np.ndarray):
    """ref: (Tref, D), run: (Trun, D). Returns (best_offset, best_rmse) via
    FFT cross-correlation, minimizing mean squared L2 distance per offset.
    """
    t_ref, d = ref.shape
    t_run = run.shape[0]
    if t_run < t_ref:
        return None, np.inf

    # sliding sum of run^2 over a length-t_ref window, per offset (summed
    # across dims), via cumulative sum.
    run_sq = (run ** 2).sum(axis=1)
    cumsum = np.concatenate([[0.0], np.cumsum(run_sq)])
    window_sumsq_run = cumsum[t_ref:] - cumsum[:-t_ref]  # (Trun - t_ref + 1,)

    ref_sumsq = float((ref ** 2).sum())

    # cross term: sum_t run[off+t, dim] * ref[t, dim], summed over dim, for
    # every valid offset -- via 'valid'-mode correlation per dim (reverse the
    # ref kernel for correlation via convolution).
    cross = np.zeros(t_run - t_ref + 1, dtype=np.float64)
    for dim in range(d):
        cross += fftconvolve(run[:, dim], ref[::-1, dim], mode="valid")

    sq_dist = window_sumsq_run - 2.0 * cross + ref_sumsq
    sq_dist = np.maximum(sq_dist, 0.0)  # guard tiny negative FP noise
    best_offset = int(np.argmin(sq_dist))
    best_mse = sq_dist[best_offset] / t_ref
    return best_offset, float(np.sqrt(best_mse))


def get_streamed_index_ranges(motion_name_csv: str):
    """Parse motion_name.csv, return list of (start_idx, end_idx) contiguous
    ranges where motion_name == 'streamed' (i.e. NOT the named library clip
    'macarena_001__A545')."""
    names = []
    with open(motion_name_csv) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",")
            names.append(parts[-1].strip('"'))
    ranges = []
    start = None
    for i, n in enumerate(names):
        is_streamed = (n == "streamed")
        if is_streamed and start is None:
            start = i
        elif not is_streamed and start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(names)))
    return ranges


def detect_g1_run_0922():
    print("=" * 90)
    print("Detecting clips in g1_run_0922 (executed G1 dof state, 50Hz, policy/sonic_no_vr_050k)")
    print("=" * 90)

    ref_paths = sorted(glob.glob(f"{EVAL_ROBOT_DIR}/*.pkl"))
    refs = {}
    for p in ref_paths:
        name = p.split("/")[-1][:-4]
        try:
            d = load_zlib_joblib(p)
            inner = d[name]
            dof = inner["dof"]
            ref_fps = inner["fps"]
        except Exception:
            continue
        t_original = np.arange(dof.shape[0]) / ref_fps
        t_50hz = np.arange(0, t_original[-1], 1.0 / 50.0)
        dof_50hz = np.stack([
            np.interp(t_50hz, t_original, dof[:, j]) for j in range(dof.shape[1])
        ], axis=1)
        refs[name] = dof_50hz
    print(f"  loaded {len(refs)} reference clips (dof, resampled to 50Hz)")

    run_dirs = sorted(d for d in glob.glob(G1_RUNS_GLOB) if d.split("/")[-1][0].isdigit())

    report = []
    for run_dir in run_dirs:
        run_name = run_dir.split("/")[-1]
        try:
            q_full = robust_load_csv(f"{run_dir}/q.csv", expected_cols=34)
            dq_full = robust_load_csv(f"{run_dir}/dq.csv", expected_cols=34)
            action_full = robust_load_csv(f"{run_dir}/action.csv", expected_cols=None)
        except Exception as e:
            print(f"  [skip run] {run_name}: {e}")
            continue
        if q_full.shape[0] == 0:
            print(f"  [skip run] {run_name}: empty q.csv")
            continue
        q = q_full[:, 5:5 + 29]
        dq = dq_full[:, 5:5 + 29] if dq_full.shape[0] else None

        ranges = get_streamed_index_ranges(f"{run_dir}/motion_name.csv")
        print(f"\n--- run {run_name} --- {len(ranges)} streamed segment(s): {ranges}")

        for seg_start, seg_end in ranges:
            seg_q = q[seg_start:seg_end]
            if seg_q.shape[0] < 10:
                continue
            best_name, best_off, best_rmse = None, None, np.inf
            for name, ref in refs.items():
                if ref.shape[0] > seg_q.shape[0]:
                    continue
                off, rmse = fast_sliding_match(ref, seg_q)
                if rmse < best_rmse:
                    best_name, best_off, best_rmse = name, off, rmse

            if best_name is None:
                print(f"  segment[{seg_start}:{seg_end}] ({(seg_end-seg_start)/50.0:.1f}s): "
                      f"no candidate clip is short enough to fit")
                continue

            ref = refs[best_name]
            n_frames = ref.shape[0]
            abs_start = seg_start + best_off
            abs_end = abs_start + n_frames
            seg_dq = dq[abs_start:abs_end] if dq is not None else None
            seg_action = action_full[abs_start:abs_end, 5:] if action_full.shape[0] else None

            joint_err_deg = np.degrees(best_rmse)
            max_joint_vel = float(np.max(np.abs(seg_dq))) if seg_dq is not None and seg_dq.shape[0] else float("nan")
            action_sat_pct = float("nan")
            if seg_action is not None and seg_action.shape[0]:
                action_sat_pct = 100.0 * float(np.mean(np.abs(seg_action) > 0.95))

            print(f"  segment[{seg_start}:{seg_end}] -> best match: '{best_name}' "
                  f"(offset={best_off}, matched[{abs_start}:{abs_end}], "
                  f"~t=[{abs_start/50.0:.1f}s,{abs_end/50.0:.1f}s])")
            print(f"      dof RMSE={best_rmse:.4f} rad ({joint_err_deg:.2f} deg)  "
                  f"max|dq|={max_joint_vel:.2f} rad/s  action_saturation={action_sat_pct:.1f}%")

            report.append({
                "run": run_name, "segment": [seg_start, seg_end],
                "matched_clip": best_name, "matched_range": [abs_start, abs_end],
                "dof_rmse_rad": best_rmse, "dof_rmse_deg": joint_err_deg,
                "max_joint_vel_rad_s": max_joint_vel, "action_saturation_pct": action_sat_pct,
            })

    return report


def detect_smpl_20260922():
    print("\n" + "=" * 90)
    print("Detecting clips in 20260922/streamed_* (streamed SMPL pose state, 50fps)")
    print("=" * 90)

    ref_paths = sorted(glob.glob(f"{EVAL_SMPL_DIR}/*.pkl"))
    refs = {}
    for p in ref_paths:
        name = p.split("/")[-1][:-4]
        try:
            d = load_zlib_joblib(p)
            pose_aa_3d = d["pose_aa"].reshape(d["pose_aa"].shape[0], -1, 3)
            pose = pose_aa_3d[:, :21, :].reshape(pose_aa_3d.shape[0], -1)
        except Exception:
            continue
        refs[name] = pose
    print(f"  loaded {len(refs)} reference clips (SMPL pose_aa, 21 joints)")

    run_dirs = sorted(glob.glob(SMPL_RUNS_GLOB))
    report = []
    for run_dir in run_dirs:
        run_name = run_dir.split("/")[-1]
        try:
            run_pose = robust_load_csv(f"{run_dir}/smpl_pose.csv", expected_cols=63)
        except Exception as e:
            print(f"  [skip] {run_name}: {e}")
            continue
        if run_pose.shape[0] == 0:
            print(f"  [skip] {run_name}: no valid rows")
            continue

        best_name, best_off, best_rmse = None, None, np.inf
        for name, ref in refs.items():
            if ref.shape[0] > run_pose.shape[0]:
                continue
            off, rmse = fast_sliding_match(ref, run_pose)
            if rmse < best_rmse:
                best_name, best_off, best_rmse = name, off, rmse

        if best_name is None:
            print(f"  {run_name}: no candidate clip fits")
            continue
        n_frames = refs[best_name].shape[0]
        print(f"  {run_name} ({run_pose.shape[0]} frames): best match '{best_name}' "
              f"offset={best_off} [{best_off}:{best_off+n_frames}] "
              f"~t=[{best_off/50.0:.1f}s,{(best_off+n_frames)/50.0:.1f}s]  "
              f"pose_aa RMSE={best_rmse:.4f} rad")
        report.append({
            "run": run_name, "matched_clip": best_name,
            "matched_range": [best_off, best_off + n_frames],
            "pose_rmse_rad": best_rmse,
        })
    return report


if __name__ == "__main__":
    g1_report = detect_g1_run_0922()
    smpl_report = detect_smpl_20260922()

    with open("/tmp/detect_clips_and_eval_report.json", "w") as f:
        json.dump({"g1_run_0922": g1_report, "smpl_20260922": smpl_report}, f, indent=2)
    print("\nWrote /tmp/detect_clips_and_eval_report.json")
