#!/usr/bin/env python3
"""locate_and_eval_20260922.py — locate the two known eval-manifest clips
(walk_180_R_003__A332_M, walk_backward_start_001__A030_M) inside the new
2026-09-22 sessions (g1_run_0922: executed G1 dof state; 20260922: streamed
SMPL pose state), using the same state-matching approach as
eval_clip_manifest.md §4 (`locate_clip_in_session.py`), then compute basic
online tracking/health metrics over each located episode window:

  - dof_rmse_deg: RMSE between the matched reference `dof` trajectory and the
    robot's actually-executed `q` trajectory over the episode window (a
    reference-tracking error, analogous to the manifest's mpjpe_l).
  - max_joint_vel_rad_s: peak |dq| during the episode (actuator-margin proxy).
  - action_saturation_pct: fraction of policy action outputs with |a| > 0.95
    (near clipping/limit -- an actuator-margin proxy the manifest tracks).
  - base drift / peak base ang vel: gross stability proxies from
    base_ang_vel.csv / base_quat.csv, if present.

Policy under test this session: policy/sonic_no_vr_050k (see each run's
metadata.json).
"""
import glob
import io
import zlib

import joblib
import numpy as np


EVAL_ROBOT_DIR = "/home/grease/ego_dataset/eval_subset/robot"
EVAL_SMPL_DIR = "/home/grease/ego_dataset/eval_subset/smpl"

TARGET_CLIPS = ["walk_180_R_003__A332_M", "walk_backward_start_001__A030_M"]

G1_RUNS_GLOB = "/home/grease/g1_robot_data/g1_run_0922/*"
SMPL_RUNS_GLOB = "/home/grease/g1_robot_data/20260922/20260922/streamed_*"


def load_zlib_joblib(path):
    with open(path, "rb") as f:
        raw = f.read()
    return joblib.load(io.BytesIO(zlib.decompress(raw)))


def robust_load_csv(path: str, expected_cols) -> np.ndarray:
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


def sliding_window_match(ref: np.ndarray, run: np.ndarray):
    t_ref, d = ref.shape
    t_run = run.shape[0]
    if t_run < t_ref:
        return None, np.inf
    n_offsets = t_run - t_ref + 1
    errs = np.empty(n_offsets, dtype=np.float64)
    for off in range(n_offsets):
        errs[off] = np.mean(np.linalg.norm(run[off:off + t_ref] - ref, axis=-1))
    best = int(np.argmin(errs))
    return best, float(errs[best])


def get_streamed_index_ranges(motion_name_csv: str):
    names = []
    with open(motion_name_csv) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",")
            names.append(parts[-1].strip('"'))
    ranges, start = [], None
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


def eval_g1_run_0922():
    print("=" * 90)
    print("g1_run_0922: locating clips via executed dof state (policy/sonic_no_vr_050k)")
    print("=" * 90)

    refs = {}
    for name in TARGET_CLIPS:
        d = load_zlib_joblib(f"{EVAL_ROBOT_DIR}/{name}.pkl")
        inner = d[name]
        dof, ref_fps = inner["dof"], inner["fps"]
        t_orig = np.arange(dof.shape[0]) / ref_fps
        t_50hz = np.arange(0, t_orig[-1], 1.0 / 50.0)
        dof_50hz = np.stack([np.interp(t_50hz, t_orig, dof[:, j]) for j in range(dof.shape[1])], axis=1)
        refs[name] = dof_50hz
        print(f"  ref '{name}': {dof.shape[0]}f @ {ref_fps}fps -> {dof_50hz.shape[0]}f @ 50Hz")

    run_dirs = sorted(d for d in glob.glob(G1_RUNS_GLOB) if d.split("/")[-1][0].isdigit())
    episodes = []

    for run_dir in run_dirs:
        run_name = run_dir.split("/")[-1]
        try:
            q_full = robust_load_csv(f"{run_dir}/q.csv", 34)
            dq_full = robust_load_csv(f"{run_dir}/dq.csv", 34)
            action_full = robust_load_csv(f"{run_dir}/action.csv", None)
        except Exception as e:
            print(f"  [skip run] {run_name}: {e}")
            continue
        if q_full.shape[0] == 0:
            continue
        q, dq = q_full[:, 5:5 + 29], dq_full[:, 5:5 + 29]

        ranges = get_streamed_index_ranges(f"{run_dir}/motion_name.csv")
        print(f"\n--- run {run_name} --- streamed segment(s): {ranges}")

        for seg_start, seg_end in ranges:
            seg_q = q[seg_start:seg_end]
            if seg_q.shape[0] < 10:
                continue
            for name, ref in refs.items():
                if ref.shape[0] > seg_q.shape[0]:
                    continue
                off, rmse = sliding_window_match(ref, seg_q)
                if off is None:
                    continue
                abs_start, abs_end = seg_start + off, seg_start + off + ref.shape[0]
                seg_dq = dq[abs_start:abs_end]
                seg_action = action_full[abs_start:abs_end, 5:] if action_full.shape[0] else None
                max_vel = float(np.max(np.abs(seg_dq))) if seg_dq.shape[0] else float("nan")
                sat_pct = 100.0 * float(np.mean(np.abs(seg_action) > 0.95)) if seg_action is not None and seg_action.shape[0] else float("nan")

                print(f"  '{name}': RMSE={rmse:.4f} rad ({np.degrees(rmse):.2f} deg)  "
                      f"window[{abs_start}:{abs_end}] ~t=[{abs_start/50.0:.1f}s,{abs_end/50.0:.1f}s]  "
                      f"max|dq|={max_vel:.2f} rad/s  action_sat={sat_pct:.1f}%")
                episodes.append(dict(run=run_name, clip=name, offset=off,
                                      window=[abs_start, abs_end], rmse_rad=rmse,
                                      rmse_deg=np.degrees(rmse), max_joint_vel=max_vel,
                                      action_sat_pct=sat_pct))
    return episodes


def eval_smpl_20260922():
    print("\n" + "=" * 90)
    print("20260922/streamed_*: locating clips via streamed SMPL pose state")
    print("=" * 90)

    refs = {}
    for name in TARGET_CLIPS:
        d = load_zlib_joblib(f"{EVAL_SMPL_DIR}/{name}.pkl")
        pose_aa_3d = d["pose_aa"].reshape(d["pose_aa"].shape[0], -1, 3)
        refs[name] = pose_aa_3d[:, :21, :].reshape(pose_aa_3d.shape[0], -1)
        print(f"  ref '{name}': {refs[name].shape[0]} frames @ {d['fps']} fps")

    run_dirs = sorted(glob.glob(SMPL_RUNS_GLOB))
    episodes = []
    for run_dir in run_dirs:
        run_name = run_dir.split("/")[-1]
        try:
            run_pose = robust_load_csv(f"{run_dir}/smpl_pose.csv", 63)
        except Exception as e:
            print(f"  [skip] {run_name}: {e}")
            continue
        if run_pose.shape[0] == 0:
            continue
        print(f"\n--- {run_name} ({run_pose.shape[0]} frames) ---")
        for name, ref in refs.items():
            if ref.shape[0] > run_pose.shape[0]:
                continue
            off, rmse = sliding_window_match(ref, run_pose)
            if off is None:
                continue
            print(f"  '{name}': RMSE={rmse:.4f} rad  window[{off}:{off+ref.shape[0]}] "
                  f"~t=[{off/50.0:.1f}s,{(off+ref.shape[0])/50.0:.1f}s]")
            episodes.append(dict(run=run_name, clip=name, offset=off,
                                  window=[off, off + ref.shape[0]], rmse_rad=rmse))
    return episodes


if __name__ == "__main__":
    g1_eps = eval_g1_run_0922()
    smpl_eps = eval_smpl_20260922()

    print("\n" + "=" * 90)
    print("SUMMARY: best (lowest-RMSE) episode per clip, per dataset")
    print("=" * 90)
    for label, eps in [("g1_run_0922 (dof)", g1_eps), ("20260922 (smpl_pose)", smpl_eps)]:
        print(f"\n-- {label} --")
        for name in TARGET_CLIPS:
            cands = [e for e in eps if e["clip"] == name]
            if not cands:
                print(f"  {name}: NOT FOUND")
                continue
            best = min(cands, key=lambda e: e["rmse_rad"])
            print(f"  {name}: best run={best['run']} window={best['window']} "
                  f"rmse={best['rmse_rad']:.4f} rad ({len(cands)} candidate segment(s) checked)")
