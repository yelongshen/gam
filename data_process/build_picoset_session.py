"""Stage 5 of the build_picoset_*.sh chain for any session: pair the 50 fps smpl clips
(stage 1) with the 30 fps motion_lib robot clips (stage 4) into `<out>/{robot,smpl}/`, frame
counts aligned with gmr.tools.fix_evalset_frame_mismatch.align_pair (same as
build_picoset_20261015.py, which hardcodes its session's paths).

Usage (on grease)::

    /home/grease/miniforge3/envs/env_isaaclab/bin/python data_process/build_picoset_session.py \
        --smpl-dir /home/grease/gam/logs_pkl/picoset_20261020 \
        --robot-dir /home/grease/GMR/picoset_20261020_retargeted_g1_fps30_motion_lib \
        --out /home/grease/ego_dataset/picoset_20261020

Staircase envs (`envs/`) are added afterwards by hfm
dev_notes/stair_subset_build/pico_stairs_to_env.py --session ... --dir <out>.
"""

import argparse
import glob
import importlib.util
import os
import sys

import joblib

GMR_TOOLS_PATH = "/home/grease/gamc/gmr/tools/fix_evalset_frame_mismatch.py"


def load_gmr_align_module():
    spec = importlib.util.spec_from_file_location("fix_evalset_frame_mismatch",
                                                  GMR_TOOLS_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fix_evalset_frame_mismatch"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl-dir", required=True, help="stage 1 output, 50 fps")
    ap.add_argument("--robot-dir", required=True, help="stage 4 output, 30 fps motion_lib")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gmr-tools", default=GMR_TOOLS_PATH)
    a = ap.parse_args()
    SMPL_DIR, ROBOT_DIR, OUT_DIR = a.smpl_dir, a.robot_dir, a.out
    globals()["GMR_TOOLS_PATH"] = a.gmr_tools
    if os.path.exists(OUT_DIR) and os.listdir(OUT_DIR):
        raise SystemExit(f"{OUT_DIR} exists and is not empty")
    gmr = load_gmr_align_module()

    robot_files = {os.path.splitext(os.path.basename(f))[0]: f
                   for f in glob.glob(os.path.join(ROBOT_DIR, "*.pkl"))}
    smpl_files = {os.path.splitext(os.path.basename(f))[0]: f
                  for f in glob.glob(os.path.join(SMPL_DIR, "*.pkl"))}
    common = sorted(set(robot_files) & set(smpl_files))

    print(f"robot clips: {len(robot_files)}   smpl clips: {len(smpl_files)}")
    print(f"matched pairs: {len(common)}")
    for extra, label in ((sorted(set(robot_files) - set(smpl_files)), "robot-only"),
                         (sorted(set(smpl_files) - set(robot_files)), "smpl-only")):
        if extra:
            print(f"  {label} (skipped): {extra}")

    out_robot = os.path.join(OUT_DIR, "robot")
    out_smpl = os.path.join(OUT_DIR, "smpl")
    os.makedirs(out_robot, exist_ok=True)
    os.makedirs(out_smpl, exist_ok=True)

    status_counts = {}
    n_ok = n_failed = 0

    for name in common:
        try:
            motion_key, robot_vals = gmr.unwrap_robot(joblib.load(robot_files[name]))
            smpl_vals = joblib.load(smpl_files[name])

            new_robot_vals, new_smpl_vals, status = gmr.align_pair(
                robot_vals, smpl_vals, gmr.TARGET_FPS)
            status_counts[status] = status_counts.get(status, 0) + 1

            rn = gmr.robot_resampled_num_frames(
                new_robot_vals["root_trans_offset"].shape[0],
                new_robot_vals.get("fps", 30.0), gmr.TARGET_FPS)
            sn = (new_smpl_vals["smpl_joints"].shape[0] if "smpl_joints" in new_smpl_vals
                  else new_smpl_vals["pose_aa"].shape[0])
            flag = "OK " if rn == sn else "MISMATCH"
            print(f"[{status:22s}] {name:32s} robot@50={rn:5d} smpl={sn:5d} {flag}")

            joblib.dump(new_smpl_vals, os.path.join(out_smpl, f"{name}.pkl"),
                        compress=True)
            joblib.dump({motion_key: new_robot_vals},
                        os.path.join(out_robot, f"{name}.pkl"), compress=True)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {name}: {e}")
            n_failed += 1

    print(f"\nBuilt {OUT_DIR}:")
    print(f"  smpl/  -> {n_ok} files (50 fps)")
    print(f"  robot/ -> {n_ok} files (30 fps)")
    print(f"  failed: {n_failed}")
    print("\nFrame alignment breakdown:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
