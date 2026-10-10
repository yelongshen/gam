"""Build `ego_dataset/picoset_20260924/` (robot/ + smpl/) from the cleaned
2026-09-24 PICO teleop clips.

Same layout/convention as `eval_subset/` and `pico_evalset/`:

    smpl:  50 fps, `smpl_filtered` format  (gam/logs_pkl/pico_20260924)
    robot: 30 fps, `motion_lib` format     (GMR/pico20260924_retargeted_g1_motion_lib)

Applies the robot(fps30)-vs-SMPL(fps50) frame-count ALIGNMENT described in
`GR00T-WholeBodyControl/dev_notes/fps_check_alignment/README.md`, reusing
`gmr.tools.fix_evalset_frame_mismatch.align_pair()` exactly as
`build_pico_evalset.py` / `build_amass_trainset.py` do — so the result never
trips motion_lib_base.py's

    assert curr_motion["smpl_joints"].shape[0] == curr_motion["global_translation"].shape[1]

Usage::

    python data_process/build_picoset_20260924.py
"""

import glob
import importlib.util
import os
import sys

import joblib

SMPL_DIR = "/home/grease/gam/logs_pkl/pico_20260924"              # 50 fps
ROBOT_DIR = "/home/grease/GMR/pico20260924_retargeted_g1_motion_lib"  # 30 fps
GMR_TOOLS_PATH = "/home/grease/gamc/gmr/tools/fix_evalset_frame_mismatch.py"
OUT_DIR = "/home/grease/ego_dataset/picoset_20260924"


def load_gmr_align_module():
    spec = importlib.util.spec_from_file_location("fix_evalset_frame_mismatch",
                                                  GMR_TOOLS_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fix_evalset_frame_mismatch"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
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

    status_counts = {"already_aligned": 0, "trimmed_smpl": 0,
                     "trimmed_robot": 0, "padded_smpl_fallback": 0}
    n_ok = n_failed = 0

    for name in common:
        try:
            motion_key, robot_vals = gmr.unwrap_robot(joblib.load(robot_files[name]))
            smpl_vals = joblib.load(smpl_files[name])

            new_robot_vals, new_smpl_vals, status = gmr.align_pair(
                robot_vals, smpl_vals, gmr.TARGET_FPS)
            status_counts[status] = status_counts.get(status, 0) + 1

            rn = gmr.robot_resampled_num_frames(
                robot_vals["root_trans_offset"].shape[0],
                robot_vals.get("fps", 30.0), gmr.TARGET_FPS)
            sn = (smpl_vals["smpl_joints"].shape[0] if "smpl_joints" in smpl_vals
                  else smpl_vals["pose_aa"].shape[0])
            print(f"[{status:22s}] {name:46s} robot={rn:5d} smpl={sn:5d}")

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
