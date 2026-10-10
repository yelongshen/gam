"""Build `ego_dataset/picoset_20260929/` (robot/ + smpl/) from the 2026-09-29
PICO teleop clips -- 163 clips chunked from session `20260929_174102`
(84,742 frames @ ~86 Hz), recorded with REAL root translation
(`--record_root_pos`, converted with `--transl_mode body_pos_w`).

Stage 5 of `build_picoset_20260929.sh` -- run that driver rather than this
script directly unless the upstream stages are already done.

Same layout/convention as `eval_subset/`, `pico_evalset/` and
`picoset_20260924/`:

    smpl:  50 fps, `smpl_filtered` format  (gam/logs_pkl/picoset_20260929)
    robot: 30 fps, `motion_lib` format     (GMR/..._retargeted_g1_fps30_motion_lib)

The two branches are retargeted SEPARATELY (50 fps smpl pkl for the smpl side,
30 fps smpl pkl -> 30 fps retarget for the robot side) so each lands on its
required native rate, then `gmr.tools.fix_evalset_frame_mismatch.align_pair()`
reconciles the counts exactly as `build_picoset_20260924.py` does -- so the
result never trips motion_lib_base.py's

    assert curr_motion["smpl_joints"].shape[0] == curr_motion["global_translation"].shape[1]

Usage::

    python data_process/build_picoset_20260929.py
"""

import glob
import importlib.util
import os
import sys

import joblib

SMPL_DIR = "/home/grease/gam/logs_pkl/picoset_20260929"                              # 50 fps
ROBOT_DIR = "/home/grease/GMR/picoset_20260929_retargeted_g1_fps30_motion_lib"       # 30 fps
GMR_TOOLS_PATH = "/home/grease/gamc/gmr/tools/fix_evalset_frame_mismatch.py"
OUT_DIR = "/home/grease/ego_dataset/picoset_20260929"


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
