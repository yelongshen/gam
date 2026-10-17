#!/usr/bin/env bash
# picoset_20261015: two clips still had a retarget start-up fault with --yaw_normalize --settle 30:
#   pico1015_161659_clip_022  bowed IK (robot torso 74 deg more leaned than the human; the human faces yaw0=-131 deg)
#   pico1015_161659_clip_017  first-frame reference speed 12.9 rad/s
# Re-retargeting them WITHOUT --yaw_normalize (settle 30 kept) gives lean diff -1 deg / first-frame speed 1.7 (022) and 4.9 rad/s (017)
# (see compare_retarget_variants_20261015.py). Originals are kept in <raw>_pre_fix and <ego>/picoset_20261015_pre_fix.
set -euo pipefail
SET=picoset_20261015
GMR=/home/grease/GMR; GAM=/home/grease/gam; EGO=/home/grease/ego_dataset
PY_GMR=/home/grease/miniforge3/envs/gmr/bin/python
PY_BUILD=/home/grease/miniforge3/envs/env_isaaclab/bin/python
RAW=$GMR/${SET}_retargeted_g1_fps30; NPZ=$GMR/${SET}_smplx_npz_fps30; MOT=${RAW}_motion_lib; BAK=${RAW}_pre_fix
CLIPS=("$@")   # clip stems to re-retarget without --yaw_normalize, e.g. pico1015_161659_clip_022
[ ${#CLIPS[@]} -gt 0 ] || { echo "usage: $0 <clip stem> ..." >&2; exit 1; }
export GMR_MEM_THRESHOLD_GB=12
mkdir -p "$BAK"
for c in "${CLIPS[@]}"; do cp -n "$RAW/$c.pkl" "$BAK/"; done
[ -d "$EGO/${SET}_pre_fix" ] || cp -r "$EGO/$SET" "$EGO/${SET}_pre_fix"
cd "$GMR"
"$PY_GMR" scripts/retarget_warm_start.py --src "$NPZ" --tgt "$RAW" --clips "${CLIPS[@]}" --settle 30 > /tmp/refix_noyaw_${SET}.log 2>&1
echo "retargeted: $(grep -c '^done' /tmp/refix_noyaw_${SET}.log)"
rm -rf "$MOT"
"$PY_GMR" scripts/amass_pipeline/04_convert_to_motion_lib.py --src_folder "$RAW" --tgt_folder "$MOT" > /tmp/refix_noyaw_${SET}_s4.log 2>&1
echo "motion_lib: $(ls "$MOT" | wc -l) files"
cd "$GAM"
"$PY_BUILD" data_process/build_${SET}.py > /tmp/refix_noyaw_${SET}_s5.log 2>&1
grep -E "failed|Built|trimmed_smpl:|already_aligned:" /tmp/refix_noyaw_${SET}_s5.log | tail -n 5
echo DONE
