#!/usr/bin/env bash
# Re-retarget the clips of a PICO dataset that show a retargeting start-up fault, and rebuild the paired dataset.
#   ./data_process/refix_retarget.sh picoset_20260929 /tmp/fix_picoset_20260929.txt
#
# Faults (see model_eval/early_contingency.py):
#   A  bowed-forward IK solution (robot torso leans >20 deg more than the human)   -> --yaw_normalize
#   B  first-frame reference joint speed > 10 rad/s (frame 0 under-converged)      -> --settle 30
# Originals are kept: raw retargets in <raw>_pre_fix/, the paired dataset in <ego_dataset>/<set>_pre_fix/.
set -euo pipefail
SET=$1; LIST=$2
GMR=/home/grease/GMR; GAM=/home/grease/gam; EGO=/home/grease/ego_dataset
PY_GMR=/home/grease/miniforge3/envs/gmr/bin/python
PY_BUILD=/home/grease/miniforge3/envs/env_isaaclab/bin/python
RAW=$GMR/${SET}_retargeted_g1_fps30; NPZ=$GMR/${SET}_smplx_npz_fps30; MOT=${RAW}_motion_lib; BAK=${RAW}_pre_fix
mapfile -t CLIPS < "$LIST"
echo "[$SET] ${#CLIPS[@]} clips to re-retarget"
mkdir -p "$BAK"
for c in "${CLIPS[@]}"; do cp -n "$RAW/$c.pkl" "$BAK/"; done
[ -d "$EGO/${SET}_pre_fix" ] || cp -r "$EGO/$SET" "$EGO/${SET}_pre_fix"
cd "$GMR"
"$PY_GMR" scripts/retarget_warm_start.py --src "$NPZ" --tgt "$RAW" --clips "${CLIPS[@]}" --yaw_normalize --settle 30 > "/tmp/refix_${SET}_retarget.log" 2>&1
echo "[$SET] retargeted: $(grep -c '^done' /tmp/refix_${SET}_retarget.log)"
rm -rf "$MOT"
EXTRA=()
[ "$SET" = picoset_20260928 ] && EXTRA=(--filtered_smpl_folder "$GAM/logs_pkl/${SET}_FPS30")
"$PY_GMR" scripts/amass_pipeline/04_convert_to_motion_lib.py --src_folder "$RAW" --tgt_folder "$MOT" "${EXTRA[@]}" > "/tmp/refix_${SET}_s4.log" 2>&1
echo "[$SET] motion_lib: $(ls "$MOT" | wc -l) files"
cd "$GAM"
"$PY_BUILD" "data_process/build_${SET}.py" > "/tmp/refix_${SET}_s5.log" 2>&1
grep -E "failed|aligned|trimmed|Built" "/tmp/refix_${SET}_s5.log" | tail -n 4
echo "[$SET] DONE"
