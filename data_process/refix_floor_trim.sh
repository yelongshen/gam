#!/usr/bin/env bash
# Re-convert + re-retarget selected clips of a PICO dataset, then rebuild the paired dataset.
#   A clips: floor fix  -> pico_to_smpl_filtered.py --floor_mode foot_envelope   (standing frames sit too high above the floor)
#   B clips: trim       -> start cut at the first foot-on-floor frame (clip starts airborne / elevated), default floor
# All of them are then retargeted with --yaw_normalize --settle 30 and the set is rebuilt (stage 4 + 5).
#
#   ./data_process/refix_floor_trim.sh <set> <raw_clips_dir> "<A clips>" "<B clips>"
#   e.g. ./data_process/refix_floor_trim.sh picoset_20260930 ~/g1_robot_data/pico_raw/20260930_combined_clips "145524_clip_012" "145524_clip_034 145524_clip_021"
# Clip names are the raw file stems; dataset names are PREFIX + stem.  Originals are kept in *_pre_floorfix next to each folder.
set -euo pipefail
SET=$1; CLIPS=$2; A=$3; B=$4
GMR=/home/grease/GMR; GAM=/home/grease/gam; EGO=/home/grease/ego_dataset
PY_GAM=$GAM/.venv_sim/bin/python; PY_TEL=$GAM/.venv_teleop/bin/python
PY_GMR=/home/grease/miniforge3/envs/gmr/bin/python; PY_BUILD=/home/grease/miniforge3/envs/env_isaaclab/bin/python
case $SET in picoset_20260928) PREFIX=pico0928_;; picoset_20260929) PREFIX=pico0929_;; picoset_20260930) PREFIX=pico0930_;; picoset_20261003) PREFIX=pico1003_;; *) echo bad set; exit 1;; esac
S50=$GAM/logs_pkl/$SET; S30=$GAM/logs_pkl/${SET}_FPS30; NPZ=$GMR/${SET}_smplx_npz_fps30
RAW=$GMR/${SET}_retargeted_g1_fps30; MOT=${RAW}_motion_lib
TRIM=$(dirname "$CLIPS")/${SET}_trimmed_clips; mkdir -p "$TRIM"
for d in "$S50" "$S30" "$NPZ" "$RAW"; do mkdir -p "${d}_pre_floorfix"; done
[ -d "$EGO/${SET}_pre_floorfix" ] || cp -r "$EGO/$SET" "$EGO/${SET}_pre_floorfix"
ALL=()
for c in $A $B; do
  n=${PREFIX}$c; ALL+=("$n")
  cp -n "$S50/$n.pkl" "${S50}_pre_floorfix/"; cp -n "$S30/$n.pkl" "${S30}_pre_floorfix/"
  cp -n "$NPZ/$n.npz" "${NPZ}_pre_floorfix/"; cp -n "$RAW/$n.pkl" "${RAW}_pre_floorfix/"
done
conv() {   # name src_npz extra...
  local n=$1 src=$2; shift 2
  for fps in 50 30; do
    out=$S50; [ $fps = 30 ] && out=$S30
    "$PY_GAM" "$GAM/data_process/pico_to_smpl_filtered.py" --dir "$src" --target_fps $fps --transl_mode body_pos_w "$@" --out "$out/$n.pkl" > /dev/null
  done
}
for c in $A; do conv "${PREFIX}$c" "$CLIPS/$c.npz" --floor_mode foot_envelope; echo "A $c converted"; done
for c in $B; do
  "$PY_TEL" "$GAM/data_process/trim_raw_clip.py" "$CLIPS/$c.npz" "$TRIM/$c.npz"
  conv "${PREFIX}$c" "$TRIM/$c.npz"; echo "B $c converted"
done
# stage 2 (only the affected clips): SMPL -> SMPL-X npz
TMP=$(mktemp -d); for n in "${ALL[@]}"; do ln -s "$S30/$n.pkl" "$TMP/$n.pkl"; rm -f "$NPZ/$n.npz"; done
"$PY_GMR" "$GMR/scripts/amass_pipeline/debug/convert_teleop_logs_to_smplx_npz.py" --src_folder "$TMP" --tgt_folder "$NPZ" --root_correction_deg 90 > "/tmp/ft_${SET}_s2.log" 2>&1
for n in "${ALL[@]}"; do [ -s "$NPZ/$n.npz" ] || { echo "MISSING npz $n"; exit 1; }; done
# stage 3: retarget with the fixes
cd "$GMR"
"$PY_GMR" scripts/retarget_warm_start.py --src "$NPZ" --tgt "$RAW" --clips "${ALL[@]}" --yaw_normalize --settle 30 > "/tmp/ft_${SET}_s3.log" 2>&1
echo "retargeted: $(grep -c '^done' /tmp/ft_${SET}_s3.log)/${#ALL[@]}"
# stage 4 + 5
rm -rf "$MOT"; EXTRA=(); [ "$SET" = picoset_20260928 ] && EXTRA=(--filtered_smpl_folder "$S30")
"$PY_GMR" scripts/amass_pipeline/04_convert_to_motion_lib.py --src_folder "$RAW" --tgt_folder "$MOT" "${EXTRA[@]}" > "/tmp/ft_${SET}_s4.log" 2>&1
cd "$GAM"; "$PY_BUILD" "data_process/build_${SET}.py" > "/tmp/ft_${SET}_s5.log" 2>&1
grep -E "failed|aligned|trimmed|Built" "/tmp/ft_${SET}_s5.log" | tail -n 4
echo "[$SET] DONE"
