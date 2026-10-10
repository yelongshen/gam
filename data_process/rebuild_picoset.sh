#!/usr/bin/env bash
# Rebuild a PICO dataset from its raw clips with ALL fixes:
#   1. correct frame rate (measured from timestamps -- pico_fps made clips play ~1.8x too fast)
#   2. floor fix  (--floor_mode foot_envelope) for the "A" clips (all clips for the sit set)
#   3. start trim for the "B" clips (start airborne / elevated)
#   4. retargeting fixes for ALL clips: --yaw_normalize (bowed-forward IK) and --settle 30 (first-frame spike)
# Old outputs are renamed to *_wrongspeed (nothing is deleted).
#
#   ./data_process/rebuild_picoset.sh picoset_20260928|picoset_20260929|picoset_20260930|picoset_20261002_sit
set -euo pipefail
SET=$1
GMR=/home/grease/GMR; GAM=/home/grease/gam; EGO=/home/grease/ego_dataset; PR=/home/grease/g1_robot_data/pico_raw
PY_GAM=$GAM/.venv_sim/bin/python; PY_TEL=$GAM/.venv_teleop/bin/python
PY_GMR=/home/grease/miniforge3/envs/gmr/bin/python; PY_BUILD=/home/grease/miniforge3/envs/env_isaaclab/bin/python
ALL_FLOOR=0
case $SET in
  picoset_20260928) PREFIX=pico0928_; CLIPS=$PR/20260928_162215_clips; A="clip_021 clip_003 clip_005_006 clip_013"; B="clip_018";;
  picoset_20260929) PREFIX=pico0929_; CLIPS=$PR/20260929_174102_clips; A="clip_054 clip_053 clip_077"; B="";;
  picoset_20260930) PREFIX=pico0930_; CLIPS=$PR/20260930_combined_clips; A="145524_clip_012"; B="145524_clip_034 145524_clip_021";;
  picoset_20261002_sit) PREFIX=pico1002sit_; CLIPS=$PR/20261002_151150_sit_clips; A=""; B=""; ALL_FLOOR=1;;
  *) echo "unknown set"; exit 1;;
esac
S50=$GAM/logs_pkl/$SET; S30=$GAM/logs_pkl/${SET}_FPS30; NPZ=$GMR/${SET}_smplx_npz_fps30
RAW=$GMR/${SET}_retargeted_g1_fps30; MOT=${RAW}_motion_lib; OUT=$EGO/$SET
TRIM=$PR/${SET}_trimmed_clips
# clip stems = the clips currently in the dataset (e.g. 0928 has the merged clip_005_006 instead of 005 and 006)
mapfile -t STEMS < <(ls "$OUT/smpl" | sed "s/^${PREFIX}//; s/\.pkl$//" | sort)
echo "[$SET] ${#STEMS[@]} clips"
for d in "$S50" "$S30" "$NPZ" "$RAW" "$MOT" "$OUT"; do
  [ -e "$d" ] && [ ! -e "${d}_wrongspeed" ] && mv "$d" "${d}_wrongspeed"
  mkdir -p "$d"
done
mkdir -p "$TRIM"
JOBS=$(mktemp)
for c in "${STEMS[@]}"; do
  src="$CLIPS/$c.npz"; extra=""
  if [[ " $B " == *" $c "* ]]; then "$PY_TEL" "$GAM/data_process/trim_raw_clip.py" "$src" "$TRIM/$c.npz"; src="$TRIM/$c.npz"; fi
  if [[ " $A " == *" $c "* ]] || [ "$ALL_FLOOR" = 1 ]; then extra="--floor_mode foot_envelope"; fi
  for fps in 50 30; do
    o=$S50; [ $fps = 30 ] && o=$S30
    echo "$PY_GAM $GAM/data_process/pico_to_smpl_filtered.py --dir $src --target_fps $fps --transl_mode body_pos_w $extra --out $o/${PREFIX}$c.pkl > /dev/null 2>&1 || echo FAILED ${PREFIX}$c $fps" >> "$JOBS"
  done
done
echo "[$SET] stage 1: $(wc -l < "$JOBS") conversions"
xargs -P 6 -I{} bash -c "{}" < "$JOBS"
echo "[$SET] stage 1 done: $(ls "$S50" | wc -l) @50, $(ls "$S30" | wc -l) @30"
"$PY_GMR" "$GMR/scripts/amass_pipeline/debug/convert_teleop_logs_to_smplx_npz.py" --src_folder "$S30" --tgt_folder "$NPZ" --root_correction_deg 90 > "/tmp/rb_${SET}_s2.log" 2>&1
echo "[$SET] stage 2 done: $(ls "$NPZ" | wc -l) npz"
cd "$GMR"
printf '%s\n' "${STEMS[@]/#/$PREFIX}" | GMR_MEM_THRESHOLD_GB=12 xargs -P 4 -n 15 "$PY_GMR" scripts/retarget_warm_start.py --src "$NPZ" --tgt "$RAW" --yaw_normalize --settle 30 --clips > "/tmp/rb_${SET}_s3.log" 2>&1
echo "[$SET] stage 3 done: $(ls "$RAW" | wc -l) retargeted"
EXTRA=(); [ "$SET" = picoset_20260928 ] && EXTRA=(--filtered_smpl_folder "$S30")
"$PY_GMR" scripts/amass_pipeline/04_convert_to_motion_lib.py --src_folder "$RAW" --tgt_folder "$MOT" "${EXTRA[@]}" > "/tmp/rb_${SET}_s4.log" 2>&1
echo "[$SET] stage 4 done: $(ls "$MOT" | wc -l)"
cd "$GAM"; "$PY_BUILD" "data_process/build_${SET}.py" > "/tmp/rb_${SET}_s5.log" 2>&1
grep -E "failed|aligned|trimmed|Built" "/tmp/rb_${SET}_s5.log" | tail -n 4
[ "$SET" = picoset_20261002_sit ] && ./data_process/build_sit_chairs.sh "$OUT" 2>&1 | tail -n 4
echo "[$SET] DONE"
