#!/usr/bin/env bash
# Build `ego_dataset/picoset_20261003` from the 2026-09-29 PICO raw session.
#
# Session: /home/grease/g1_robot_data/pico_raw/20261003_combined
#   84,742 frames @ ~86 Hz (29.0 min) -> 163 clips (28.5 min kept) via
#   chunk_pico_session.py (motion-activity segmentation; `toggle_data_collection`
#   is recorded by the rig but never asserted in these sessions, so there are no
#   manual markers to cut on).
#
# This is the same 5-stage chain that produced `picoset_20260928`, just driven
# end-to-end. The two branches are retargeted SEPARATELY so each lands on its
# required native rate:
#
#   stage 1a  pico_to_smpl_filtered.py --target_fps 50  -> gam/logs_pkl/picoset_20261003
#   stage 1b  pico_to_smpl_filtered.py --target_fps 30  -> gam/logs_pkl/picoset_20261003_FPS30
#   stage 2   convert_teleop_logs_to_smplx_npz.py (+90 deg root fix, neutral betas)
#   stage 3   03_retarget_to_robot.py            -> GMR/..._retargeted_g1_fps30
#   stage 4   04_convert_to_motion_lib.py        -> GMR/..._retargeted_g1_fps30_motion_lib
#   stage 5   build_picoset_20261003.py          -> ego_dataset/picoset_20261003
#
# `--transl_mode body_pos_w` is REQUIRED: this session (like 0928) was recorded
# with `--record_root_pos`, so `body_pos_w` carries the real world pelvis
# position. Without it `transl` is zeroed and the clips become in-place only,
# which is what made `picoset_20260924`'s mpjpe_g meaningless (see
# model_eval/checkpoint_comparison.md).
#
# Source fps is read per-clip from `pico_fps` inside the npz and resampled, so
# the ~86 Hz capture rate needs no special handling.
#
# Usage:
#   ./data_process/build_picoset_20261003.sh              # all stages
#   STAGES=1 ./data_process/build_picoset_20261003.sh     # just the smpl conversions
set -euo pipefail

SESSION=${SESSION:-/home/grease/g1_robot_data/pico_raw/20261003_combined}
CLIPS=${CLIPS:-${SESSION}_clips}
TAG=${TAG:-picoset_20261003}
PREFIX=${PREFIX:-pico1003_}
STAGES=${STAGES:-all}

# `pico_to_smpl_filtered.py` needs `.venv_sim` (it has joblib; the plain `.venv`
# does not -- see that script's own docstring). The GMR retargeting stages need
# the dedicated `gmr` conda env (natsort + mujoco); the isaaclab env lacks them.
PY_GAM=${PY_GAM:-/home/grease/gam/.venv_sim/bin/python}
PY_GMR=${PY_GMR:-/home/grease/miniforge3/envs/gmr/bin/python}
PY_BUILD=${PY_BUILD:-/home/grease/miniforge3/envs/env_isaaclab/bin/python}
GAM=/home/grease/gam
GMR=/home/grease/GMR

SMPL50=$GAM/logs_pkl/$TAG
SMPL30=$GAM/logs_pkl/${TAG}_FPS30
NPZ30=$GMR/${TAG}_smplx_npz_fps30
RETGT=$GMR/${TAG}_retargeted_g1_fps30
MOTLIB=$GMR/${TAG}_retargeted_g1_fps30_motion_lib

# GMR's `smplx_to_robot_dataset.check_memory()` pauses 2 min whenever AVAILABLE
# RAM is below a threshold, and aborts after 10 pauses. Its default of 30 GB
# never clears on this 62 GB box while a training job is resident (~24 GB free),
# so no clip ever starts. 12 GB is comfortably above actual per-clip usage.
export GMR_MEM_THRESHOLD_GB=${GMR_MEM_THRESHOLD_GB:-12}

[ -f "$CLIPS/clips.csv" ] || { echo "ERROR: no $CLIPS/clips.csv -- run chunk_pico_session.py first" >&2; exit 1; }

N=$(($(wc -l < "$CLIPS/clips.csv") - 1))
echo "session : $SESSION"
echo "clips   : $N  (manifest $CLIPS/clips.csv)"
echo "out     : /home/grease/ego_dataset/$TAG"
echo

# ---- stage 1: raw frames -> smpl_filtered pkl, at BOTH 50 and 30 fps ----------
if [ "$STAGES" = all ] || [ "$STAGES" = 1 ]; then
  mkdir -p "$SMPL50" "$SMPL30"
  echo "=== stage 1: pico -> smpl_filtered (50 fps + 30 fps) ==="
  # IMPORTANT: feed `--dir` the CONSOLIDATED clip .npz from chunk_pico_session.py,
  # NOT the raw session dir + --start/--end. Both are accepted, but the legacy
  # session-dir loader (`load_pico_session`) never reads `body_pos_w`, so it
  # fails with "transl_mode='body_pos_w' but the clip has no body_pos_w".
  # Only `load_pico_clip_npz` carries the real root translation.
  # clips.csv columns: clip,start_frame,end_frame,frames,start_s,dur_s,path_m,disp_m,mean_act
  tail -n +2 "$CLIPS/clips.csv" | while IFS=, read -r clip rest; do
    name="${PREFIX}$(basename "$clip" .npz)"
    for spec in "50:$SMPL50" "30:$SMPL30"; do
      fps=${spec%%:*}; out=${spec#*:}
      [ -s "$out/$name.pkl" ] && continue
      "$PY_GAM" "$GAM/data_process/pico_to_smpl_filtered.py" \
        --dir "$CLIPS/$clip" \
        --target_fps "$fps" --transl_mode body_pos_w \
        --out "$out/$name.pkl" > /dev/null 2>&1 \
        || echo "  WARN: failed $name @ ${fps}fps"
    done
  done
  echo "  50fps: $(ls "$SMPL50"/*.pkl 2>/dev/null | wc -l) pkl"
  echo "  30fps: $(ls "$SMPL30"/*.pkl 2>/dev/null | wc -l) pkl"
fi

# ---- stage 2: smpl pkl (30 fps) -> SMPL-X npz, +90deg root fix ---------------
if [ "$STAGES" = all ] || [ "$STAGES" = 2 ]; then
  echo "=== stage 2: -> SMPL-X npz (+90deg root correction, neutral betas) ==="
  "$PY_GMR" "$GMR/scripts/amass_pipeline/debug/convert_teleop_logs_to_smplx_npz.py" \
    --src_folder "$SMPL30" --tgt_folder "$NPZ30" --root_correction_deg 90
fi

# ---- stage 3: retarget to G1 --------------------------------------------------
if [ "$STAGES" = all ] || [ "$STAGES" = 3 ]; then
  echo "=== stage 3: retarget to unitree_g1 ==="
  # Known-issue fixes (PICO_TELEOP_RETARGETING_PIPELINE.md §B): --yaw_normalize + --settle 30
  mkdir -p "$RETGT"; cd "$GMR"
  ls "$NPZ30" | sed 's/\.npz$//' | xargs -P 4 -n 15 "$PY_GMR" scripts/retarget_warm_start.py \
    --src "$NPZ30" --tgt "$RETGT" --yaw_normalize --settle 30 --clips
fi

# ---- stage 4: -> motion_lib format -------------------------------------------
if [ "$STAGES" = all ] || [ "$STAGES" = 4 ]; then
  echo "=== stage 4: -> motion_lib format ==="
  "$PY_GMR" "$GMR/scripts/amass_pipeline/04_convert_to_motion_lib.py" \
    --src_folder "$RETGT" --tgt_folder "$MOTLIB"
fi

# ---- stage 5: pair + frame-align into ego_dataset -----------------------------
if [ "$STAGES" = all ] || [ "$STAGES" = 5 ]; then
  echo "=== stage 5: pair smpl(50) + robot(30), align frame counts ==="
  "$PY_BUILD" "$GAM/data_process/build_picoset_20261003.py"
fi

echo
echo "done."
