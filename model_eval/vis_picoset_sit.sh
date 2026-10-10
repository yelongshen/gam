#!/usr/bin/env bash
# Render one video per clip of the sit set WITH its chair, with every failure termination relaxed
# (FULL=1: the episode plays to the last frame of the clip, so you see the whole failure instead of the
# video stopping at the moment of divergence).  Chair flags as in eval_picoset_sit.sh.
#
#   DS=/home/grease/ego_dataset/picoset_20261002_sit_short CKPT=<...>.pt TAG=sitmixv2 \
#     ./model_eval/vis_picoset_sit.sh clip_001 clip_024 ...        # clip ids without the pico1002sit_ prefix
set -uo pipefail

REPO=/home/grease/GR00T-WholeBodyControl
PY=${PY:-/home/grease/miniforge3/envs/env_isaaclab/bin/python}
CKPT=${CKPT:-$REPO/logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_no_teleop_sit_mixed_v2-20260914_153300/model_step_004000.pt}
TAG=${TAG:-sitmixv2}
DS=${DS:-/home/grease/ego_dataset/picoset_20261002_sit_short}
CAM=${CAM:-'[2.5,2.5,1.5]'}
OUT=${OUT:-$REPO/model_eval/vis_picoset_sit_${TAG}_FULL}
cd "$REPO"; mkdir -p "$OUT"

if [ "$#" -gt 0 ]; then CLIPS=("$@"); else mapfile -t CLIPS < <(cd "$DS/robot" && ls *.pkl | sed 's/^pico1002sit_//; s/\.pkl$//'); fi
echo "=== ${#CLIPS[@]} clip(s) -> $OUT ==="
for c in "${CLIPS[@]}"; do
  key="pico1002sit_$c"
  [ -s "$OUT/$c/000000.mp4" ] && { echo "--- skip $c"; continue; }
  echo "--- $c ---"
  "$PY" gear_sonic/eval_agent_trl.py \
    checkpoint="$CKPT" +headless=true +num_envs=1 +run_once=true \
    +eval_callbacks=im_eval +eval_output_dir='${eval_log_dir}' \
    +manager_env.commands.motion.motion_lib_cfg.motion_file="$DS/robot" \
    +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="$DS/smpl" \
    +manager_env.commands.motion.motion_lib_cfg.object_motion_file="$DS/envs" \
    ++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys="[$key]" \
    ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=1 \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.config.add_object=true \
    ++manager_env.config.object_usd_path="$DS/envs/chair_seat.usda" \
    ++manager_env.config.object_is_dynamic=false \
    ++manager_env.config.object_collision_enabled=true \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    +manager_env/terminations=tracking/eval \
    manager_env/recorders=render ++manager_env.config.render_results=True \
    ++manager_env.config.eval_camera_offset="$CAM" ++manager_env.config.save_rendering_dir="$OUT/$c" \
    ++manager_env.terminations.anchor_pos.params.threshold=100.0 \
    ++manager_env.terminations.anchor_pos.params.threshold_adaptive=false \
    ++manager_env.terminations.anchor_pos.params.down_threshold=100.0 \
    ++manager_env.terminations.anchor_pos.params.root_height_threshold=-100.0 \
    ++manager_env.terminations.anchor_ori_full.params.threshold=100.0 \
    ++manager_env.terminations.ee_body_pos.params.threshold=100.0 \
    ++manager_env.terminations.ee_body_pos.params.threshold_adaptive=false \
    ++manager_env.terminations.ee_body_pos.params.down_threshold=100.0 \
    ++manager_env.terminations.ee_body_pos.params.root_height_threshold=-100.0 \
    ++manager_env.terminations.foot_pos_xyz.params.threshold=100.0 \
    eval_name="VIDSIT1002_${TAG}_${c}" > "/tmp/vis_sit1002_${TAG}_${c}.log" 2>&1
  echo "  exit=$?"
done
find "$OUT" -name '*.mp4' -printf '%10s  %p\n' | sort -k2
