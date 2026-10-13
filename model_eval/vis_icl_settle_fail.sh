#!/usr/bin/env bash
# Render one video per clip of ICL_hardset_settle2 (first frame held 2 s) with the policy, to SEE why the
# hold-failure clips fall at the settle pose (reference velocity is already 0 there).
#
# FULL=1 (default): every failure termination is relaxed, so the episode plays through the whole 2 s hold
#   and the motion instead of stopping at the first termination (~0.2 s, too short to see anything).
# FULL=0: default eval terminations; the video stops where the eval would have failed the clip.
#
#   CKPT=<model.pt> TAG=base100k ./model_eval/vis_icl_settle_fail.sh [clip ...]
#   DS=/home/grease/ego_dataset/ICL_hardset_recut_settle2 ./model_eval/vis_icl_settle_fail.sh
# Output: $OUT/<clip>/000000.mp4   (logs: /tmp/vis_iclsettle_<TAG>_<clip>.log)
set -uo pipefail

REPO=/home/grease/GR00T-WholeBodyControl
PY=${PY:-/home/grease/miniforge3/envs/env_isaaclab/bin/python}
CKPT=${CKPT:-/home/grease/ckpts_reuben/sonic_no_vr_low_latency_with_amass_envs_16K/model_step_100000.pt}
TAG=${TAG:-base100k}
DS=${DS:-/home/grease/ego_dataset/ICL_hardset_settle2}
CAM=${CAM:-'[2.5,2.5,1.5]'}
FULL=${FULL:-1}
OUT=${OUT:-$REPO/model_eval/vis_iclsettle_${TAG}$([ "$FULL" = 1 ] && echo _FULL)}
DEFAULT_CLIPS=(run1_subject5__w035s fight1_subject3__w148s jumps1_subject1__w050s
               multipleActions1_subject1__w124s fight1_subject2__w219s dance2_subject4__w196s)
cd "$REPO"; mkdir -p "$OUT"

if [ "$#" -gt 0 ]; then CLIPS=("$@"); else CLIPS=("${DEFAULT_CLIPS[@]}"); fi
RELAX=()
if [ "$FULL" = 1 ]; then
  RELAX=(++manager_env.terminations.anchor_pos.params.threshold=100.0
         ++manager_env.terminations.anchor_pos.params.threshold_adaptive=false
         ++manager_env.terminations.anchor_pos.params.down_threshold=100.0
         ++manager_env.terminations.anchor_pos.params.root_height_threshold=-100.0
         ++manager_env.terminations.anchor_ori_full.params.threshold=100.0
         ++manager_env.terminations.ee_body_pos.params.threshold=100.0
         ++manager_env.terminations.ee_body_pos.params.threshold_adaptive=false
         ++manager_env.terminations.ee_body_pos.params.down_threshold=100.0
         ++manager_env.terminations.ee_body_pos.params.root_height_threshold=-100.0
         ++manager_env.terminations.foot_pos_xyz.params.threshold=100.0)
fi

echo "=== ${#CLIPS[@]} clip(s), ckpt=$CKPT, FULL=$FULL -> $OUT ==="
for c in "${CLIPS[@]}"; do
  [ -s "$OUT/$c/000000.mp4" ] && { echo "--- skip $c"; continue; }
  echo "--- $c ---"
  "$PY" gear_sonic/eval_agent_trl.py \
    checkpoint="$CKPT" +headless=true +num_envs=1 +run_once=true \
    +eval_callbacks=im_eval '+eval_output_dir=${eval_log_dir}' \
    +manager_env.commands.motion.motion_lib_cfg.motion_file="$DS/robot" \
    +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="$DS/smpl" \
    ++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys="[$c]" \
    ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=1 \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    +manager_env/terminations=tracking/eval \
    manager_env/recorders=render ++manager_env.config.render_results=True \
    ++manager_env.config.eval_camera_offset="$CAM" ++manager_env.config.save_rendering_dir="$OUT/$c" \
    "${RELAX[@]}" \
    eval_name="VISICLSETTLE_${TAG}_${c}" > "/tmp/vis_iclsettle_${TAG}_${c}.log" 2>&1
  echo "  exit=$?"
done
find "$OUT" -name '*.mp4' -printf '%10s  %p\n' | sort -k2
