#!/usr/bin/env bash
# Motion-only evaluation (no chair) of PICO datasets with termination-reason logging, to find EARLY failures.
#   ./model_eval/eval_pico_early.sh picoset_20260928 picoset_20260929 picoset_20260930
# Results: GR00T repo logs_eval/*EVAL_early_<set>_<TAG>/metrics_eval.json and /tmp/terms_early_<set>_<TAG>.jsonl
set -uo pipefail
REPO=/home/grease/GR00T-WholeBodyControl
PY=${PY:-/home/grease/miniforge3/envs/env_isaaclab/bin/python}
CKPT=${CKPT:-/home/grease/ckpts_reuben/sonic_no_vr_envs_16K/model_step_050000.pt}
TAG=${TAG:-novr050k}
cd "$REPO"
for set in "$@"; do
  DS=/home/grease/ego_dataset/$set
  N=$(ls "$DS"/robot/*.pkl | wc -l)
  rm -f "/tmp/terms_early_${set}_${TAG}.jsonl"*
  echo "--- $set ($N clips) ---"
  GEAR_SONIC_LOG_TERMS="/tmp/terms_early_${set}_${TAG}.jsonl" "$PY" gear_sonic/eval_agent_trl.py \
    checkpoint="$CKPT" +headless=true +num_envs="$N" +run_once=true \
    +eval_callbacks=im_eval +eval_output_dir='${eval_log_dir}' \
    +manager_env.commands.motion.motion_lib_cfg.motion_file="$DS/robot" \
    +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="$DS/smpl" \
    ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions="$N" \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    +manager_env/terminations=tracking/eval \
    eval_name="EVAL_early_${set}_${TAG}" > "/tmp/eval_early_${set}_${TAG}.log" 2>&1
  echo "  exit=$?"
done
echo EARLY_DONE
