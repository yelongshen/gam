#!/usr/bin/env bash
# Evaluate a sit policy on the 2026-10-02 PICO sit clips (picoset_20261002_sit), with and without the
# per-clip chair.  Same flags as gear_sonic/scripts/eval_sit_robustness.sh (the sit_evalset template).
#
#   ./model_eval/eval_picoset_sit.sh                        # mixsitv2 @ 4k, chair then no-chair
#   VARIANTS=chair ./model_eval/eval_picoset_sit.sh
#   CKPT=<...>.pt TAG=other ./model_eval/eval_picoset_sit.sh
#
# Results: <GR00T repo>/logs_eval/<timestamp>-EVAL_picoset1002sit_<variant>_<TAG>/metrics_eval.json
set -uo pipefail

REPO=/home/grease/GR00T-WholeBodyControl
PY=${PY:-/home/grease/miniforge3/envs/env_isaaclab/bin/python}
CKPT=${CKPT:-$REPO/logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_no_teleop_sit_mixed_v2-20260914_153300/model_step_004000.pt}
TAG=${TAG:-sitmixv2_004k}
DS=${DS:-/home/grease/ego_dataset/picoset_20261002_sit}
VARIANTS=${VARIANTS:-"chair nochair"}
N=$(ls "$DS"/robot/*.pkl | wc -l)

cd "$REPO"
[ -f "$CKPT" ] || { echo "missing checkpoint $CKPT"; exit 1; }
echo "ckpt   : $CKPT"
echo "dataset: $DS ($N clips)"

for v in $VARIANTS; do
  OBJ=()
  if [ "$v" = chair ]; then
    OBJ=(
      +manager_env.commands.motion.motion_lib_cfg.object_motion_file="$DS/envs"
      ++manager_env.config.add_object=true
      ++manager_env.config.object_usd_path="$DS/envs/chair_seat.usda"
      ++manager_env.config.object_is_dynamic=false
      ++manager_env.config.object_collision_enabled=true
    )
  fi
  echo "--- $v ---"
  "$PY" gear_sonic/eval_agent_trl.py \
    checkpoint="$CKPT" \
    +headless=true +num_envs="$N" +run_once=true \
    +eval_callbacks=im_eval +eval_output_dir='${eval_log_dir}' \
    +manager_env.commands.motion.motion_lib_cfg.motion_file="$DS/robot" \
    +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="$DS/smpl" \
    ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions="$N" \
    "${OBJ[@]}" \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    +manager_env/terminations=tracking/eval \
    eval_name="EVAL_picoset1002sit_${v}_${TAG}" \
    > "/tmp/eval_picoset1002sit_${v}_${TAG}.log" 2>&1
  echo "  exit=$?  log=/tmp/eval_picoset1002sit_${v}_${TAG}.log"
done
echo EVAL_DONE
