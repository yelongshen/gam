#!/usr/bin/env bash
# Termination ablation on ICL_hardset: which tracking-termination term kills each clip?
# Runs the base checkpoint on all 35 clips once per variant, relaxing ONE termination term (or all) at a time.
#   ./model_eval/term_ablation_icl.sh            (about 2 min per variant; results in logs_eval/*EVAL_iclabl_*)
# Read with: python sim2real/icl_term_ablation_report.py
set -uo pipefail
REPO=/home/grease/GR00T-WholeBodyControl
PY=${PY:-/home/grease/miniforge3/envs/env_isaaclab/bin/python}
CKPT=${CKPT:-/home/grease/ckpts_reuben/sonic_no_vr_low_latency_with_amass_envs_16K/model_step_100000.pt}
DIR=${DIR:-/home/grease/ego_dataset/ICL_hardset}
N=$(ls "$DIR"/robot/*.pkl | wc -l)
cd "$REPO"; mkdir -p logs_train

relax() {  # term -> hydra overrides that effectively disable it
  case "$1" in
    anchor_pos)     echo "++manager_env.terminations.anchor_pos.params.threshold=100.0 ++manager_env.terminations.anchor_pos.params.threshold_adaptive=false ++manager_env.terminations.anchor_pos.params.down_threshold=100.0 ++manager_env.terminations.anchor_pos.params.root_height_threshold=-100.0" ;;
    anchor_ori)     echo "++manager_env.terminations.anchor_ori_full.params.threshold=100.0" ;;
    ee_body_pos)    echo "++manager_env.terminations.ee_body_pos.params.threshold=100.0 ++manager_env.terminations.ee_body_pos.params.threshold_adaptive=false ++manager_env.terminations.ee_body_pos.params.down_threshold=100.0 ++manager_env.terminations.ee_body_pos.params.root_height_threshold=-100.0" ;;
    foot_pos_xyz)   echo "++manager_env.terminations.foot_pos_xyz.params.threshold=100.0" ;;
    baseline)       echo "" ;;
    all)            for t in anchor_pos anchor_ori ee_body_pos foot_pos_xyz; do relax $t; done ;;
  esac
}

for v in ${VARIANTS:-baseline anchor_pos anchor_ori ee_body_pos foot_pos_xyz all}; do
  echo "=== $v"
  # shellcheck disable=SC2046
  "$PY" gear_sonic/eval_agent_trl.py checkpoint="$CKPT" +num_envs="$N" +headless=true +run_once=true \
    +eval_callbacks=im_eval '+eval_output_dir=${eval_log_dir}' \
    ++manager_env.observations.policy.enable_corruption=False ++manager_env.observations.tokenizer.enable_corruption=False \
    +manager_env/terminations=tracking/eval \
    +manager_env.commands.motion.motion_lib_cfg.motion_file="$DIR/robot" \
    +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="$DIR/smpl" \
    ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions="$N" \
    $(relax $v) eval_name="EVAL_${PREFIX:-iclabl}_${v}" > "logs_train/${PREFIX:-iclabl}_${v}.log" 2>&1
  echo "  exit=$?"
done
echo ABLATION_DONE
