#!/usr/bin/env bash
# Re-evaluate 10 checkpoints from model_eval/checkpoint_comparison.md on the new
# picoset_20260924 eval set, to check whether the numbers line up with the
# existing pico_evalset column of that table.
#
# Same flags as every other eval in checkpoint_comparison.md (see its
# "Termination-config caveat" section: the termination set is always the merged
# tracking/base + tracking/eval one, which is fine because it is uniform across
# all runs being compared).
set -u
cd /home/grease/GR00T-WholeBodyControl
export PYTHONPATH=$PWD
PY=~/miniforge3/envs/env_isaaclab/bin/python
CK=/home/grease/ckpts_reuben
ROBOT=/home/grease/ego_dataset/picoset_20260924/robot
SMPL=/home/grease/ego_dataset/picoset_20260924/smpl
N=93   # one env per clip

run () {  # $1 = short tag, $2 = checkpoint path
  local tag=$1 ckpt=$2
  if [ ! -f "$ckpt" ]; then echo "[skip] $tag: missing $ckpt"; return; fi
  if ls -d logs_eval/*EVAL_picoset20260924_${tag} >/dev/null 2>&1; then
    echo "[done] $tag already evaluated"; return
  fi
  echo "=== $tag ==="
  $PY gear_sonic/eval_agent_trl.py \
    checkpoint="$ckpt" \
    +headless=true +num_envs=$N \
    +manager_env.commands.motion.motion_lib_cfg.motion_file=$ROBOT \
    +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$SMPL \
    eval_name=EVAL_picoset20260924_${tag} \
    algo.config.eval.num_eval_episodes=1 \
    +run_once=true +eval_callbacks=im_eval \
    +eval_output_dir='${eval_log_dir}' \
    > /tmp/eval_pico_${tag}.log 2>&1
  echo "  exit=$? -> $(ls -dt logs_eval/*EVAL_picoset20260924_${tag} 2>/dev/null | head -1)"
}

# Baseline
run LOW_LATENCY   /home/grease/gam/gear_sonic_deploy/policy/low_latency/last.pt
# no-VR family
run novr_050k     $CK/sonic_no_vr_envs_16K/model_step_050000.pt
# no-VR low-latency family (the 020k -> 074k trajectory)
run novrll_020k   $CK/sonic_no_vr_low_latency_envs_16K/model_step_020000.pt
run novrll_038k   $CK/sonic_no_vr_low_latency_envs_16K/model_step_038000.pt
run novrll_052k   $CK/sonic_no_vr_low_latency_envs_16K/model_step_052000.pt
run novrll_062k   $CK/sonic_no_vr_low_latency_envs_16K/model_step_062000.pt
run novrll_074k   $CK/sonic_no_vr_low_latency_envs_16K/model_step_074000.pt
# + AMASS family (incl. 080k, the recommended checkpoint)
run LLAM030k      $CK/sonic_no_vr_low_latency_with_amass_envs_16K/model_step_030000.pt
run LLAM050k      $CK/sonic_no_vr_low_latency_with_amass_envs_16K/model_step_050000.pt
run LLAM080k      $CK/sonic_no_vr_low_latency_with_amass_envs_16K/model_step_080000.pt

echo "ALL DONE"
