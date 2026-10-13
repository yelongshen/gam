#!/usr/bin/env bash
# Run the base + 3 mixv3b checkpoints (default terminations) on an ICL settle set and write the report.
#   ./model_eval/run_icl_settle_evals.sh                      # ICL_hardset_settle, eval prefix iclsettlev2
#   DIR=~/ego_dataset/ICL_hardset_settle PREFIX=iclsettlev2 ./model_eval/run_icl_settle_evals.sh
# Then: SET=$DIR PREFIX=$PREFIX python sim2real/icl_settle_report.py
set -uo pipefail
export DIR=${DIR:-/home/grease/ego_dataset/ICL_hardset_settle} VARIANTS=baseline
PREFIX=${PREFIX:-iclsettlev2}
REPO=/home/grease/GR00T-WholeBodyControl
V3B=$REPO/logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_no_teleop_mixpico_v3b-20261005_164106
cd /home/grease/gam
PREFIX=${PREFIX}_base100k ./model_eval/term_ablation_icl.sh
for s in 010000 014000 020000; do
  CKPT=$V3B/model_step_$s.pt PREFIX=${PREFIX}_mixv3b_$s ./model_eval/term_ablation_icl.sh
done
echo SETTLE_DONE
