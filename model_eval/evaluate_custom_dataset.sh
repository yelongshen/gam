#!/bin/bash
# High-level wrapper to evaluate any custom directory of smpl_filtered clips

INPUT_DIR="${1:-/home/grease/ego_dataset/eval_subset/smpl}"
OUTPUT_PREFIX="${2:-eval_subset}"

SPLIT_CSV="data_analysis/split/split_${OUTPUT_PREFIX}.csv"
RES_CSV="model_eval/results/sim_eval_results_${OUTPUT_PREFIX}.csv"
REP_TXT="model_eval/results/sim_eval_report_${OUTPUT_PREFIX}.txt"

echo "1. Generating split manifest -> $SPLIT_CSV"
.venv_sim/bin/python model_eval/evaluate_eval_subset.py --input_dir "$INPUT_DIR" --out_csv "$SPLIT_CSV"

echo ""
echo "2. Launching evaluation across all clips (resumable)..."
echo "   Results: $RES_CSV"
echo "   Report:  $REP_TXT"
echo ""

.venv_sim/bin/python model_eval/run_sim_eval.py \
    --split_csv "$SPLIT_CSV" \
    --out_csv "$RES_CSV" \
    --report "$REP_TXT" \
    --per_category 0 \
    --seed 0
