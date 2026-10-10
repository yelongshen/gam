#!/usr/bin/env bash
# Regenerate the G1 robot reference for LAFAN1 clips from `smpl_joints` (the only populated
# SMPL channel for LAFAN; non-root `pose_aa` is all zero, which made the old retargeted robot
# data a static T-pose with a moving root).
#
# Pipeline (documented in soma-retargeter/LAFAN1_PIPELINE_DEBUG_NOTES.md §1b/§2/§3/§8):
#   smpl_filtered FPS30 pkl -> SOMA BVH (geometric) -> SOMA retargeter (Newton) -> CSV
#   -> motion_lib pkl -> per-frame foot/floor clamp -> {name: entry} pkl in <OUT>
#
# Usage:  SET=lafan1_evalset ./data_process/regenerate_lafan_robot.sh
#         (SET=lafan1_trainset for the train split)   Output: ~/ego_dataset/${SET}/robot_fixed/
set -euo pipefail
SET=${SET:-lafan1_evalset}
DS=$HOME/ego_dataset
WORK=${WORK:-/tmp/lafan_fix_${SET}}
GAM=$HOME/gam
G=$HOME/GR00T-WholeBodyControl
SOMA=$HOME/soma-retargeter
ISAAC_PY=$HOME/miniforge3/envs/env_isaaclab/bin/python
MJCF=$G/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml
OUT=$DS/$SET/robot_fixed
rm -rf "$WORK"; mkdir -p "$WORK"/{pkl,bvh,csv,ml,clamped} "$OUT"

echo "[1/5] staging smpl_filtered FPS30 pkls for $SET"
for f in "$DS/$SET/smpl/"*.pkl; do
  n=$(basename "$f"); ln -s "$DS/lafan1_smpl_filtered_FPS30/$n" "$WORK/pkl/$n"
done
echo "[2/5] smpl_filtered -> SOMA BVH"
cd "$GAM/data_process"
"$GAM/.venv_sim/bin/python" convert_smpl_filtered_to_bvh.py --input_dir "$WORK/pkl" --output_dir "$WORK/bvh" \
  --template "$SOMA/assets/motions/bvh/Neutral_walk_forward_002__A057.bvh" --num_workers 8
echo "[3/5] SOMA retarget -> G1 CSV"
cat > "$WORK/cfg.json" <<EOF
{"import_folder": "$WORK/bvh", "export_folder": "$WORK/csv", "batch_size": 100,
 "retargeter": "Newton", "retarget_source": "soma", "retarget_target": "unitree_g1",
 "retarget_source_facing_direction": "Mujoco"}
EOF
cd "$SOMA"; .venv/bin/python app/bvh_to_csv_converter.py --config "$WORK/cfg.json" --viewer null
echo "[4/5] CSV -> motion_lib"
cd "$G"; "$ISAAC_PY" gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
  --input "$WORK/csv" --output "$WORK/ml" --individual --fps 30 --fps_source 30
echo "[5/5] per-frame floor clamp"
for f in "$WORK"/ml/csv/*.pkl; do
  "$ISAAC_PY" gear_sonic/data_process/foot_floor_clamp.py --motion_lib "$f" --mjcf "$MJCF" \
    --epsilon_m 0.003 --out "$OUT/$(basename "$f")" > "$WORK/clamp_$(basename "$f").log" 2>&1 || echo "clamp failed: $f"
done
echo "done -> $OUT ($(ls "$OUT" | wc -l) files)"
