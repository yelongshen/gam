#!/usr/bin/env bash
# Try several IK-weight overrides on a 30 s dance window of dance2_subject3 and print arm/torso metrics.
#   ./data_process/variant_test.sh
set -euo pipefail
W=/tmp/lafan_var
SOMA=$HOME/soma-retargeter
GAM=$HOME/gam
NAME=dance2_subject3
CONV=${CONV:-convert_smpl_filtered_to_bvh.py}; TAG=${TAG:-v2}
SRC_BVH=/tmp/lafan_fix_src_$TAG/$NAME.bvh          # smpl_filtered(30 fps) -> SOMA BVH (geometric)
F0=${F0:-3600}; N=${N:-900}
rm -rf $W; mkdir -p $W/{bvh,smpl} /tmp/lafan_fix_src_$TAG
if [ ! -s "$SRC_BVH" ]; then
  (cd "$GAM/data_process" && "$GAM/.venv_sim/bin/python" $CONV \
    --pkl "$HOME/ego_dataset/lafan1_smpl_filtered_FPS30/$NAME.pkl" \
    --template "$SOMA/assets/motions/bvh/Neutral_walk_forward_002__A057.bvh" --out "$SRC_BVH" > /dev/null)
fi

"$GAM/.venv_sim/bin/python" - <<EOF
import joblib, numpy as np
lines = open("$SRC_BVH").read().split("\n")
mo = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
head, data = lines[:mo], [l for l in lines[mo + 3:] if l.strip()]
sel = data[$F0:$F0 + $N]
open("$W/bvh/$NAME.bvh", "w").write("\n".join(head + ["MOTION", f"Frames: {len(sel)}", "Frame Time: 0.033333"] + sel) + "\n")
d = joblib.load("$HOME/ego_dataset/lafan1_smpl_filtered_FPS30/$NAME.pkl")
c = {k: (v[$F0:$F0 + $N] if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == len(d["transl"]) else v) for k, v in d.items()}
joblib.dump(c, "$W/smpl/$NAME.pkl")
print("cropped", len(sel), "frames")
EOF

run() {  # name  RW-json
  local v=$1 rw=$2
  mkdir -p $W/in_$v $W/csv_$v
  cp $W/bvh/$NAME.bvh $W/in_$v/
  cat > $W/cfg_$v.json <<EOF
{"import_folder": "$W/in_$v", "export_folder": "$W/csv_$v", "batch_size": 100, "retargeter": "Newton",
 "retarget_source": "soma", "retarget_target": "unitree_g1", "retarget_source_facing_direction": "Mujoco"}
EOF
  (cd $SOMA && RW="$rw" .venv/bin/python "$GAM/data_process/retarget_soma_override.py" --config $W/cfg_$v.json --viewer null) > $W/log_$v.txt 2>&1 || { echo "variant $v FAILED"; tail -3 $W/log_$v.txt; }
}

run base '{}'
run arms '{"LeftForeArm":0.0,"RightForeArm":0.0,"LeftHand":0.0,"RightHand":0.0}'
run armstorso '{"Hips":0.0,"Chest":0.0,"LeftForeArm":0.0,"RightForeArm":0.0,"LeftHand":0.0,"RightHand":0.0}'
run all0 '{"Hips":0.0,"Chest":0.0,"LeftArm":0.0,"RightArm":0.0,"LeftForeArm":0.0,"RightForeArm":0.0,"LeftHand":0.0,"RightHand":0.0,"LeftShin":0.0,"RightShin":0.0}'
"$GAM/.venv_sim/bin/python" "$GAM/data_process/eval_retarget_variants.py" base arms armstorso all0
