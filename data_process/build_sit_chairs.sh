#!/usr/bin/env bash
# (Re)generate the per-clip chairs of a sit dataset from its retargeted robot trajectories:
#   infer chair pose -> build envs (shared slab + per-clip pkl/json) -> make the slab 0.40 deep x 0.45 wide x 0.03 thick,
#   shifted 0.05 m behind the pelvis -> fit each seat height to the robot (enforce_sit_contact --mode chair).
#   ./data_process/build_sit_chairs.sh [dataset_dir]
set -euo pipefail
D=${1:-/home/grease/ego_dataset/picoset_20261002_sit}
REPO=/home/grease/GR00T-WholeBodyControl
PY=/home/grease/gam/.venv_sim/bin/python
PYI=/home/grease/miniforge3/envs/env_isaaclab/bin/python
cd "$REPO"
rm -rf "$D/envs"; mkdir -p "$D/envs"
$PY dev_notes/sit_subset_build/infer_chair_from_motion.py --dir "$D" --out "$D/envs/chair_params.csv" | sed -n 1,3p
$PY dev_notes/sit_subset_build/build_chair_env.py --dir "$D" --csv "$D/envs/chair_params.csv" --seat-w 0.40 --seat-thick 0.03 --seat-back-offset 0.05 | tail -n 2
$PY - "$D" <<'E'
import glob, json, sys, importlib.util
D = sys.argv[1] + "/envs"
spec = importlib.util.spec_from_file_location("bce", "/home/grease/GR00T-WholeBodyControl/dev_notes/sit_subset_build/build_chair_env.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
DEPTH, WIDTH, THICK = 0.40, 0.45, 0.03
txt = m.usda_text(DEPTH, THICK).replace(f"({DEPTH}, {DEPTH}, {THICK})", f"({DEPTH}, {WIDTH}, {THICK})")
assert f"({DEPTH}, {WIDTH}, {THICK})" in txt
open(D + "/chair_seat.usda", "w").write(txt)
for f in sorted(glob.glob(D + "/*.json")):
    j = json.load(open(f)); j["seat_size"] = [DEPTH, WIDTH, THICK]
    j["seat_size_note"] = "[depth along facing direction, width across, thickness]; origin = centre of seat TOP surface; seat centre 0.05 m behind the pelvis"
    json.dump(j, open(f, "w"), indent=2)
print("rectangular slab written")
E
$PYI dev_notes/sit_subset_build/enforce_sit_contact.py --dir "$D" --mode chair > /tmp/enforce_chair_rebuild.log 2>&1
$PYI dev_notes/sit_subset_build/enforce_sit_contact.py --dir "$D" --check 2>&1 | tail -n 2
