#!/usr/bin/env bash
# Try retarget variants on the clips of picoset_20261015 that still show a start-up fault after --yaw_normalize --settle 30,
# writing to /tmp/rt_variants/<variant>/ so nothing in the dataset is touched.
set -uo pipefail
GMR=/home/grease/GMR
PY=/home/grease/miniforge3/envs/gmr/bin/python
NPZ=$GMR/picoset_20261015_smplx_npz_fps30
CLIPS=(pico1015_161659_clip_022 pico1015_161659_clip_017 pico1015_160754_clip_014 pico1015_161659_clip_014)
export GMR_MEM_THRESHOLD_GB=12
cd $GMR
run() {  # name, extra args...
  name=$1; shift
  mkdir -p /tmp/rt_variants/$name
  $PY scripts/retarget_warm_start.py --src $NPZ --tgt /tmp/rt_variants/$name --clips "${CLIPS[@]}" --yaw_normalize "$@" > /tmp/rt_variants/$name.log 2>&1
  echo "$name: $(grep -c '^done' /tmp/rt_variants/$name.log) clips"
}
run base_settle30 --settle 30
run settle100 --settle 100
run passes2_settle30 --passes 2 --settle 30
run passes3_settle100 --passes 3 --settle 100
echo VARIANTS_DONE
