#!/usr/bin/env bash
# Rebuild picoset_20261015 from the stand-to-stand clips (cut_picoset_20261015_stand_clips.py).
# The first (activity-chunked) build is kept under the same names with a _v1 suffix.
set -euo pipefail
GAM=/home/grease/gam; GMR=/home/grease/GMR; EGO=/home/grease/ego_dataset; RAWP=/home/grease/g1_robot_data/pico_raw
T=picoset_20261015
PY=/home/grease/miniforge3/envs/env_isaaclab/bin/python
for d in "$GAM/logs_pkl/$T" "$GAM/logs_pkl/${T}_FPS30" "$GMR/${T}_smplx_npz_fps30" "$GMR/${T}_retargeted_g1_fps30" \
         "$GMR/${T}_retargeted_g1_fps30_motion_lib" "$GMR/${T}_retargeted_g1_fps30_pre_fix" "$EGO/$T" "$EGO/${T}_pre_fix" "$RAWP/20261015_combined_clips"; do
  [ -e "$d" ] && [ ! -e "${d}_v1" ] && mv "$d" "${d}_v1"
done
for s in 20261015_160754 20261015_161659; do $PY $GAM/gear_sonic/scripts/tag_clips.py $RAWP/${s}_clips > /tmp/tag_${s}.out 2>&1; done
$PY $GAM/data_process/combine_picoset_20261015_clips.py
cd $GAM && ./data_process/build_picoset_20261015.sh > /tmp/build1015_v2.log 2>&1
tail -n 6 /tmp/build1015_v2.log
echo BUILD_DONE
