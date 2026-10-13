#!/usr/bin/env bash
# Render every clip in a clips dir to <out>/<clip>_motion.mp4 + <clip>_traj.png (visualize_pico_motion.py
# always writes motion.mp4/trajectory.png, so each clip goes through a scratch dir and is renamed).
#   render_clips.sh <clips_dir> <out_dir>
set -u
CLIPS=$1; OUT=$2
PY=${PY:-/home/grease/gam/.venv_teleop/bin/python}
VIS=/home/grease/gam/gear_sonic/scripts/visualize_pico_motion.py
mkdir -p "$OUT"
n=0; fail=0
for f in "$CLIPS"/clip_*.npz; do
  b=$(basename "$f" .npz)
  [ -s "$OUT/${b}_motion.mp4" ] && continue
  tmp=$(mktemp -d)
  if "$PY" "$VIS" "$f" --out "$tmp" --stride 2 --fps 25 > "$tmp/log.txt" 2>&1 && [ -s "$tmp/motion.mp4" ]; then
    mv "$tmp/motion.mp4" "$OUT/${b}_motion.mp4"; mv "$tmp/trajectory.png" "$OUT/${b}_traj.png"
    n=$((n+1)); echo "[$n] $b ok"
  else
    fail=$((fail+1)); echo "FAIL $b: $(tail -n 2 "$tmp/log.txt" | tr '\n' ' ')"
  fi
  rm -rf "$tmp"
done
echo "done: rendered $n, failed $fail -> $OUT"
