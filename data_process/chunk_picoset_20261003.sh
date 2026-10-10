#!/usr/bin/env bash
# Chunk the three 2026-10-03 sessions and merge the clips into one dir
# (<HHMMSS>_clip_XXX.npz + merged clips.csv), like 20260930_combined_clips.
set -euo pipefail
P=/home/grease/g1_robot_data/pico_raw
OUT=$P/20261003_combined_clips
PY=/home/grease/gam/.venv_teleop/bin/python
mkdir -p "$OUT"
first=1
for s in 20261003_143734 20261003_151300 20261003_152025; do
  t=${s#*_}
  [ -f "$P/${s}_clips/clips.csv" ] || $PY /home/grease/gam/gear_sonic/scripts/chunk_pico_session.py "$P/$s" --out "$P/${s}_clips"
  for f in "$P/${s}_clips"/clip_*.npz; do ln -sf "$f" "$OUT/${t}_$(basename "$f")"; done
  if [ $first = 1 ]; then head -1 "$P/${s}_clips/clips.csv" > "$OUT/clips.csv"; first=0; fi
  tail -n +2 "$P/${s}_clips/clips.csv" | sed "s/^/${t}_/" >> "$OUT/clips.csv"
done
echo "clips: $(($(wc -l < "$OUT/clips.csv") - 1))"
