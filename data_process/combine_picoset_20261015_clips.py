"""Combine the two 2026-10-15 clip dirs into <RAW>/20261015_combined_clips (like 20260930_combined_clips):
  <HHMMSS>_clip_XXX.npz symlinks, a merged clips.csv, and clip_tags.csv = tag_clips.py kinematic tags + the `event` label
  (crawl / jump_A / jump_B) from cut_picoset_20261015_clips.py.  Run tag_clips.py on each session's clips dir first."""
import csv
import os

RAW = "/home/grease/g1_robot_data/pico_raw"
SESS = ["20261015_160754", "20261015_161659"]
OUT = f"{RAW}/20261015_combined_clips"
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    p = os.path.join(OUT, f)
    if os.path.islink(p) or f.endswith(".csv"):
        os.remove(p)
clips_rows, tag_rows = [], []
for s in SESS:
    hh = s.split("_")[1]
    d = f"{RAW}/{s}_clips"
    ev = {r["clip"]: r.get("event", "") for r in csv.DictReader(open(f"{d}/clips.csv"))}
    for r in csv.DictReader(open(f"{d}/clips.csv")):
        for k in sorted(os.listdir(d)):
            pass
        os.symlink(f"{d}/{r['clip']}", f"{OUT}/{hh}_{r['clip']}")
        clips_rows.append({**r, "clip": f"{hh}_{r['clip']}"})
    for r in csv.DictReader(open(f"{d}/clip_tags.csv")):
        name = r["clip"] if r["clip"].endswith(".npz") else r["clip"] + ".npz"
        tag_rows.append({**r, "clip": f"{hh}_{r['clip'].replace('.npz', '')}", "event": ev.get(name, "")})
for name, rows in (("clips.csv", clips_rows), ("clip_tags.csv", tag_rows)):
    with open(f"{OUT}/{name}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
print(f"{len(clips_rows)} clips, {len(tag_rows)} tag rows -> {OUT}")
