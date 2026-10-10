#!/usr/bin/env python3
"""Side-by-side SMPL (human) vs retargeted G1 videos, BEFORE and AFTER the retargeting fix, for every re-retargeted clip.

  .venv_teleop/bin/python model_eval/render_before_after.py <name> <clip_list.txt> <smpl_dir> <robot_before_dir> <robot_after_dir> [--stride 3]

Writes  /home/grease/g1_robot_data/retarget_fix_vis/<name>/{before,after,both}/<clip>.mp4
  before / after : visualize_smpl_vs_robot.py output (left SMPL, right G1) with the old / new raw GMR retarget
  both           : before stacked ABOVE after (ffmpeg vstack) for direct comparison
"""
import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, "/home/grease/gam/gear_sonic/scripts")
import visualize_smpl_vs_robot as V  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("clips"); ap.add_argument("smpl_dir")
ap.add_argument("before"); ap.add_argument("after")
ap.add_argument("--stride", type=int, default=3); ap.add_argument("--max-frames", type=int, default=400)
a = ap.parse_args()

clips = [l.strip() for l in open(a.clips) if l.strip()]
root = f"/home/grease/g1_robot_data/retarget_fix_vis/{a.name}"
for sub in ("before", "after", "both"):
    os.makedirs(f"{root}/{sub}", exist_ok=True)
V.SMPL_DIR = a.smpl_dir
ff = shutil.which("ffmpeg") or "ffmpeg"
ok = 0
for i, c in enumerate(clips, 1):
    try:
        for tag, rdir in (("before", a.before), ("after", a.after)):
            V.ROBOT_DIR = rdir
            tmp = f"{root}/{tag}/_tmp"
            os.makedirs(tmp, exist_ok=True)
            V.render(c, tmp, a.stride, a.max_frames, 30)
            produced = [f for f in os.listdir(tmp) if f.startswith(c) and f.endswith(".mp4")]
            os.replace(f"{tmp}/{produced[0]}", f"{root}/{tag}/{c}.mp4")
            shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", f"{root}/before/{c}.mp4", "-i", f"{root}/after/{c}.mp4",
                        "-filter_complex", "vstack=inputs=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{root}/both/{c}.mp4"], check=True)
        ok += 1
        print(f"[{i}/{len(clips)}] {c}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[{i}/{len(clips)}] {c}: ERROR {e}", flush=True)
print(f"done {ok}/{len(clips)} -> {root}")
