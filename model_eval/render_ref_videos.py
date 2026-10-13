#!/usr/bin/env python3
"""Side-by-side SMPL (left) vs retargeted G1 (right) videos of the REMAINING early-failing clips.
  .venv_teleop/bin/python model_eval/render_ref_videos.py
Output: /home/grease/g1_robot_data/retarget_fix_vis/remaining/<set>/<clip>.mp4
"""
import os, sys, shutil
sys.path.insert(0, "/home/grease/gam/gear_sonic/scripts")
import visualize_smpl_vs_robot as V

SETS = {
    "picoset_20260928": ("pico0928_", ["clip_018", "clip_021", "clip_003", "clip_005_006", "clip_013"]),
    "picoset_20260929": ("pico0929_", ["clip_054", "clip_010", "clip_053", "clip_011", "clip_046", "clip_077"]),
    "picoset_20260930": ("pico0930_", ["145524_clip_034", "145524_clip_021", "145524_clip_022", "162422_clip_016", "145524_clip_012"]),
}
for ds, (pref, clips) in SETS.items():
    V.SMPL_DIR = f"/home/grease/gam/logs_pkl/{ds}"
    V.ROBOT_DIR = f"/home/grease/GMR/{ds}_retargeted_g1_fps30"
    out = f"/home/grease/g1_robot_data/retarget_fix_vis/remaining/{ds}"
    os.makedirs(out, exist_ok=True)
    for c in clips:
        k = pref + c
        tmp = out + "/_tmp"; os.makedirs(tmp, exist_ok=True)
        try:
            V.render(k, tmp, 3, 400, 30)
            f = [x for x in os.listdir(tmp) if x.startswith(k) and x.endswith(".mp4")][0]
            os.replace(f"{tmp}/{f}", f"{out}/{k}.mp4")
            print("ok", ds, k, flush=True)
        except Exception as e:  # noqa: BLE001
            print("ERR", ds, k, e, flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
