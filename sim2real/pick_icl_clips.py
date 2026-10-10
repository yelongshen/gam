#!/usr/bin/env python3
"""Assemble the ICL (multi-attempt) clip set and write cropped LAFAN windows.

Picks
  eval_subset : worst hardware tracking among the manifest clips (--n-eval)
  amass       : the 12 clips that fail in ALL THREE checkpoints in
                model_eval/notes_amass_108clips_3model_comparison.md §4
                (those that exist in amass_evalset and pass the filters)
  lafan       : LAFAN1 eval clips are ~4 min long, so each is cropped to its
                hardest WINDOW_S-second window (by mean joint speed) and only the
                hard families are kept; cropped smpl + robot pkls are written to
                <out>/cropped/{smpl,robot}/ so stream_clip_mode2.py can stream them.

Output: <out>/icl_set.json (frozen manifest with md5s) and cropped pkls.

Usage:  python sim2real/pick_icl_clips.py [--window 8] [--n-lafan 25]
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import statistics as st
from collections import defaultdict

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/ego_dataset")

AMASS_HARD_CORE = [
    "BMLhandball__S04_Expert__Trial_upper_left_right_234_poses",
    "BMLhandball__S09_Novice__Trial_upper_right_left_168_poses",
    "BMLmovi__Subject_28_F_MoSh__Subject_28_F_21_poses",
    "CMU__05__05_16_stageii",
    "CMU__141__141_15_stageii",
    "CMU__87__87_01_stageii",
    "DFaust__50021__50021_knees_stageii",
    "KIT__200__Kniebeuge01_stageii",
    "KIT__348__walking_fast07_stageii",
    "Transitions__mazen_c3d__punchkarate_stand_stageii",
    "WEIZMANN__66__Normal_StraightLong(14)_stageii",
    "WEIZMANN__67__Normal_StraightLong(4)_stageii",
]
# LAFAN1 families that are dynamic / ground-contact heavy
LAFAN_HARD_FAMILIES = ("jumps", "fight", "fightAndSports", "dance", "run", "multipleActions",
                       "pushAndStumble", "push", "obstacles")
# `ground*` and `fallAndGetUp*` are excluded on purpose (ground-contact motions, 2026-10-07).
EXCLUDED_FAMILIES = ("ground", "fallAndGetUp")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_robot(path):
    d = joblib.load(path)
    key = None
    if isinstance(d, dict) and "dof" not in d:
        key = next(iter(d))
        d = d[key]
    return key, d


def crop(obj, t0, t1, T):
    """Crop every array/list with leading dim == T to [t0, t1)."""
    if isinstance(obj, dict):
        return {k: crop(v, t0, t1, T) for k, v in obj.items()}
    if isinstance(obj, np.ndarray) and obj.ndim >= 1 and obj.shape[0] == T:
        return obj[t0:t1].copy()
    return obj


def hardest_window(dof_deg, fps, win_s):
    """Start index (robot frames) of the window with the highest mean joint speed."""
    W = int(win_s * fps)
    if len(dof_deg) <= W:
        return 0, len(dof_deg)
    jv = (np.abs(np.diff(dof_deg, axis=0)) * fps).mean(1)
    cs = np.concatenate([[0], np.cumsum(jv)])
    scores = (cs[W:] - cs[:-W]) / W
    # skip the first 2 s so the window never starts in a T-pose / idle lead-in
    lo = int(2 * fps)
    s = int(np.argmax(scores[lo:])) + lo if len(scores) > lo else int(np.argmax(scores))
    return s, s + W


def family(name):
    base = name.split("_subject")[0]
    return "".join(c for c in base if not c.isdigit())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=8.0)
    ap.add_argument("--n-lafan", type=int, default=25)
    ap.add_argument("--n-eval", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "icl_hard_set"))
    args = ap.parse_args()
    os.makedirs(f"{args.out}/cropped/smpl", exist_ok=True)
    os.makedirs(f"{args.out}/cropped/robot", exist_ok=True)

    scores = {r["name"]: r for r in csv.DictReader(open(f"{args.out}/all_scores.csv"))}
    clips = []

    # ---- eval_subset: flip clips (sim failures of mixv3b_020000) ----
    # Per-clip sim results: sim_perclip_eval_subset.csv (success 0/1, progress 0..1).
    def perclip(src):
        return {r["clip"]: r for r in csv.DictReader(open(f"{args.out}/sim_perclip_{src}.csv"))}

    es = perclip("eval_subset")
    for c, r in sorted(es.items(), key=lambda kv: float(kv[1]["progress"])):
        if "flip" not in c:
            continue
        clips.append({"name": c, "source": "eval_subset",
                      "why": f"flip clip; sim mixv3b_020000 success={r['success']} "
                             f"progress={r['progress']}",
                      "smpl": f"{ROOT}/eval_subset/smpl/{c}.pkl",
                      "robot": f"{ROOT}/eval_subset/robot/{c}.pkl", "window": None})

    # ---- AMASS: clips that FAIL in sim (mixv3b_020000, 108-clip evalset) ----
    am = perclip("amass_evalset")
    excl = json.load(open(f"{args.out}/exclusions.json"))  # tagged clips, e.g. arm retargeting
    for n, r in sorted(am.items(), key=lambda kv: float(kv[1]["progress"])):
        if r["success"] == "1":
            continue
        if n in excl:
            print(f"[amass] excluded {n}: {excl[n]['tag']}")
            continue
        if n in scores and scores[n]["drop"]:
            print(f"[amass] skipped {n}: {scores[n]['drop']}")
            continue
        clips.append({"name": n, "source": "amass_evalset",
                      "why": f"sim mixv3b_020000 success=0 progress={r['progress']}",
                      "smpl": f"{ROOT}/amass_evalset/smpl/{n}.pkl",
                      "robot": f"{ROOT}/amass_evalset/robot/{n}.pkl", "window": None})

    # ---- LAFAN eval: hardest window of hard families ----
    # NOTE: LAFAN1 robot `dof` is nearly static (mean ~0.8 deg/s vs ~12 deg/s for the
    # SMPL pose of the same clip, identical across clips), so hardness is measured on
    # the streamed SMPL joints instead. Robot frames are cropped by matching time.
    lrows = []
    for sp in sorted(glob.glob(f"{ROOT}/lafan1_evalset/smpl/*.pkl")):
        name = os.path.basename(sp)[:-4]
        if family(name) not in LAFAN_HARD_FAMILIES:
            continue
        sd = joblib.load(sp)
        sfps = float(sd.get("fps", 50))
        # LAFAN `pose_aa` is zero for every non-root joint; the motion lives in
        # `smpl_joints` (the thing that is actually streamed). Use pelvis-local joint
        # speed (m/s), which ignores global translation.
        sj = np.asarray(sd["smpl_joints"], dtype=np.float64)
        loc = sj - sj[:, :1]
        ang = np.linalg.norm(np.diff(loc, axis=0), axis=2).mean(1) * sfps
        W = int(args.window * sfps)
        if len(ang) <= W + int(2 * sfps):
            continue
        cs = np.concatenate([[0], np.cumsum(ang)])
        win = (cs[W:] - cs[:-W]) / W
        lo = int(2 * sfps)  # skip the T-pose / idle lead-in
        s_s = int(np.argmax(win[lo:])) + lo
        rp = f"{ROOT}/lafan1_evalset/robot/{name}.pkl"
        key, rd = load_robot(rp)
        fps = float(rd.get("fps", 30))
        s = int(round(s_s / sfps * fps))
        e = s + int(args.window * fps)
        root = np.asarray(rd["root_trans_offset"], dtype=np.float64)
        if e > len(root):
            continue
        lrows.append((float(win[s_s - lo + lo]), name, rp, key, rd, fps, s, e, sp, sd, s_s,
                      float(root[s:e, 2].max() - root[s:e, 2].min()), float(root[s, 2])))
    lrows = [r for r in lrows if r[12] >= 0.60]  # must start upright
    lrows.sort(key=lambda r: -r[0])
    per_fam = defaultdict(int)
    chosen = []
    for r in lrows:
        f = family(r[1])
        if per_fam[f] >= 4:
            continue
        per_fam[f] += 1
        chosen.append(r)
        if len(chosen) >= args.n_lafan:
            break

    for js, name, rp, key, rd, fps, s, e, sp, sd, s_s, hr, _ in chosen:
        T_s = len(sd["transl"])
        sfps = float(sd.get("fps", 50))
        e_s = min(s_s + int(args.window * sfps), T_s)
        T_r = len(rd["dof"])
        out_name = f"{name}__w{int(s / fps):03d}s"
        sd_c = crop(sd, s_s, e_s, T_s)
        rd_c = crop(rd, s, e, T_r)
        sd_c["fps"] = sfps
        rd_c["fps"] = fps
        sp_out = f"{args.out}/cropped/smpl/{out_name}.pkl"
        rp_out = f"{args.out}/cropped/robot/{out_name}.pkl"
        joblib.dump(sd_c, sp_out)
        joblib.dump({out_name: rd_c} if key else rd_c, rp_out)
        clips.append({"name": out_name, "source": "lafan1_evalset",
                      "why": f"hardest {args.window:g}s window of {name} "
                             f"(mean pelvis-local joint speed {js:.2f} m/s)",
                      "smpl": sp_out, "robot": rp_out,
                      "window": {"orig": name, "start_s": round(s / fps, 2),
                                 "end_s": round(e / fps, 2)}})

    for c in clips:
        c["smpl_md5"], c["robot_md5"] = md5(c["smpl"]), md5(c["robot"])
    with open(f"{args.out}/icl_set.json", "w") as fh:
        json.dump({"version": "icl-v0-picks", "window_s": args.window, "clips": clips}, fh, indent=2)

    cnt = defaultdict(int)
    for c in clips:
        cnt[c["source"]] += 1
    print(dict(cnt), "total", len(clips))
    for c in clips:
        print(f"  {c['source']:15s} {c['name']}   [{c['why']}]")


if __name__ == "__main__":
    main()
