#!/usr/bin/env python3
"""Rank motion clips by tracking difficulty and pick a "hard set" for the
multi-attempt / in-context-learning (ICL) evaluation.

Pools (all under ~/ego_dataset, each with paired smpl/ + robot/ dirs):
    amass_evalset, lafan1_evalset, eval_subset   -> held-out, usable for ICL eval
    lafan1_trainset                              -> tagged `train` (contaminated:
                                                    the policy may have seen it)

Difficulty is computed from the retargeted robot reference (30 fps `dof`,
`root_trans_offset`) as a mean percentile rank of:
    mean joint speed, peak joint speed, mean joint acceleration (jerk proxy),
    root-height range (jumps / drops), peak root speed, arm ROM, leg ROM.

Filters (clip is dropped, reason is recorded):
    * too short (< MIN_S) or too long (> MAX_S)
    * starts low (root z < MIN_START_Z): the deploy binary only starts from the
      standing init pose, so crawls / lying clips are unreachable
    * ends far below start height is NOT filtered (kneel descents are valid)

Selection: top-N by score, with a per-source quota and a cap on near-duplicate
motion families (same name stem), so one family cannot dominate.

Outputs (in --out, default sim2real/icl_hard_set/):
    all_scores.csv    every clip with its metrics, score, and drop reason
    hard_set.csv      the selected clips
    hard_set.json     frozen manifest: name, source, paths, md5, metrics, split tag

Usage:
    python sim2real/collect_hard_clips.py --n 150
    python sim2real/collect_hard_clips.py --n 200 --include-train
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

import joblib
import numpy as np

ROOT = os.path.expanduser("~/ego_dataset")
POOLS = {
    "amass_evalset": "heldout",
    "lafan1_evalset": "heldout",
    "eval_subset": "heldout",
    "lafan1_trainset": "train",
}
MIN_S, MAX_S = 2.5, 20.0
MIN_START_Z = 0.60  # m; G1 standing pelvis ~0.74-0.79

# HW joint order of robot `dof` (see clip_stats.py)
LEG_IDX = list(range(0, 12))
ARM_IDX = list(range(15, 29))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def family(name):
    """Group near-duplicates: drop take number, subject id and mirror suffix."""
    s = re.sub(r"__A\d+(_M)?$", "", name)
    s = re.sub(r"_M$", "", s)
    s = re.sub(r"_\d+$", "", s)
    s = re.sub(r"\d+$", "", s)
    return s


def clip_metrics(robot_path):
    d = joblib.load(robot_path)
    if isinstance(d, dict) and "dof" not in d:
        d = d[next(iter(d))]
    dof = np.rad2deg(np.asarray(d["dof"], dtype=np.float64))
    fps = float(d.get("fps", 30))
    root = np.asarray(d["root_trans_offset"], dtype=np.float64)
    T = len(dof)
    if T < 5:
        return None
    jv = np.abs(np.diff(dof, axis=0)) * fps
    ja = np.abs(np.diff(dof, n=2, axis=0)) * fps * fps if T > 3 else np.zeros((1, 29))
    rs = np.linalg.norm(np.diff(root, axis=0), axis=1) * fps
    rom = dof.max(0) - dof.min(0)
    return {
        "dur_s": T / fps,
        "start_z": float(root[0, 2]),
        "joint_speed_mean": float(jv.mean()),
        "joint_speed_peak": float(jv.max()),
        "joint_acc_mean": float(ja.mean()),
        "height_range": float(root[:, 2].max() - root[:, 2].min()),
        "root_speed_peak": float(rs.max()),
        "rom_arm": float(rom[ARM_IDX].mean()),
        "rom_leg": float(rom[LEG_IDX].mean()),
    }


SCORE_KEYS = ["joint_speed_mean", "joint_speed_peak", "joint_acc_mean",
              "height_range", "root_speed_peak", "rom_arm", "rom_leg"]


def pct_rank(values):
    order = np.argsort(np.argsort(values))
    return order / max(len(values) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="clips to select")
    ap.add_argument("--include-train", action="store_true",
                    help="allow lafan1_trainset clips (tagged split=train)")
    ap.add_argument("--max-per-family", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "icl_hard_set"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for pool, split in POOLS.items():
        files = sorted(glob.glob(f"{ROOT}/{pool}/robot/*.pkl"))
        print(f"[{pool}] {len(files)} clips")
        for rp in files:
            name = os.path.basename(rp)[:-4]
            sp = f"{ROOT}/{pool}/smpl/{name}.pkl"
            row = {"name": name, "source": pool, "split": split,
                   "family": family(name), "robot": rp, "smpl": sp, "drop": ""}
            if not os.path.exists(sp):
                row["drop"] = "no_smpl"
                rows.append(row)
                continue
            try:
                m = clip_metrics(rp)
            except Exception as exc:  # corrupt / unexpected format
                m = None
                row["drop"] = f"load_error:{type(exc).__name__}"
            if m is None:
                row["drop"] = row["drop"] or "empty"
                rows.append(row)
                continue
            row.update(m)
            if m["dur_s"] < MIN_S:
                row["drop"] = "too_short"
            elif m["dur_s"] > MAX_S:
                row["drop"] = "too_long"
            elif m["start_z"] < MIN_START_Z:
                row["drop"] = "starts_low"
            rows.append(row)

    # same clip may appear in eval_subset and another pool: keep the first
    seen, uniq = set(), []
    for r in rows:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        uniq.append(r)
    rows = uniq

    ok = [r for r in rows if not r["drop"]]
    # percentile ranks are computed within the *kept* pool so filters don't skew them
    for k in SCORE_KEYS:
        pr = pct_rank(np.array([r[k] for r in ok]))
        for r, p in zip(ok, pr):
            r[f"pr_{k}"] = float(p)
    for r in ok:
        r["score"] = float(np.mean([r[f"pr_{k}"] for k in SCORE_KEYS]))

    cand = [r for r in ok if args.include_train or r["split"] == "heldout"]
    cand.sort(key=lambda r: -r["score"])

    sel, per_fam = [], Counter()
    for r in cand:
        if per_fam[r["family"]] >= args.max_per_family:
            continue
        sel.append(r)
        per_fam[r["family"]] += 1
        if len(sel) >= args.n:
            break

    # ---- outputs ----
    cols = ["name", "source", "split", "family", "score", "drop", "dur_s", "start_z"] + SCORE_KEYS
    with open(os.path.join(args.out, "all_scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r.get("score", -1)):
            w.writerow(r)
    with open(os.path.join(args.out, "hard_set.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sel:
            w.writerow(r)

    manifest = []
    for r in sel:
        manifest.append({
            "name": r["name"], "source": r["source"], "split": r["split"],
            "family": r["family"], "score": round(r["score"], 4),
            "smpl": r["smpl"], "smpl_md5": md5(r["smpl"]),
            "robot": r["robot"], "robot_md5": md5(r["robot"]),
            **{k: round(r[k], 3) for k in ["dur_s"] + SCORE_KEYS},
        })
    with open(os.path.join(args.out, "hard_set.json"), "w") as fh:
        json.dump({"version": "icl-v0", "n": len(manifest),
                   "filters": {"min_s": MIN_S, "max_s": MAX_S, "min_start_z": MIN_START_Z},
                   "clips": manifest}, fh, indent=2)

    # ---- summary ----
    print(f"\nscanned {len(rows)} unique clips, {len(ok)} pass filters")
    print("dropped:", dict(Counter(r["drop"] for r in rows if r["drop"])))
    print(f"selected {len(sel)} (max {args.max_per_family} per family)")
    print("by source:", dict(Counter(r["source"] for r in sel)))
    print("by split :", dict(Counter(r["split"] for r in sel)))
    fams = defaultdict(int)
    for r in sel:
        fams[r["family"]] += 1
    print(f"distinct families: {len(fams)}")
    print("score range: %.3f .. %.3f" % (sel[-1]["score"], sel[0]["score"]))
    print("top 15:")
    for r in sel[:15]:
        print(f"  {r['score']:.3f}  {r['source']:15s} {r['name']}")
    print(f"\nwrote {args.out}/{{all_scores.csv,hard_set.csv,hard_set.json}}")


if __name__ == "__main__":
    main()
