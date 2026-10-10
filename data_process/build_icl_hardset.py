#!/usr/bin/env python3
"""Assemble the ICL hard set (multi-attempt / in-context-learning evaluation clips).

Sources (all paired smpl/ + robot/ pkls, eval_subset schema):
  * LAFAN1 kept-family 8 s windows, regenerated robot data (GMR position-driven)   sim2real/icl_hard_set/lafan_v2/{train,test}
      dance, run, jumps, fight, multipleActions, fightAndSports
      split tag: lafan_train (policy may have seen lafan1_trainset) / lafan_test
  * AMASS evalset clips that fail in all 3 compared checkpoints (amass11.txt) minus the clips tagged
    hand-arm-retargeting-error (amass_evalset_v2/EXCLUDED.txt)                    ~/ego_dataset/amass_evalset
  * eval_subset flips: flip_360_004__A415, flip_090_003__A304_M                    ~/ego_dataset/eval_subset

Output: ~/ego_dataset/ICL_hardset/{smpl,robot}/<name>.pkl  +  manifest.json  +  README.md
The robot pkl's outer key is verified to equal the file stem (needed by motion-lib key filtering).
"""
import hashlib
import json
import os
import shutil

import joblib
import numpy as np

HOME = os.path.expanduser("~")
ROOT = f"{HOME}/ego_dataset"
L = f"{HOME}/gam/sim2real/icl_hard_set"
OUT = f"{ROOT}/ICL_hardset"
FLIPS = ["flip_360_004__A415", "flip_090_003__A304_M"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def info(sp, rp, name):
    s = joblib.load(sp)
    d = joblib.load(rp)
    key = next(iter(d)) if isinstance(d, dict) and "dof" not in d else None
    r = d[key] if key else d
    dof = np.asarray(r["dof"], dtype=np.float64)
    return {"key_ok": key == name, "dur_s": round(len(dof) / float(r.get("fps", 30)), 2),
            "smpl_frames": int(len(s["transl"])), "robot_frames": int(len(dof))}


def main():
    for sub in ("smpl", "robot"):
        shutil.rmtree(f"{OUT}/{sub}", ignore_errors=True)
        os.makedirs(f"{OUT}/{sub}")
    entries = []

    def add(name, sp, rp, source, split, why):
        shutil.copy2(sp, f"{OUT}/smpl/{name}.pkl")
        shutil.copy2(rp, f"{OUT}/robot/{name}.pkl")
        i = info(f"{OUT}/smpl/{name}.pkl", f"{OUT}/robot/{name}.pkl", name)
        entries.append({"name": name, "source": source, "split": split, "why": why, **i,
                        "smpl_md5": md5(f"{OUT}/smpl/{name}.pkl"), "robot_md5": md5(f"{OUT}/robot/{name}.pkl")})

    seg = json.load(open(f"{L}/lafan_v2/segments.json"))["clips"]
    for m in seg:
        add(m["name"], m["smpl"], m["robot"], f"lafan1_{'trainset' if m['split'] == 'train' else 'evalset'}",
            f"lafan_{m['split']}",
            f"hardest 8 s flat window ({m['start_s']}-{m['end_s']} s of {m['orig']}), family {m['family']}")

    excluded = {l.strip() for l in open(f"{ROOT}/amass_evalset_v2/EXCLUDED.txt") if l.strip()}
    for n in [l.strip() for l in open(f"{L}/amass11.txt") if l.strip()]:
        if n in excluded:
            print("[skip] hand-arm retargeting error:", n)
            continue
        add(n, f"{ROOT}/amass_evalset/smpl/{n}.pkl", f"{ROOT}/amass_evalset/robot/{n}.pkl", "amass_evalset", "heldout",
            "fails in LOW_LATENCY, PRETRAINED and RELEASED (108-clip eval)")
    for n in FLIPS:
        add(n, f"{ROOT}/eval_subset/smpl/{n}.pkl", f"{ROOT}/eval_subset/robot/{n}.pkl", "eval_subset", "heldout",
            "flip; fails in sim (mixv3b_020000)")

    bad = [e["name"] for e in entries if not e["key_ok"]]
    json.dump({"version": "icl-hardset-v1", "n": len(entries), "clips": entries}, open(f"{OUT}/manifest.json", "w"), indent=2)
    by = {}
    for e in entries:
        by[(e["source"], e["split"])] = by.get((e["source"], e["split"]), 0) + 1
    with open(f"{OUT}/README.md", "w") as fh:
        fh.write("# ICL hard set (v1)\n\nPaired `smpl/` + `robot/` pkls in the eval_subset schema, built by "
                 "`data_process/build_icl_hardset.py`. `manifest.json` has per-clip source, split tag, duration and md5.\n\n"
                 "| source | split | clips |\n|---|---|---|\n")
        for (s, sp), n in sorted(by.items()):
            fh.write(f"| {s} | {sp} | {n} |\n")
        fh.write("\n`lafan_train` clips come from lafan1_trainset (the policy may have trained on them): report them "
                 "separately from the held-out sources.\n")
    print(f"{len(entries)} clips -> {OUT}")
    for k, v in sorted(by.items()):
        print("  ", k, v)
    print("robot outer key != file stem:", bad)


if __name__ == "__main__":
    main()
