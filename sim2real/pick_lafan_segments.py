#!/usr/bin/env python3
"""Cut the hardest WINDOW_S-second segment from each kept LAFAN1 clip (train + test), using the
REGENERATED robot data (`robot_fixed/`), and write paired cropped smpl/robot pkls so they can be
rendered / streamed.

Kept families (user decision 2026-10-07): dance, run, jumps, fight, multipleActions, fightAndSports
(from both lafan1_trainset and lafan1_evalset). ground*, fallAndGetUp*, walk, obstacles, aiming,
push*, sprint are NOT kept.

Window = highest mean pelvis-local joint speed of `smpl_joints`, subject to the ROBOT reference
being upright: root z at window start >= MIN_START_Z and root z over the window >= MIN_Z
(rejects windows where the retarget lies on the floor / goes underground).

Output: sim2real/icl_hard_set/lafan_v2/<train|test>/{smpl,robot}/<clip>__w<start>s.pkl and
sim2real/icl_hard_set/lafan_v2/segments.json (frozen manifest incl. md5).

Usage: .venv_teleop/bin/python sim2real/pick_lafan_segments.py [--window 8]
"""
import argparse
import hashlib
import json
import os

import joblib
import numpy as np

ROOT = os.path.expanduser("~/ego_dataset")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icl_hard_set", "lafan_v2")
KEEP = ("dance", "run", "jumps", "fight", "multipleActions", "fightAndSports")
SETS = {"train": "lafan1_trainset", "test": "lafan1_evalset"}
MIN_START_Z, MIN_Z = 0.60, 0.35


def family(name):
    return "".join(c for c in name.split("_subject")[0] if not c.isdigit())


def first(d):
    return d[next(iter(d))] if isinstance(d, dict) and "dof" not in d else d


def crop(o, t0, t1, T):
    if isinstance(o, dict):
        return {k: crop(v, t0, t1, T) for k, v in o.items()}
    if isinstance(o, np.ndarray) and o.ndim >= 1 and o.shape[0] == T:
        return o[t0:t1].copy()
    return o


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=8.0)
    args = ap.parse_args()
    manifest, skipped = [], []
    for split, ds in SETS.items():
        for sub in ("smpl", "robot"):
            os.makedirs(f"{OUT}/{split}/{sub}", exist_ok=True)
        rdir = f"{ROOT}/{ds}/robot_fixed"
        for fn in sorted(os.listdir(rdir)):
            name = fn[:-4]
            if family(name) not in KEEP:
                continue
            s = joblib.load(f"{ROOT}/{ds}/smpl/{name}.pkl")
            r = first(joblib.load(f"{rdir}/{fn}"))
            sj = np.asarray(s["smpl_joints"], dtype=np.float64)
            sfps, rfps = float(s.get("fps", 50)), float(r.get("fps", 30))
            root = np.asarray(r["root_trans_offset"], dtype=np.float64)
            T_s, T_r = len(sj), len(root)
            W_r = int(args.window * rfps)
            sp = np.linalg.norm(np.diff(sj - sj[:, :1], axis=0), axis=2).mean(1) * sfps
            # speed per robot frame (nearest SMPL frame), then sliding mean over the window
            ridx = np.minimum(np.round(np.arange(T_r - 1) / rfps * sfps).astype(int), len(sp) - 1)
            sp_r = sp[ridx]
            cs = np.concatenate([[0], np.cumsum(sp_r)])
            best, best_t = -1.0, None
            for t0 in range(int(2 * rfps), T_r - W_r - 1):
                if root[t0, 2] < MIN_START_Z or root[t0:t0 + W_r, 2].min() < MIN_Z:
                    continue
                v = (cs[t0 + W_r] - cs[t0]) / W_r
                if v > best:
                    best, best_t = v, t0
            if best_t is None:
                skipped.append(name)
                continue
            t0, t1 = best_t, best_t + W_r
            s0 = int(round(t0 / rfps * sfps))
            s1 = min(s0 + int(args.window * sfps), T_s)
            out = f"{name}__w{int(t0 / rfps):03d}s"
            sc, rc = crop(s, s0, s1, T_s), crop(r, t0, t1, T_r)
            sc["fps"], rc["fps"] = sfps, rfps
            sp_out, rp_out = f"{OUT}/{split}/smpl/{out}.pkl", f"{OUT}/{split}/robot/{out}.pkl"
            joblib.dump(sc, sp_out)
            joblib.dump({out: rc}, rp_out)
            manifest.append({"split": split, "name": out, "orig": name, "family": family(name),
                             "start_s": round(t0 / rfps, 2), "end_s": round(t1 / rfps, 2),
                             "smpl_local_speed_m_s": round(best, 3),
                             "robot_z_min": round(float(root[t0:t1, 2].min()), 3),
                             "robot_z_max": round(float(root[t0:t1, 2].max()), 3),
                             "smpl": sp_out, "robot": rp_out,
                             "smpl_md5": md5(sp_out), "robot_md5": md5(rp_out)})
    json.dump({"window_s": args.window, "keep": list(KEEP), "clips": manifest, "skipped": skipped},
              open(f"{OUT}/segments.json", "w"), indent=2)
    for split in SETS:
        sub = [m for m in manifest if m["split"] == split]
        fams = {}
        for m in sub:
            fams[m["family"]] = fams.get(m["family"], 0) + 1
        print(f"{split}: {len(sub)} segments {fams}")
    print("skipped (no upright window):", skipped)
    for split in SETS:
        open(f"{OUT}/{split}.txt", "w").write(
            "\n".join(m["name"] for m in manifest if m["split"] == split) + "\n")


if __name__ == "__main__":
    main()
