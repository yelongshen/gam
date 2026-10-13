#!/usr/bin/env python3
"""Cut the hardest WINDOW_S-second segment from each kept LAFAN1 clip (train + test), from the
REGENERATED robot data (`robot_v2/`, GMR position-driven retarget) and write paired cropped smpl/robot pkls.

Kept families (user decision 2026-10-07): dance, run, jumps, fight, multipleActions, fightAndSports
(from both lafan1_trainset and lafan1_evalset).

LAFAN is captured on a stage with steps/ramps: the retargeted root height contains real terrain
(clips start up to ~2 m above the lowest floor). Policies here are trained on flat ground, so a window
is accepted only if it is FLAT: the lowest robot body in every 1.5 s sub-chunk stays within FLAT_TOL of
the window's lowest body (flight phases are fine - the chunk still contains the landing). The window's
floor is then re-zeroed (lowest body = 0) and the root xy re-origined to the window start, exactly like
the eval_subset robot files (xy0 ~ 0, lowest point on the ground).

Window score = mean pelvis-local joint speed of `smpl_joints` (m/s).

Output: sim2real/icl_hard_set/lafan_v2/<train|test>/{smpl,robot}/<clip>__w<start>s.pkl, <split>.txt,
        segments.json (frozen manifest incl. md5).
Usage:  .venv_teleop/bin/python sim2real/pick_lafan_segments_gmr.py [--window 8]
"""
import argparse
import hashlib
import json
import os

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = os.path.expanduser("~/ego_dataset")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icl_hard_set", "lafan_v2")
KEEP = ("dance", "run", "jumps", "fight", "multipleActions", "fightAndSports")
SETS = {"train": "lafan1_trainset", "test": "lafan1_evalset"}
FLAT_TOL = 0.12      # m
MIN_START_Z = 0.55   # pelvis height above the window floor at window start


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


def lowest_body_z(raw):
    """Per-frame world z of the lowest robot body from GMR's raw pkl (root-local body positions)."""
    lbp = np.asarray(raw["local_body_pos"], dtype=np.float64)           # (T,N,3), identity root
    Rr = Rot.from_quat(np.asarray(raw["root_rot"], dtype=np.float64)).as_matrix()   # xyzw
    z = np.asarray(raw["root_pos"], dtype=np.float64)[:, 2, None] + np.einsum("tij,tnj->tni", Rr, lbp)[..., 2]
    return z.min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=8.0)
    args = ap.parse_args()
    manifest, skipped = [], []
    for split, ds in SETS.items():
        for sub in ("smpl", "robot"):
            os.makedirs(f"{OUT}/{split}/{sub}", exist_ok=True)
        rdir = f"{ROOT}/{ds}/robot_v2"
        for fn in sorted(os.listdir(rdir)):
            name = fn[:-4]
            if family(name) not in KEEP:
                continue
            s = joblib.load(f"{ROOT}/{ds}/smpl/{name}.pkl")
            r = first(joblib.load(f"{rdir}/{fn}"))
            with open(f"{ROOT}/{ds}/robot_gmr_raw/{name}.pkl", "rb") as fh:
                import pickle
                raw = pickle.load(fh)
            sj = np.asarray(s["smpl_joints"], dtype=np.float64)
            sfps, rfps = float(s.get("fps", 50)), float(r.get("fps", 30))
            root = np.asarray(r["root_trans_offset"], dtype=np.float64)
            T_s, T_r = len(sj), len(root)
            low = lowest_body_z(raw)                                    # (T_r,)
            W_r = int(args.window * rfps)
            C = int(1.5 * rfps)
            sp = np.linalg.norm(np.diff(sj - sj[:, :1], axis=0), axis=2).mean(1) * sfps
            ridx = np.minimum(np.round(np.arange(T_r - 1) / rfps * sfps).astype(int), len(sp) - 1)
            cs = np.concatenate([[0], np.cumsum(sp[ridx])])
            best, best_t = -1.0, None
            for t0 in range(int(2 * rfps), T_r - W_r - 1, 3):
                seg = low[t0:t0 + W_r]
                floor = seg.min()
                chunk_min = np.array([seg[i:i + C].min() for i in range(0, W_r - C + 1, C // 2)])
                if (chunk_min - floor).max() > FLAT_TOL:
                    continue
                if root[t0, 2] - floor < MIN_START_Z:
                    continue
                v = (cs[t0 + W_r] - cs[t0]) / W_r
                if v > best:
                    best, best_t = v, t0
            if best_t is None:
                skipped.append(name)
                continue
            t0, t1 = best_t, best_t + W_r
            floor = low[t0:t1].min()
            s0 = int(round(t0 / rfps * sfps))
            s1 = min(s0 + int(args.window * sfps), T_s)
            out = f"{name}__w{int(t0 / rfps):03d}s"
            sc, rc = crop(s, s0, s1, T_s), crop(r, t0, t1, T_r)
            rt = np.asarray(rc["root_trans_offset"], dtype=np.float64)
            rt[:, 2] -= floor
            rt[:, :2] -= rt[0, :2]
            rc["root_trans_offset"] = rt.astype(np.float32)
            sc["fps"], rc["fps"] = sfps, int(round(rfps))
            sp_out, rp_out = f"{OUT}/{split}/smpl/{out}.pkl", f"{OUT}/{split}/robot/{out}.pkl"
            joblib.dump(sc, sp_out)
            joblib.dump({out: rc}, rp_out)
            manifest.append({"split": split, "name": out, "orig": name, "family": family(name),
                             "start_s": round(t0 / rfps, 2), "end_s": round(t1 / rfps, 2),
                             "smpl_local_speed_m_s": round(best, 3),
                             "robot_z_min": round(float(rt[:, 2].min()), 3), "robot_z_max": round(float(rt[:, 2].max()), 3),
                             "smpl": sp_out, "robot": rp_out, "smpl_md5": md5(sp_out), "robot_md5": md5(rp_out)})
    json.dump({"window_s": args.window, "keep": list(KEEP), "flat_tol_m": FLAT_TOL, "clips": manifest,
               "skipped_no_flat_window": skipped}, open(f"{OUT}/segments.json", "w"), indent=2)
    for split in SETS:
        sub = [m for m in manifest if m["split"] == split]
        fams = {}
        for m in sub:
            fams[m["family"]] = fams.get(m["family"], 0) + 1
        print(f"{split}: {len(sub)} segments {fams}")
        open(f"{OUT}/{split}.txt", "w").write("\n".join(m["name"] for m in sub) + "\n")
    print("skipped (no flat window):", skipped)


if __name__ == "__main__":
    main()
