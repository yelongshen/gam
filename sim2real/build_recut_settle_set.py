#!/usr/bin/env python
"""Re-cut ICL hold-failure clips at a clean start frame and build their 2 s settle set.

Input: sim2real/clean_start_frames.csv from find_clean_start_window.py. Only rows with
clean == True are re-cut (their frame 0 has a foot on the ground, CoM inside the support
and low tilt, so holding it is physically feasible). The window keeps its length; it is
slid from the old start to new_start_frame.

Conventions copied from the existing ICL_hardset_aligned windows (verified bit-exact):
  robot : crop of <set>/robot_v2/<seq>.pkl ; root xy re-based to 0 at the window start ;
          root z shifted by the clip's constant dz ; other fields cropped unchanged.
  smpl  : crop of <set>/smpl/<seq>.pkl at round(t0/30*50), same length as the aligned window.
Then data_process/build_settle_set.py is run on the result (frame 0 held --settle s).

    .venv_sim/bin/python sim2real/build_recut_settle_set.py
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import joblib
import numpy as np

EGO = os.path.expanduser("~/ego_dataset")
GAM = os.path.expanduser("~/gam")
ROBOT_KEYS_SKIP = ("fps",)


def first(d):
    return d[next(iter(d))] if isinstance(d, dict) and "dof" not in d else d


def crop(o, t0, t1, T):
    if isinstance(o, dict):
        return {k: crop(v, t0, t1, T) for k, v in o.items()}
    if isinstance(o, np.ndarray) and o.ndim >= 1 and o.shape[0] == T:
        return o[t0:t1].copy()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=f"{GAM}/sim2real/clean_start_frames.csv")
    ap.add_argument("--src", default=f"{EGO}/ICL_hardset_aligned")
    ap.add_argument("--dst", default=f"{EGO}/ICL_hardset_recut_aligned")
    ap.add_argument("--settle-dst", default=f"{EGO}/ICL_hardset_recut_settle2")
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--python", default=os.path.expanduser("~/miniforge3/envs/env_isaaclab/bin/python"),
                    help="env with torch, used for build_settle_set.py")
    a = ap.parse_args()

    man = {c["name"]: c for c in json.load(open(f"{EGO}/ICL_hardset/manifest.json"))["clips"]}
    rows = [r for r in csv.DictReader(open(a.csv)) if r["clean"] == "True"]
    if not rows:
        sys.exit("no clean rows in csv")
    assert not os.path.exists(a.dst), f"{a.dst} exists"
    for sub in ("robot", "smpl"):
        os.makedirs(f"{a.dst}/{sub}")

    entries = []
    for r in rows:
        name, base, t0 = r["clip"], r["full_seq"], int(r["new_start_frame"])
        ds = man[name]["source"]
        win = first(joblib.load(f"{a.src}/robot/{name}.pkl"))
        sw = joblib.load(f"{a.src}/smpl/{name}.pkl")
        full = joblib.load(f"{EGO}/{ds}/robot_v2/{base}.pkl")
        outer = next(iter(full)) if "dof" not in full else None
        f = first(full)
        sfull = joblib.load(f"{EGO}/{ds}/smpl/{base}.pkl")
        W, Ws = len(win["dof"]), len(sw["transl"])
        T_r, T_s = len(f["dof"]), len(sfull["transl"])
        fps, sfps = float(f.get("fps", 30.0)), float(sfull.get("fps", 50.0))

        # locate the old window to recover its z shift (and sanity-check the match)
        err = np.abs(np.asarray(f["dof"])[: T_r - W + 1] - np.asarray(win["dof"])[0]).sum(1)
        o = int(err.argmin())
        assert err[o] < 1e-3, (name, err[o])
        root = np.asarray(f["root_trans_offset"], dtype=np.float64)
        dz = float(np.median(np.asarray(win["root_trans_offset"])[:, 2] - root[o:o + W, 2]))

        s0 = int(round(t0 / fps * sfps))
        assert 0 <= t0 and t0 + W <= T_r and s0 + Ws <= T_s, (name, t0, s0, T_r, T_s)
        rc = crop(f, t0, t0 + W, T_r)
        rt = np.asarray(rc["root_trans_offset"], dtype=np.float64).copy()
        rt[:, :2] -= rt[0, :2]
        rt[:, 2] += dz
        rc["root_trans_offset"] = rt.astype(np.asarray(f["root_trans_offset"]).dtype)
        rc["fps"] = fps
        sc = crop(sfull, s0, s0 + Ws, T_s)
        sc["fps"] = sfps
        joblib.dump({name: rc}, f"{a.dst}/robot/{name}.pkl")
        joblib.dump(sc, f"{a.dst}/smpl/{name}.pkl")
        entries.append({"name": name, "source": ds, "orig_start_frame": o, "new_start_frame": t0,
                        "shift_s": round((t0 - o) / fps, 3), "smpl_start_frame": s0, "dz": round(dz, 4),
                        "robot_frames": W, "smpl_frames": Ws})
        print(f"{name}: {o} -> {t0} ({(t0 - o) / fps:+.2f} s), smpl {s0}, dz {dz:+.3f}, {W}/{Ws} frames")

    json.dump({"version": "icl-hardset-recut-v1", "n": len(entries), "clips": entries},
              open(f"{a.dst}/manifest.json", "w"), indent=2)
    subprocess.check_call([a.python, f"{GAM}/data_process/build_settle_set.py", "--src", a.dst,
                           "--dst", a.settle_dst, "--settle", str(a.settle)])
    print(f"\nrecut set    : {a.dst}\nsettle set   : {a.settle_dst}")
    print("verify frame 0: .venv_sim/bin/python sim2real/diag_frame0_contact.py "
          f"--src {a.dst} --clips " + " ".join(e["name"] for e in entries))


if __name__ == "__main__":
    main()
