#!/usr/bin/env python
"""Build a SETTLE copy of a paired robot/smpl motion dataset: the first frame is held for SETTLE seconds before the real
motion starts. Purpose: (1) test whether the policy can stay stable at the clip's start pose for 2 s, and (2) separate
"cannot hold the start pose" failures from "cannot execute the motion" failures (a clip that fails within the hold
never reached the motion; one that survives it and fails right after the hold fails on motion onset).

Robot (30 fps native): frame 0 repeated PAD_R = round(settle*fps) times at the front, for root_trans_offset / pose_aa /
dof / root_rot (smpl_joints in the robot file are zeros and padded too). SMPL (50 Hz): frame 0 repeated at the front of
pose_aa / transl / smpl_joints, with the pad length chosen so that len(smpl) == the loader's resampled robot length
(torch.arange rule of dev_notes/fps_check_alignment; input must already be aligned). original_* fields are left alone.

    ~/miniforge3/envs/env_isaaclab/bin/python data_process/build_settle_set.py \
        --src ~/ego_dataset/ICL_hardset_aligned --dst ~/ego_dataset/ICL_hardset_settle2 --settle 2.0
"""
import argparse
import glob
import json
import os
import shutil

import joblib
import numpy as np
import torch

ROBOT_KEYS = ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints")
SMPL_KEYS = ("pose_aa", "transl", "smpl_joints")


def rs(n, fps, target=50.0):
    return n if fps == target else len(torch.arange(0, (n - 1) / fps, 1 / target))


def pad_front(a, k):
    return np.concatenate([np.repeat(a[:1], k, axis=0), a], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--settle", type=float, default=2.0)
    a = ap.parse_args()
    src, dst = os.path.expanduser(a.src), os.path.expanduser(a.dst)
    assert not os.path.exists(dst), f"{dst} exists"
    os.makedirs(f"{dst}/robot")
    os.makedirs(f"{dst}/smpl")
    info = []
    for rp in sorted(glob.glob(f"{src}/robot/*.pkl")):
        name = os.path.basename(rp)[:-4]
        rd = joblib.load(rp)
        key = next(iter(rd)) if "dof" not in rd else None
        r = rd[key] if key else rd
        s = joblib.load(f"{src}/smpl/{name}.pkl")
        fps = float(r.get("fps", 30.0))
        n0 = len(r["dof"])
        pad_r = int(round(a.settle * fps))
        for k in ROBOT_KEYS:
            if k in r and hasattr(r[k], "shape") and r[k].shape[0] == n0:
                r[k] = pad_front(np.asarray(r[k]), pad_r)
        target = rs(n0 + pad_r, fps)
        s_len = len(s["transl"])
        pad_s = target - s_len
        assert pad_s >= 0, (name, target, s_len)
        for k in SMPL_KEYS:
            if k in s and hasattr(s[k], "shape") and s[k].shape[0] == s_len:
                s[k] = pad_front(np.asarray(s[k]), pad_s)
        joblib.dump({key: r} if key else r, f"{dst}/robot/{name}.pkl")
        joblib.dump(s, f"{dst}/smpl/{name}.pkl")
        assert len(s["transl"]) == rs(len(r["dof"]), fps), name
        info.append({"name": name, "robot_pad_frames": pad_r, "smpl_pad_frames": int(pad_s), "robot_frames": len(r["dof"]),
                     "smpl_frames": len(s["transl"]), "settle_s": a.settle,
                     "motion_start_s": round(pad_r / fps, 3)})
    if os.path.exists(f"{src}/manifest.json"):
        m = json.load(open(f"{src}/manifest.json"))
        m["settle_s"] = a.settle
        m["note"] = f"first frame held for {a.settle}s before the motion (data_process/build_settle_set.py)"
        json.dump(m, open(f"{dst}/manifest.json", "w"), indent=2)
    json.dump(info, open(f"{dst}/settle_info.json", "w"), indent=1)
    shutil.copy(__file__, f"{dst}/build_settle_set.py")
    print(f"{len(info)} clips -> {dst}; robot pad {info[0]['robot_pad_frames']} frames, smpl pad {sorted({i['smpl_pad_frames'] for i in info})}")


if __name__ == "__main__":
    main()
