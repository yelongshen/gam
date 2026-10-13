#!/usr/bin/env python3
"""SOMA BVH helpers: hierarchy parser (with parents + channel order) and forward kinematics.

SOMA's BVH stores each joint OFFSET in the parent's LOCAL (bone-aligned) frame, so the
all-zero-rotation pose is not a T-pose; the neutral T-pose is `soma_zero_frame0.bvh` frame 0.
"""
import os

import numpy as np
import scipy.spatial.transform as sT

SOMA_ZERO = os.path.expanduser("~/soma-retargeter/soma_retargeter/configs/soma/soma_zero_frame0.bvh")


def parse_bvh(path):
    """-> dict(joints, parent(name->name|None), offset(name->(3,)), chan(name->list), chan_start(name->int),
    nch, frames (F,nch) float array, frame_time)"""
    lines = open(path).read().split("\n")
    mo = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    joints, parent, offset, chan, start = [], {}, {}, {}, {}
    stack, ci = [], 0
    for line in lines[:mo]:
        s = line.strip()
        if s.startswith("ROOT ") or s.startswith("JOINT "):
            n = s.split()[1]
            joints.append(n)
            parent[n] = stack[-1] if stack else None
            stack.append(n)
        elif s.startswith("End Site"):
            stack.append(None)
        elif s.startswith("OFFSET") and stack and stack[-1] is not None:
            p = s.split()
            offset[stack[-1]] = np.array([float(p[1]), float(p[2]), float(p[3])])
        elif s.startswith("CHANNELS") and stack and stack[-1] is not None:
            p = s.split()
            k = int(p[1])
            chan[stack[-1]] = p[2:2 + k]
            start[stack[-1]] = ci
            ci += k
        elif s == "}" and stack:
            stack.pop()
    nf = int(lines[mo + 1].split(":")[1])
    ft = float(lines[mo + 2].split(":")[1])
    data = np.array([[float(x) for x in l.split()] for l in lines[mo + 3:mo + 3 + nf] if l.strip()])
    return dict(joints=joints, parent=parent, offset=offset, chan=chan, start=start, nch=ci,
                frames=data, frame_time=ft, header=lines[:mo])


def rot_from_channels(types, vals):
    """Compose R = R_ch1 @ R_ch2 @ R_ch3 (BVH order) for the rotation channels; vals in degrees."""
    R = np.eye(3)
    for t, v in zip(types, vals):
        if t.endswith("rotation"):
            ax = t[0].lower()
            R = R @ sT.Rotation.from_euler(ax, v, degrees=True).as_matrix()
    return R


def fk(bvh, frame):
    """World rotation + position (cm) of every joint for a single frame vector."""
    Rw, Pw = {}, {}
    for j in bvh["joints"]:
        types = bvh["chan"].get(j, [])
        st = bvh["start"].get(j)
        vals = frame[st:st + len(types)] if st is not None else []
        Rl = rot_from_channels(types, vals)
        pos_l = bvh["offset"][j].copy()
        for t, v in zip(types, vals):
            if t == "Xposition":
                pos_l[0] += v
            elif t == "Yposition":
                pos_l[1] += v
            elif t == "Zposition":
                pos_l[2] += v
        p = bvh["parent"][j]
        if p is None:
            Rw[j], Pw[j] = Rl, pos_l
        else:
            Rw[j] = Rw[p] @ Rl
            Pw[j] = Pw[p] + Rw[p] @ pos_l
    return Rw, Pw


if __name__ == "__main__":
    b = parse_bvh(SOMA_ZERO)
    print("frames", b["frames"].shape, "joints", len(b["joints"]), "channel order example:", b["chan"]["Hips"], b["chan"]["LeftArm"])
    Rw, Pw = fk(b, b["frames"][0])
    Rz, Pz = fk(b, np.zeros(b["nch"]))

    def d(P, a, c):
        v = P[c] - P[a]
        return np.round(v / np.linalg.norm(v), 2)

    for lbl, P in (("frame0", Pw), ("ALL-ZERO rotations", Pz)):
        print(f"\n{lbl}: pelvis pos {np.round(P['Hips'], 1)}")
        for a, c in [("Hips", "Spine1"), ("Spine1", "Chest"), ("Chest", "Neck1"), ("LeftLeg", "LeftShin"), ("LeftShin", "LeftFoot"),
                     ("LeftFoot", "LeftToeBase"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"), ("RightArm", "RightForeArm")]:
            print(f"  {a:12s}->{c:12s} dir {d(P, a, c)}")
        print("  left hip - right hip:", np.round(P["LeftLeg"] - P["RightLeg"], 1))
    print("\nframe-0 local rotations (deg):", {j: np.round(b["frames"][0][b["start"][j]:b["start"][j] + 6], 1).tolist() for j in ["Hips", "Spine1", "LeftArm", "LeftForeArm", "LeftShin"]})
