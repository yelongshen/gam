#!/usr/bin/env python3
"""smpl_filtered (.pkl, `smpl_joints` positions) -> SOMA-skeleton BVH, v3: NEUTRAL-POSE-RELATIVE, SWING-ONLY.

Why v3: the v2 converter (`convert_smpl_filtered_to_bvh.py`) builds each joint's world frame from the live
bone direction + the HIP-LATERAL vector as twist reference, and aligns it to frames built from the SOMA
template's raw OFFSETs. That is wrong for the SOMA BVH convention:
  * SOMA stores every OFFSET in the parent's bone-aligned local frame -> the all-zero pose is NOT a
    standing pose; the canonical neutral pose is frame 0 of soma_zero_frame0.bvh (spine up, legs down,
    arms hanging, elbows bent forward). v2 compares live bones against raw offsets, so the neutral
    posture mismatch becomes a permanent ~25 deg pelvis tilt (waist_pitch pinned at its limit).
  * arm bones are (anti)parallel to the hip-lateral reference in a T-pose / arms-out poses, so the arm
    twist is degenerate and flips -> shoulder_yaw +-130 deg, elbows pinned, wrists spinning.
v3 instead, for every mapped joint j with primary child c:
    D_j = S_j * D_parent,     W_j = D_j * N_j,     L_j = W_parent^T * W_j
  N_j  = world rotation of j in the SOMA neutral pose (FK of soma_zero_frame0.bvh frame 0)
  S_j  = SHORTEST-ARC rotation taking (D_parent * neutral bone dir of j->c) onto the live bone dir
         (minimal twist w.r.t. the parent: smooth, no degenerate frames)
  Hips: D = F_live * F_neutral^T with frames from (pelvis->spine2 up, hip lateral)  [robust, well posed]
Unmapped / leaf joints (hands, toes, head, fingers, eyes...) inherit their parent's delta, i.e. stay at
the SOMA neutral relative orientation (no invented wrist roll). Bone DIRECTIONS (positions) are exact.

Usage:
  .venv_sim/bin/python convert_smpl_filtered_to_bvh_v3.py --pkl in.pkl --out out.bvh --template <soma>.bvh
  .venv_sim/bin/python convert_smpl_filtered_to_bvh_v3.py --input_dir DIR --output_dir DIR --template ...
"""
import argparse
import glob
import multiprocessing
import os
import sys
import time

import joblib
import numpy as np
import scipy.spatial.transform as sT

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert_smpl_filtered_to_bvh as V  # noqa: E402
import soma_bvh as S  # noqa: E402


def shortest_arc(a, b):
    """(T,3) unit vectors -> (T,3,3) rotations taking a to b with minimal angle."""
    v = np.cross(a, b)
    c = np.clip((a * b).sum(-1), -1.0, 1.0)
    s = np.linalg.norm(v, axis=-1)
    ang = np.arctan2(s, c)
    axis = np.zeros_like(v)
    ok = s > 1e-9
    axis[ok] = v[ok] / s[ok, None]
    anti = (~ok) & (c < 0)  # antiparallel: any perpendicular axis
    if anti.any():
        ref = np.where(np.abs(a[anti, 0:1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        p = np.cross(a[anti], ref)
        axis[anti] = p / np.linalg.norm(p, axis=-1, keepdims=True)
    return sT.Rotation.from_rotvec(axis * ang[:, None]).as_matrix()


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def convert(pkl_path, template_path, out_path, verbose=True):
    d = joblib.load(pkl_path)
    smpl_joints = np.asarray(d["smpl_joints"], dtype=np.float64)   # (T,24,3) Z-up m, root-rotated
    transl = np.asarray(d["transl"], dtype=np.float64)
    fps = float(d.get("fps", 50.0))
    T = len(smpl_joints)

    tpl = S.parse_bvh(template_path)
    neu_src = S.parse_bvh(S.SOMA_ZERO)
    # neutral frame vector on the TEMPLATE skeleton: rotation channels copied by joint name
    neutral = np.zeros(tpl["nch"])
    for j, types in tpl["chan"].items():
        if j in neu_src["start"]:
            s0, s1 = neu_src["start"][j], tpl["start"][j]
            for k, t in enumerate(types):
                if t.endswith("rotation") and k < len(neu_src["chan"][j]):
                    neutral[s1 + k] = neu_src["frames"][0][s0 + k]
    Nw, Pn = S.fk(tpl, neutral)

    # live joints in the BVH raw Y-up convention (raw.x = zup.x, raw.y = zup.z, raw.z = -zup.y)
    sj = np.stack([smpl_joints[..., 0], smpl_joints[..., 2], -smpl_joints[..., 1]], axis=-1)

    name_of = V.SMPL_TO_SOMA
    delta = {}
    eye = np.tile(np.eye(3), (T, 1, 1))

    # ---- Hips (root of the body) ----
    up_l = sj[:, 6] - sj[:, 0]
    lat_l = sj[:, 2] - sj[:, 1]
    up_n = (Pn["Spine2"] - Pn["Hips"])[None]
    lat_n = (Pn["RightLeg"] - Pn["LeftLeg"])[None]
    F_live = V._build_frames_batch(up_l, lat_l)
    F_neu = V._build_frames_batch(up_n, lat_n)[0]
    n0 = min(5, T)
    F_h0 = V._build_frames_batch(unit(up_l[:n0].mean(0))[None], lat_l[:n0].mean(0)[None])[0]
    D0 = F_h0 @ F_neu.T                       # SOMA neutral -> person at frame 0
    delta["Hips"] = np.einsum("tij,kj,kl->til", F_live, F_h0, D0)

    child_of = V.PRIMARY_CHILD
    # Joints whose bone is ~perpendicular to the hip-lateral axis (legs, feet, spine, neck): twist is
    # anchored to the live hip-lateral vector (keeps feet flat / pelvis level, no twist accumulation).
    FRAME_JOINTS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12}
    # Everything else with a primary child (collars, arms, forearms): bone can be (anti)parallel to the
    # lateral axis (arms out), so use swing-only relative to the parent (twist minimal, never degenerate).
    for j in tpl["joints"]:
        if j in ("Root", "Hips"):
            continue
        par = tpl["parent"][j]
        Dp = delta.get(par, eye)
        smpl_idx = next((k for k, v in name_of.items() if v == j), None)
        c = child_of.get(smpl_idx) if smpl_idx is not None else None
        if smpl_idx is None or c is None or smpl_idx in V.LEAF_JOINTS or name_of.get(c) not in Pn:
            delta[j] = Dp
            continue
        cn = name_of[c]
        u_n = unit(Pn[cn] - Pn[j])
        b = unit(sj[:, c] - sj[:, smpl_idx])
        if smpl_idx in FRAME_JOINTS:
            # delta from the PERSON'S OWN frame-0 posture (LAFAN's bind skeleton has non-vertical spine /
            # neck segments, so absolute directions would be read as a permanent forward lean). Frame 0 is
            # upright standing (T-pose) = SOMA neutral for torso and legs; D0 maps SOMA neutral onto the
            # person's frame-0 heading/tilt.
            F_live_j = V._build_frames_batch(b, lat_l)
            n0 = min(5, T)
            F_rest_j = V._build_frames_batch(unit(b[:n0].mean(0))[None], lat_l[:n0].mean(0)[None])[0]
            delta[j] = np.einsum("tij,kj,kl->til", F_live_j, F_rest_j, D0)
        else:
            a = np.einsum("tij,j->ti", Dp, u_n)
            delta[j] = np.einsum("tij,tjk->tik", shortest_arc(a, b), Dp)

    # ---- world / local rotations -> Euler channels ----
    motion = np.zeros((T, tpl["nch"]), dtype=np.float64)
    W = {}
    for j in tpl["joints"]:
        W[j] = np.einsum("tij,jk->tik", delta.get(j, eye), Nw[j]) if j != "Root" else eye
    for j in tpl["joints"]:
        types = tpl["chan"].get(j)
        if not types or not any(t.endswith("rotation") for t in types):
            continue
        par = tpl["parent"][j]
        Rl = W[j] if par is None else np.einsum("tji,tjk->tik", W[par], W[j])
        eul = sT.Rotation.from_matrix(Rl).as_euler("ZYX", degrees=True)
        col = {"Zrotation": 0, "Yrotation": 1, "Xrotation": 2}
        s0 = tpl["start"][j]
        for k, t in enumerate(types):
            if t in col:
                motion[:, s0 + k] = eul[:, col[t]]

    # ---- root translation (same scaling logic as v2) ----
    def dist(a, b):
        return np.linalg.norm(smpl_joints[0, a] - smpl_joints[0, b])
    ll_smpl = (dist(0, 1) + dist(1, 4) + dist(4, 7) + dist(7, 10)) * 100.0
    ll_soma = sum(np.linalg.norm(tpl["offset"].get(n, np.zeros(3))) for n in ("LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase"))
    scale = ll_soma / ll_smpl if ll_smpl > 0 else 1.0
    pos = {"Xposition": transl[:, 0] * 100.0 * scale, "Yposition": transl[:, 2] * 100.0 * scale,
           "Zposition": -transl[:, 1] * 100.0 * scale}
    s0 = tpl["start"]["Hips"]
    for k, t in enumerate(tpl["chan"]["Hips"]):
        if t in pos:
            motion[:, s0 + k] = pos[t]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(tpl["header"]))
        f.write(f"\nMOTION\nFrames: {T}\nFrame Time: {1.0 / fps:.6f}\n")
        for t in range(T):
            f.write(" ".join(f"{v:.6f}" for v in motion[t]) + "\n")
    if verbose:
        print(f"saved {out_path}  ({T} frames @ {fps} fps)")
    return out_path


def _task(a):
    pkl, tpl, out = a
    try:
        convert(pkl, tpl, out, verbose=False)
        return os.path.basename(out), "ok", None
    except Exception as e:  # noqa: BLE001
        return os.path.basename(out), "fail", repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl")
    ap.add_argument("--out")
    ap.add_argument("--input_dir", nargs="*")
    ap.add_argument("--output_dir")
    ap.add_argument("--template", required=True)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()
    if a.pkl:
        convert(a.pkl, a.template, a.out)
        return
    os.makedirs(a.output_dir, exist_ok=True)
    tasks = []
    for d in a.input_dir:
        prefix = os.path.basename(os.path.normpath(d)).replace("_smpl_filtered", "")
        for p in sorted(glob.glob(os.path.join(d, "*.pkl"))):
            n = os.path.splitext(os.path.basename(p))[0]
            tasks.append((p, a.template, os.path.join(a.output_dir, f"{prefix}__{n}.bvh")))
    t0 = time.time()
    ok = fail = 0
    with multiprocessing.Pool(a.num_workers) as pool:
        for name, st, err in pool.imap_unordered(_task, tasks):
            if st == "ok":
                ok += 1
            else:
                fail += 1
                print("[fail]", name, err)
    print(f"done: {ok} ok, {fail} fail, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
