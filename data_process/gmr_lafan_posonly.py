#!/usr/bin/env python3
"""LAFAN1 (smpl_filtered `smpl_joints`) -> Unitree G1 with GMR, driven by joint POSITIONS only.

Why: LAFAN only provides joint positions in our smpl_filtered files. The SOMA-BVH route had to invent
joint orientations (twist is not observable) and produced limit-pinned arms, a ~25 deg pelvis tilt and a
pinned waist. Here the IK targets are positions for pelvis / hips / knees / ankles / shoulders / elbows /
wrists, plus only the orientations that ARE well defined from positions:
  pelvis (hip lateral x spine up), torso (shoulder lateral x spine up), feet (toe heading, flat).
Arms and hands get no orientation target, so wrist roll/pitch/yaw stay near neutral instead of spinning.

Run with the `gmr` conda env:
  ~/miniforge3/envs/gmr/bin/python data_process/gmr_lafan_posonly.py --set lafan1_evalset \
      --out ~/ego_dataset/lafan1_evalset/robot_gmr_raw [--names a b c] [--workers 8] [--start 0 --end N]
Then convert to motion_lib with GMR/scripts/amass_pipeline/04_convert_to_motion_lib.py.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R

GMR_ROOT = os.path.expanduser("~/GMR")
sys.path.insert(0, GMR_ROOT)
ROOT = os.path.expanduser("~/ego_dataset")

# G1 link -> (human body, pos_weight, rot_weight)
TABLE = {
    "pelvis": ("Hips", 100, 10),
    # G1's hip / shoulder joints sit ~20 / ~15 cm from the LAFAN joints (different skeleton), so matching
    # them only distorts the pose: leave them free; elbows / wrists / knees / ankles carry the pose.
    "left_hip_yaw_link": ("LeftUpLeg", 0, 0),
    "right_hip_yaw_link": ("RightUpLeg", 0, 0),
    "left_knee_link": ("LeftLeg", 10, 0),
    "right_knee_link": ("RightLeg", 10, 0),
    "left_ankle_roll_link": ("LeftFootMod", 50, 3),
    "right_ankle_roll_link": ("RightFootMod", 50, 3),
    "torso_link": ("Spine2", 0, 10),
    "left_shoulder_yaw_link": ("LeftArm", 0, 0),
    "right_shoulder_yaw_link": ("RightArm", 0, 0),
    "left_elbow_link": ("LeftForeArm", 30, 0),
    "right_elbow_link": ("RightForeArm", 30, 0),
    "left_wrist_yaw_link": ("LeftHand", 40, 0),
    "right_wrist_yaw_link": ("RightHand", 40, 0),
}
# G1 wrist roll / pitch / yaw (and the hand) do not move any tracked body, and LAFAN gives no hand
# orientation, so the IK leaves them wherever limits / damping push them (pinned at limits). Hold them
# at neutral instead (indices into the 29 DoF vector, MuJoCo actuator order).
WRIST_DOF = [19, 20, 21, 26, 27, 28]
LEG_LAT = 0.68   # G1 hip half-width 0.0645 m / (LAFAN 0.1055 m * GMR leg scale 0.9)
# human body -> smpl_joints index (LAFAN skeleton stored in SMPL-24 slots)
IDX = {"Hips": 0, "LeftUpLeg": 1, "RightUpLeg": 2, "LeftLeg": 4, "RightLeg": 5, "LeftFootMod": 7, "RightFootMod": 8,
       "Spine2": 9, "LeftArm": 16, "RightArm": 17, "LeftForeArm": 18, "RightForeArm": 19, "LeftHand": 20, "RightHand": 21}
SCALE = {"Hips": 0.9, "Spine2": 0.9, "LeftUpLeg": 0.9, "RightUpLeg": 0.9, "LeftLeg": 0.9, "RightLeg": 0.9,
         "LeftFootMod": 0.9, "RightFootMod": 0.9, "LeftArm": 0.75, "RightArm": 0.75, "LeftForeArm": 0.75,
         "RightForeArm": 0.75, "LeftHand": 0.75, "RightHand": 0.75}


def make_config(path):
    tbl = {link: [body, pw, rw, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]] for link, (body, pw, rw) in TABLE.items()}
    cfg = {"robot_root_name": "pelvis", "human_root_name": "Hips", "ground_height": 0.0,
           "human_height_assumption": 1.8, "use_ik_match_table1": False, "use_ik_match_table2": True,
           "human_scale_table": SCALE, "ik_match_table1": tbl, "ik_match_table2": tbl}  # GMR reads offsets from table1
    json.dump(cfg, open(path, "w"), indent=1)


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def frame_quat(forward, left, up):
    """x=forward, y=left, z=up -> wxyz. Re-orthogonalised from (left, up)."""
    up = unit(up)
    fwd = unit(np.cross(left, up))
    lft = np.cross(up, fwd)
    M = np.stack([fwd, lft, up], axis=-1)
    return R.from_matrix(M).as_quat(scalar_first=True)


def human_frames(sj, transl):
    """(T,24,3) local joints + (T,3) pelvis world -> list of GMR human_data dicts (positions + frame quats)."""
    T = len(sj)
    world = sj - sj[:, :1] + transl[:, None, :]
    # LAFAN hips are 0.21 m apart, G1's 0.13 m. GMR scales the human uniformly (x0.9), so the feet would be
    # asked to stand ~1.5x wider than the G1 hips and the legs splay (hip_roll / ankle_roll pinned at their
    # limits). Compress the LATERAL offset (pelvis frame) of knees and feet by the hip-width ratio.
    up0 = world[:, 6] - world[:, 0]
    lf0 = world[:, 1] - world[:, 2]
    f0 = unit(np.cross(lf0, unit(up0)))
    Mp = np.stack([f0, np.cross(unit(up0), f0), unit(up0)], axis=-1)           # (T,3,3) body->world
    for k in (4, 5, 7, 8, 10, 11):
        d = world[:, k] - world[:, 0]
        db = np.einsum("tji,tj->ti", Mp, d)
        db[:, 1] *= LEG_LAT
        world[:, k] = world[:, 0] + np.einsum("tij,tj->ti", Mp, db)
    up_h = world[:, 6] - world[:, 0]
    left_h = world[:, 1] - world[:, 2]
    q_pelvis = R.from_matrix(np.stack([
        unit(np.cross(left_h, unit(up_h))), np.cross(unit(up_h), unit(np.cross(left_h, unit(up_h)))), unit(up_h)], axis=-1)
    ).as_quat(scalar_first=True)
    up_t = world[:, 12] - world[:, 6]
    left_t = world[:, 16] - world[:, 17]
    # LAFAN's spine2->neck segment leans forward ~10-20 deg in the skeleton's own rest pose (it is not
    # a bend of the actor). Remove that constant lean using frame 0 (upright standing) as the reference,
    # expressed in the pelvis frame, so only the actual torso bending is passed to the waist.
    Rp = R.from_quat(q_pelvis, scalar_first=True).as_matrix()               # (T,3,3) body->world
    up_b = np.einsum("tji,tj->ti", Rp, unit(up_t))                           # up in pelvis frame
    n0 = min(5, T)
    rest = unit(up_b[:n0].mean(0))
    zb = np.array([0.0, 0.0, 1.0])
    ax = np.cross(rest, zb)
    s_, c_ = np.linalg.norm(ax), float(rest @ zb)
    C = R.from_rotvec(ax / s_ * np.arctan2(s_, c_)).as_matrix() if s_ > 1e-9 else np.eye(3)
    up_t = np.einsum("tij,tj->ti", Rp, up_b @ C.T)
    q_torso = R.from_matrix(np.stack([
        unit(np.cross(left_t, unit(up_t))), np.cross(unit(up_t), unit(np.cross(left_t, unit(up_t)))), unit(up_t)], axis=-1)
    ).as_quat(scalar_first=True)
    z = np.array([0.0, 0.0, 1.0])

    def foot_q(ank, toe):
        f = world[:, toe] - world[:, ank]
        f[:, 2] = 0.0
        f = unit(f)
        y = np.cross(np.broadcast_to(z, f.shape), f)
        return R.from_matrix(np.stack([f, y, np.broadcast_to(z, f.shape)], axis=-1)).as_quat(scalar_first=True)
    q_lf, q_rf = foot_q(7, 10), foot_q(8, 11)
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    used = {body for body, pw, rw in TABLE.values() if pw or rw}   # GMR needs data only for bodies with a task
    frames = []
    for t in range(T):
        d = {}
        for body, i in IDX.items():
            if body not in used:
                continue
            q = ident
            if body == "Hips":
                q = q_pelvis[t]
            elif body == "Spine2":
                q = q_torso[t]
            elif body == "LeftFootMod":
                q = q_lf[t]
            elif body == "RightFootMod":
                q = q_rf[t]
            d[body] = [world[t, i].copy(), q.copy()]
        frames.append(d)
    return frames


_G = None


def _init(cfg_path):
    global _G
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    import general_motion_retargeting.params as P
    P.IK_CONFIG_DICT["lafan_posonly"] = {"unitree_g1": cfg_path}
    _G = (GMR, P)


def retarget_one(args):
    name, in_dir, out_dir, cfg_path, start, end = args
    out = os.path.join(out_dir, f"{name}.pkl")
    if os.path.exists(out):
        return name, "skip", 0.0
    t0 = time.time()
    GMR, _ = _G
    d = joblib.load(os.path.join(in_dir, f"{name}.pkl"))
    sj = np.asarray(d["smpl_joints"], dtype=np.float64)[start:end]
    tr = np.asarray(d["transl"], dtype=np.float64)[start:end]
    fps = float(d.get("fps", 30.0))
    frames = human_frames(sj, tr)
    rt = GMR(src_human="lafan_posonly", tgt_robot="unitree_g1", actual_human_height=1.75, verbose=False)
    def step(f):
        """One IK step; if numeric drift leaves the configuration marginally outside the joint limits
        (mink raises NotWithinConfigurationLimits), clamp it back to the limits and retry."""
        for attempt in range(3):
            try:
                return rt.retarget(f).copy()
            except Exception as exc:  # noqa: BLE001
                if "NotWithinConfigurationLimits" not in type(exc).__name__ and attempt == 2:
                    raise
                m = rt.model
                q = rt.configuration.q.copy()
                for j in range(m.njnt):
                    if m.jnt_limited[j]:
                        a = m.jnt_qposadr[j]
                        q[a] = np.clip(q[a], m.jnt_range[j, 0], m.jnt_range[j, 1])
                rt.configuration.update(q)
        return rt.retarget(f).copy()

    for _ in range(30):                    # converge the first frame from the default pose
        step(frames[0])
    qpos = np.array([step(f) for f in frames])
    qpos[:, 7 + np.array(WRIST_DOF)] = 0.0
    root_pos = qpos[:, :3].copy()
    root_rot = qpos[:, [4, 5, 6, 3]].copy()   # wxyz -> xyzw
    dof = qpos[:, 7:].copy()
    import torch
    from general_motion_retargeting.kinematics_model import KinematicsModel
    km = KinematicsModel(rt.xml_file, device="cpu")
    fkp, fkr = torch.zeros((len(dof), 3)), torch.zeros((len(dof), 4))
    fkr[:, -1] = 1.0
    local_body_pos, _ = km.forward_kinematics(fkp, fkr, torch.from_numpy(dof).float())
    body_pos, _ = km.forward_kinematics(torch.from_numpy(root_pos).float(), torch.from_numpy(root_rot).float(),
                                        torch.from_numpy(dof).float())
    root_pos[:, 2] -= float(body_pos[..., 2].min())          # lowest body point on the ground
    root_pos[:, :2] -= root_pos[0, :2]                       # start at the origin (as eval_subset)
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"fps": int(round(fps)), "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof,
                     "local_body_pos": local_body_pos.numpy(), "link_body_list": km.body_names}, fh)
    return name, "ok", time.time() - t0


def _safe(job):
    try:
        return retarget_one(job)
    except Exception as exc:  # noqa: BLE001  (never let one clip hang the pool)
        return job[0], f"FAILED {type(exc).__name__}: {exc}"[:150], 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="lafan1_evalset")
    ap.add_argument("--src", default=f"{ROOT}/lafan1_smpl_filtered_FPS30", help="30 fps smpl_filtered pkls")
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    cfg_path = os.path.join(out, "_ik_config.json")
    make_config(cfg_path)
    names = a.names or sorted(f[:-4] for f in os.listdir(f"{ROOT}/{a.set}/smpl") if f.endswith(".pkl"))
    jobs = [(n, a.src, out, cfg_path, a.start, a.end) for n in names]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init, initargs=(cfg_path,)) as pool:
        for n, st, dt in pool.imap_unordered(_safe, jobs):
            print(f"{n}: {st} {dt:.0f}s", flush=True)
    print(f"done {len(jobs)} clips in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
