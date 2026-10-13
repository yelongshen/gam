#!/usr/bin/env python
"""Build a GROUNDED settle set: like build_settle_set.py (frame 0 held for --settle s), but the held
pose is lowered so the lowest foot sole touches the ground (z = 0) instead of hanging in the air.

Why: most hold failures are clips whose frame 0 is airborne (soles 10-55 cm above the ground). The sim
spawns the robot on that frozen airborne reference and it just falls. Here, per clip:
  * drop = lowest sole z of frame 0 (real foot collision geometry, same as diag_frame0_contact.py)
  * hold frames: root z -= drop            (robot + SMPL transl / smpl_joints z)
  * motion frames: root z -= drop * (1 - smoothstep(t / ramp))  -> the offset fades out over the first
    --ramp seconds of the real motion, so the clip returns to its original heights (no lasting shift).
Joint angles, root orientation and xy are unchanged. Spawn velocity stays 0 (frame 0 repeated).
NOT fixed here: root tilt, CoM outside the support, one-foot-in-the-air poses (reported per clip).

    .venv_sim/bin/python sim2real/build_grounded_settle_set.py
    .venv_sim/bin/python sim2real/build_grounded_settle_set.py --ramp 1.0 --clips run1_subject5__w035s
"""
import argparse
import json
import os
import sys

import joblib
import mujoco
import numpy as np

sys.path.insert(0, os.path.expanduser("~/gam/sim2real"))
from detect_arm_retarget_errors import G1  # noqa: E402
from diag_frame0_contact import FEET, HOLD_CLIPS, geom_points  # noqa: E402

EGO = os.path.expanduser("~/ego_dataset")
ROBOT_KEYS = ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints")
SMPL_KEYS = ("pose_aa", "transl", "smpl_joints")
SHORT_CLIPS = ["run1_subject5__w035s", "run2_subject4__w205s", "jumps1_subject1__w050s", "jumps1_subject2__w169s"]


def rs(n, fps, target=50.0):
    """Loader's resampled length (same rule as torch.arange(0, (n-1)/fps, 1/target))."""
    return n if fps == target else int(np.ceil((n - 1) / fps / (1 / target) - 1e-9))


def pad_front(a, k):
    return np.concatenate([np.repeat(a[:1], k, axis=0), a], axis=0)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def offsets(n_pad, n_motion, drop, ramp_frames):
    """Per-frame z offset (<=0 when drop>0): -drop on the hold, fading to 0 over ramp_frames of motion."""
    off = np.empty(n_pad + n_motion)
    off[:n_pad] = -drop
    t = np.arange(n_motion)
    off[n_pad:] = -drop * (1.0 - smoothstep(t / max(ramp_frames, 1)))
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{EGO}/ICL_hardset_aligned")
    ap.add_argument("--dst", default=f"{EGO}/ICL_hardset_settle")
    ap.add_argument("--clips", nargs="*", default=HOLD_CLIPS)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--ramp", type=float, default=1.0, help="seconds over which the z offset fades out")
    ap.add_argument("--clearance", type=float, default=0.0, help="leave the lowest sole this high (m)")
    ap.add_argument("--short-clips", nargs="*", default=SHORT_CLIPS,
                    help="clips whose start pose cannot be held (tilt / CoM outside support): short settle")
    ap.add_argument("--short-settle", type=float, default=0.2, help="settle seconds for --short-clips")
    ap.add_argument("--copy-rest-from", default=f"{EGO}/ICL_hardset_settle2",
                    help="copy every other clip (already settled, unchanged) from this set; '' to skip")
    a = ap.parse_args()
    src, dst = os.path.expanduser(a.src), os.path.expanduser(a.dst)
    assert not os.path.exists(dst), f"{dst} exists"
    os.makedirs(f"{dst}/robot")
    os.makedirs(f"{dst}/smpl")

    g = G1()
    m, d = g.m, g.d
    body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in FEET]
    foot_geoms = [[i for i in range(m.ngeom) if m.geom_bodyid[i] == b] for b in body_ids]

    def lowest_sole(root, quat, dof):
        d.qpos[:] = 0
        d.qpos[0:3] = root
        d.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
        for i, adr in enumerate(g.adr):
            d.qpos[adr] = dof[i]
        mujoco.mj_kinematics(m, d)
        return [np.vstack([geom_points(m, d, gi) for gi in gs])[:, 2].min() for gs in foot_geoms]

    info = []
    print(f"{'clip':34s} {'soleL':>6s} {'soleR':>6s} {'drop':>6s}  after: soleL soleR")
    for name in a.clips:
        rd = joblib.load(f"{src}/robot/{name}.pkl")
        key = next(iter(rd)) if "dof" not in rd else None
        r = rd[key] if key else rd
        s = joblib.load(f"{src}/smpl/{name}.pkl")
        fps, sfps = float(r.get("fps", 30.0)), float(s.get("fps", 50.0))
        n0 = len(r["dof"])
        settle = a.short_settle if name in a.short_clips else a.settle
        pad_r = int(round(settle * fps))

        zl, zr = lowest_sole(np.asarray(r["root_trans_offset"][0], dtype=np.float64),
                             np.asarray(r["root_rot"][0], dtype=np.float64),
                             np.asarray(r["dof"][0], dtype=np.float64))
        drop = min(zl, zr) - a.clearance

        # ---- robot
        for k in ROBOT_KEYS:
            if k in r and hasattr(r[k], "shape") and r[k].shape[0] == n0:
                r[k] = pad_front(np.asarray(r[k]), pad_r)
        off_r = offsets(pad_r, n0, drop, int(round(a.ramp * fps)))
        rt = np.asarray(r["root_trans_offset"], dtype=np.float64).copy()
        rt[:, 2] += off_r
        r["root_trans_offset"] = rt.astype(np.asarray(r["root_trans_offset"]).dtype)

        # ---- smpl (pad so len(smpl) == loader-resampled robot length)
        s_len = len(s["transl"])
        target = rs(n0 + pad_r, fps)
        pad_s = target - s_len
        assert pad_s >= 0, (name, target, s_len)
        for k in SMPL_KEYS:
            if k in s and hasattr(s[k], "shape") and s[k].shape[0] == s_len:
                s[k] = pad_front(np.asarray(s[k]), pad_s)
        off_s = offsets(pad_s, s_len, drop, int(round(a.ramp * sfps)))
        tr = np.asarray(s["transl"], dtype=np.float64).copy()
        tr[:, 2] += off_s
        s["transl"] = tr.astype(np.asarray(s["transl"]).dtype)
        if "smpl_joints" in s and np.abs(s["smpl_joints"]).max() > 0:
            sj = np.asarray(s["smpl_joints"], dtype=np.float64).copy()
            sj[:, :, 2] += off_s[:, None]
            s["smpl_joints"] = sj.astype(np.asarray(s["smpl_joints"]).dtype)

        assert len(s["transl"]) == rs(len(r["dof"]), fps), name
        joblib.dump({key: r} if key else r, f"{dst}/robot/{name}.pkl")
        joblib.dump(s, f"{dst}/smpl/{name}.pkl")

        nl, nr = lowest_sole(np.asarray(r["root_trans_offset"][0], dtype=np.float64),
                             np.asarray(r["root_rot"][0], dtype=np.float64),
                             np.asarray(r["dof"][0], dtype=np.float64))
        print(f"{name[:34]:34s} {zl:6.3f} {zr:6.3f} {drop:6.3f}  {nl:11.3f} {nr:5.3f}")
        info.append({"name": name, "drop_m": round(float(drop), 4), "sole_before": [round(float(zl), 4), round(float(zr), 4)],
                     "sole_after": [round(float(nl), 4), round(float(nr), 4)], "robot_pad_frames": pad_r,
                     "smpl_pad_frames": int(pad_s), "ramp_s": a.ramp, "robot_frames": len(r["dof"]),
                     "smpl_frames": len(s["transl"]), "settle_s": settle, "grounded": True,
                     "motion_start_s": round(pad_r / fps, 3)})

    n_fixed = len(info)
    if a.copy_rest_from:
        import shutil
        rest = os.path.expanduser(a.copy_rest_from)
        old_info = {i["name"]: i for i in json.load(open(f"{rest}/settle_info.json"))}
        done = {i["name"] for i in info}
        for fn in sorted(os.listdir(f"{rest}/robot")):
            n = fn[:-4]
            if n in done:
                continue
            shutil.copy2(f"{rest}/robot/{fn}", f"{dst}/robot/{fn}")
            shutil.copy2(f"{rest}/smpl/{fn}", f"{dst}/smpl/{fn}")
            info.append({**old_info[n], "grounded": False})
        print(f"copied {len(info) - n_fixed} unchanged clips from {rest}")

    base = f"{EGO}/ICL_hardset/manifest.json"
    man = json.load(open(base)) if os.path.exists(base) else {"clips": []}
    man.update({"settle_s": a.settle, "short_settle_s": a.short_settle, "short_clips": a.short_clips,
                "note": "ICL_hardset_aligned + 2 s settle; hold clips grounded (lowest sole at z=0, offset fades "
                        "over ramp s); 4 unholdable clips get a short settle (sim2real/build_grounded_settle_set.py)"})
    json.dump(man, open(f"{dst}/manifest.json", "w"), indent=2)
    json.dump(info, open(f"{dst}/settle_info.json", "w"), indent=1)
    print(f"\n{len(info)} clips ({n_fixed} re-built) -> {dst}")


if __name__ == "__main__":
    main()
