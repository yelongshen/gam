"""Convert an XRoboToolkit (PICO) on-device recording -- `trackingData_<stamp>.txt` next to
`CameraRecord_<stamp>.mp4`, e.g. /home/grease/g1_robot_data/pico_video/20261020_060016 -- into
the CONSOLIDATED clip npz that `pico_to_smpl_filtered.py --dir <clip>.npz --transl_mode
body_pos_w` reads (the format `gear_sonic/scripts/chunk_pico_session.py` writes), so the
`build_picoset_*.sh` chain (stages 1-5) runs on it unchanged.

The headset's own recorder stores, per tracking sample, the same 24 body joints that
`xrt.get_body_joints_pose()` returns live (`Body.joints[i].p` = "x,y,z,qx,qy,qz,qw", Unity
Y-up, world frame). Each frame goes through the same math as the live server
(`pico_manager_thread_server.compute_from_body_poses` -> `process_smpl_joints`), batched:

    smpl_pose    (T, 21, 3)  local axis-angle of SMPL joints 1..21
    body_quat_w  (T, 4)      Z-up, base-rotation-removed root quat, [w, x, y, z]
    smpl_joints  (T, 24, 3)  root-local SMPL joints, Z-up
    body_pos_w   (T, 3)      root world position, PICO Y-up (what --record_root_pos records)

and, for the stair tools, `body_joints_w` (T, 24, 3): every joint's world position, PICO Y-up.
Differences from a `pico_manager_thread_server --record_dir --record_root_pos` capture:

  * frames are the native tracking samples (~90 Hz), not the server's --target_fps resample;
    `timestamp_monotonic` is the device timestamp, so pico_to_smpl_filtered measures the rate.
  * `vr_position` is zero (the live value is the calibrated 3-point pose). pico_to_smpl_filtered
    only reads it for --transl_mode head, which this data does not need.
  * no joint_pos / hand / controller-trigger keys (not read by the build chain).
  * `video_t0_ns`: device time of the first video frame (the trackingData header), so
    `(timestamp_ns - video_t0_ns) * 1e-9 * video_fps` is the video frame of a sample.

Outputs:
    <out>/session.npz            every frame
    <out>_clips/clip_NNN.npz     with --cut stairs: one clip per pass up or down a flight
    <out>_clips/clips.csv        chunk_pico_session.py columns + direction

--cut stairs finds the passes from the pelvis height: below the lower threshold the walker
is at the bottom of the flight, above the upper one at the top (thresholds at 30% / 70% of
the pelvis height range between its 5th and 95th percentiles), and a pass runs from the last
frame at one end to the first frame at the other, padded by --pad seconds of walking.

Usage:
    python data_process/xrt_tracking_to_pico_npz.py \\
        --src /home/grease/g1_robot_data/pico_video/20261020_060016 \\
        --out /home/grease/g1_robot_data/pico_raw/20261020_060016 --cut stairs
    # then the build chain with SESSION=.../pico_raw/20261020_060016 (CLIPS = SESSION_clips)
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as sRot
import torch

# The live server's SMPL parent table (pico_manager_thread_server.PicoPoseStreamer).
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22]


def load_tracking(path):
    """trackingData_*.txt -> (body (T, 24, 7) [x, y, z, qx, qy, qz, qw], stamp_ns (T,), header).
    Samples without 24 body joints and repeats of the previous device stamp are dropped (the
    live PicoReader skips repeated stamps too)."""
    body, stamps = [], []
    with open(path) as f:
        header = json.loads(f.readline())
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            joints = (d.get("Body") or {}).get("joints") or []
            if len(joints) != 24:
                continue
            ns = int(d["timeStampNs"])
            if stamps and ns <= stamps[-1]:
                continue
            body.append([[float(v) for v in j["p"].split(",")] for j in joints])
            stamps.append(ns)
    if not body:
        raise SystemExit(f"no body tracking samples in {path}")
    return np.asarray(body, dtype=np.float64), np.asarray(stamps, dtype=np.int64), header


def smpl_from_body_poses(body, human_joints_info_path, batch=4096):
    """Batched `compute_from_body_poses` + `process_smpl_joints` (same functions, same order)."""
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
    from gear_sonic.trl.utils.torch_transform import (
        angle_axis_to_quaternion,
        compute_human_joints,
        quat_apply,
        quat_inv,
        quaternion_to_angle_axis,
    )

    T = body.shape[0]
    g = sRot.from_quat(body[:, :, 3:7].reshape(-1, 4)) * sRot.from_euler("y", 180, degrees=True)
    g = g.as_quat().reshape(T, 24, 4)
    parent = np.array(PARENTS[1:])
    local = (sRot.from_quat(g[:, parent].reshape(-1, 4)).inv()
             * sRot.from_quat(g[:, 1:].reshape(-1, 4))).as_rotvec().reshape(T, 23, 3)
    root_aa = sRot.from_quat(g[:, 0]).as_rotvec()
    smpl_pose = local[:, :21]

    out_joints, out_quat = [], []
    for a in range(0, T, batch):
        body_pose = torch.from_numpy(local[a:a + batch].reshape(-1, 69)).float()
        global_orient = torch.from_numpy(root_aa[a:a + batch]).float()
        q = smpl_root_ytoz_up(angle_axis_to_quaternion(global_orient))
        joints = compute_human_joints(body_pose=body_pose[..., :63],
                                      global_orient=quaternion_to_angle_axis(q),
                                      human_joints_info_path=human_joints_info_path)
        q = remove_smpl_base_rot(q, w_last=False)
        q_inv = quat_inv(q).unsqueeze(1).repeat(1, joints.shape[1], 1)
        out_joints.append(quat_apply(q_inv, joints).numpy())
        out_quat.append(q.numpy())
    return (smpl_pose.astype(np.float32), np.concatenate(out_quat).astype(np.float32),
            np.concatenate(out_joints).astype(np.float32))


def stair_passes(pelvis_h, t, pad_s):
    """(lo, hi, 'up'|'down') frame ranges of passes between the two ends of a flight."""
    p5, p95 = np.percentile(pelvis_h, [5, 95])
    lo_thr, hi_thr = p5 + 0.3 * (p95 - p5), p5 + 0.7 * (p95 - p5)
    state = np.where(pelvis_h < lo_thr, -1, np.where(pelvis_h > hi_thr, 1, 0))
    idx = np.nonzero(state)[0]
    passes = []
    for prev, i in zip(idx[:-1], idx[1:]):
        if state[i] != state[prev]:
            a = np.searchsorted(t, t[prev] - pad_s)
            b = min(np.searchsorted(t, t[i] + pad_s), len(t))
            passes.append((int(a), int(b), "up" if state[i] > 0 else "down"))
    return passes, (lo_thr, hi_thr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="dir with trackingData_*.txt (or the file)")
    ap.add_argument("--out", required=True, help="output dir; clips go to <out>_clips")
    ap.add_argument("--cut", choices=["none", "stairs"], default="none")
    ap.add_argument("--pad", type=float, default=2.0, help="s of walking kept around a pass")
    ap.add_argument("--min-len", type=float, default=3.0, help="min clip length (s)")
    ap.add_argument("--human-joints-info", default="gear_sonic/data/human/human_joints_info.pkl")
    a = ap.parse_args()

    src = a.src if a.src.endswith(".txt") else glob.glob(os.path.join(a.src, "trackingData_*.txt"))
    if isinstance(src, list):
        if len(src) != 1:
            raise SystemExit(f"expected one trackingData_*.txt in {a.src}, found {len(src)}")
        src = src[0]
    body, ns, header = load_tracking(src)
    t = (ns - ns[0]) * 1e-9
    fps = (len(t) - 1) / t[-1]
    print(f"{src}: {len(t)} samples, {t[-1]:.1f} s, {fps:.2f} Hz")

    smpl_pose, body_quat_w, smpl_joints = smpl_from_body_poses(body, a.human_joints_info)
    T = len(t)
    d = dict(
        smpl_pose=smpl_pose,
        body_quat_w=body_quat_w,
        smpl_joints=smpl_joints,
        body_pos_w=body[:, 0, :3].astype(np.float32),
        body_joints_w=body[:, :, :3].astype(np.float32),
        vr_position=np.zeros((T, 9), dtype=np.float32),
        frame_index=np.arange(T, dtype=np.int64),
        pico_fps=np.full((T, 1), fps, dtype=np.float32),
        timestamp_monotonic=t.reshape(T, 1),
        timestamp_ns=ns,
        video_t0_ns=np.int64(header.get("timeStampNs", 0)),
    )
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "session.npz"), **d)
    print(f"  -> {os.path.join(a.out, 'session.npz')}")
    if a.cut == "none":
        return

    # Pelvis height above the floor, Z-up (PICO Y-up y).
    passes, (lo_thr, hi_thr) = stair_passes(body[:, 0, 1], t, a.pad)
    passes = [p for p in passes if t[p[1] - 1] - t[p[0]] >= a.min_len]
    print(f"  {len(passes)} passes (pelvis y < {lo_thr:.2f} m bottom, > {hi_thr:.2f} m top)")
    clips_dir = a.out.rstrip("/") + "_clips"
    os.makedirs(clips_dir, exist_ok=True)
    rows = []
    per_frame = [k for k, v in d.items() if np.ndim(v) >= 1 and len(v) == T]
    for i, (lo, hi, direction) in enumerate(passes):
        name = f"clip_{i:03d}.npz"
        clip = {k: d[k][lo:hi] for k in per_frame}
        clip.update(video_t0_ns=d["video_t0_ns"], clip_start_index=np.int64(lo),
                    clip_duration_s=np.float32(t[hi - 1] - t[lo]))
        np.savez_compressed(os.path.join(clips_dir, name), **clip)
        P = body[lo:hi, 0, :3]
        rows.append(dict(clip=name, start_frame=lo, end_frame=hi, frames=hi - lo,
                         start_s=round(float(t[lo]), 2), dur_s=round(float(t[hi - 1] - t[lo]), 2),
                         path_m=round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), 2),
                         disp_m=round(float(np.linalg.norm(P[-1] - P[0])), 2),
                         direction=direction))
        print(f"  {name}  {direction:>4}  frames {lo:6d}-{hi:6d}  {rows[-1]['dur_s']:5.1f} s")
    with open(os.path.join(clips_dir, "clips.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {clips_dir}/clips.csv")


if __name__ == "__main__":
    main()
