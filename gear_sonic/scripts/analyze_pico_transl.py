"""Analyze a pico_raw recording directory: check whether root translation
(`body_pos_w`) was actually captured, and summarize the trajectory.

Usage:
    python gear_sonic/scripts/analyze_pico_transl.py /path/to/pico_raw/20260928_161440
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def load_session(session_dir: str):
    files = sorted(glob.glob(os.path.join(session_dir, "pose_*.npz")))
    if not files:
        raise FileNotFoundError(f"no pose_*.npz found in {session_dir}")

    # Keys worth checking for liveness. Each npz holds a rolling buffer of
    # `num_frames_to_send` frames; the LAST entry is the newest frame.
    track_keys = [
        "smpl_pose",
        "smpl_joints",
        "body_quat_w",
        "joint_pos",
        "vr_position",
        "body_pos_w",
    ]
    acc: dict[str, list] = {k: [] for k in track_keys}
    ts, idx = [], []
    present = set()
    for f in files:
        d = np.load(f, allow_pickle=True)
        present |= set(d.files)
        for k in track_keys:
            if k in d.files:
                a = d[k]
                acc[k].append(a[-1] if a.ndim > 1 else a)
        ts.append(float(d["timestamp_monotonic"][0]))
        idx.append(int(d["frame_index"][-1]))

    out = {
        "files": files,
        "t": np.asarray(ts),
        "frame_index": np.asarray(idx),
        "has_root": "body_pos_w" in present,
    }
    for k in track_keys:
        out[k] = np.asarray(acc[k]) if acc[k] else None
    out["joints"] = out["smpl_joints"]
    return out


def liveness_report(s) -> bool:
    """Per-key TEMPORAL variation check. Returns True if the stream is live.

    This is the check that matters: a frozen SDK snapshot still yields a
    perfectly plausible-looking single pose, so only frame-to-frame deltas
    can distinguish "live capture" from "stale buffer repeated N times".
    """
    print("\n--- stream liveness (temporal variation) ---")
    print(f"{'key':14s} {'temporal std (max)':>18s} {'max |frame delta|':>18s} {'uniq frames':>12s}")
    all_frozen = True
    T = len(s["files"])
    for k in ["smpl_pose", "smpl_joints", "body_quat_w", "joint_pos",
              "vr_position", "body_pos_w"]:
        a = s.get(k)
        if a is None:
            print(f"{k:14s} {'MISSING':>18s}")
            continue
        a = np.asarray(a, dtype=np.float64).reshape(T, -1)
        tstd = a.std(axis=0).max()
        dmax = np.abs(np.diff(a, axis=0)).max() if T > 1 else 0.0
        uniq = len(np.unique(a, axis=0))
        flag = "" if uniq > 1 else "   <== FROZEN"
        if uniq > 1:
            all_frozen = False
        print(f"{k:14s} {tstd:18.6f} {dmax:18.6f} {uniq:12d}{flag}")

    if all_frozen:
        print("\n*** ALL STREAMS FROZEN: every frame is byte-identical. The "
              "XRoboToolkit service/headset was not feeding live data. ***")
    return not all_frozen


def analyze(session_dir: str):
    s = load_session(session_dir)
    T = len(s["files"])
    dur = s["t"][-1] - s["t"][0]
    print(f"session   : {session_dir}")
    print(f"frames    : {T}   duration: {dur:.1f}s   avg fps: {T / max(dur, 1e-6):.1f}")

    live = liveness_report(s)

    print("\n--- transl / root position check ---")
    if not s["has_root"]:
        print("body_pos_w MISSING -> recorded WITHOUT --record_root_pos; transl is lost.")
        return s

    p = s["body_pos_w"].astype(np.float64)
    rng = p.max(0) - p.min(0)
    disp = np.linalg.norm(p[-1] - p[0])
    path_len = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    print(f"body_pos_w shape : {p.shape}")
    print(f"first frame      : {np.round(p[0], 4)}")
    print(f"last  frame      : {np.round(p[-1], 4)}")
    print(f"mean             : {np.round(p.mean(0), 4)}")
    print(f"min              : {np.round(p.min(0), 4)}")
    print(f"max              : {np.round(p.max(0), 4)}")
    print(f"per-axis range   : {np.round(rng, 4)}  (x, y, z in Pico world frame)")
    print(f"per-axis std     : {np.round(p.std(0), 4)}")
    print(f"net displacement : {disp:.3f} m")
    print(f"path length      : {path_len:.3f} m")
    if T > 1:
        print(f"per-frame step   : mean {step.mean() * 1000:.2f} mm, "
              f"max {step.max() * 1000:.2f} mm")
        n_static = int((step < 1e-9).sum())
        print(f"static frames    : {n_static}/{T - 1} "
              f"({100.0 * n_static / max(T - 1, 1):.1f}% identical to previous)")

    if not live:
        verdict = "STREAM FROZEN -> recording unusable, re-capture."
    elif np.allclose(p, 0.0):
        verdict = "ALL ZERO -> transl NOT converted back (in-place clip)."
    elif np.allclose(p, p[0]):
        verdict = "CONSTANT -> transl present but frozen (no motion captured)."
    elif rng.max() < 0.05:
        verdict = f"near-constant (max range {rng.max():.3f} m) -> essentially in-place."
    else:
        verdict = "VARYING -> transl IS converted back and recorded correctly."
    print(f"\nVERDICT: {verdict}")

    print(f"\nnote: axis with smallest motion range = {int(np.argmin(rng))}; "
          f"per-axis std = {np.round(p.std(0), 4)}")

    print("\n--- smpl_joints (streamed) check ---")
    j = s["joints"].astype(np.float64)
    root_j = j[:, 0, :]
    print(f"smpl_joints shape      : {j.shape}")
    print(f"joint0 (pelvis) mean   : {np.round(root_j.mean(0), 4)}")
    print(f"joint0 range           : {np.round(root_j.max(0) - root_j.min(0), 4)}")
    print("-> smpl_joints are root-LOCAL (pelvis ~ fixed); world motion must come "
          "from body_pos_w.")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    analyze(ap.parse_args().session_dir)
