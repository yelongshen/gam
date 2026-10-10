"""Merge consecutive clips from a chunked PICO session into a single clip.

Only adjacent clips (clip N's end_frame == clip N+1's start_frame) can be
merged losslessly; the script verifies contiguity before concatenating and
refuses to merge across a gap unless --force is given.

Usage:
    python gear_sonic/scripts/merge_pico_clips.py <clips_dir> --clips 005 006
    python gear_sonic/scripts/merge_pico_clips.py <clips_dir> --clips 005 006 --out clip_005_006.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# Scalar/metadata keys that must not be concatenated along time.
META_KEYS = {"clip_start_index", "clip_duration_s"}


def merge(clips_dir: str, ids: list[str], out_name: str | None, force: bool):
    paths = [os.path.join(clips_dir, f"clip_{c}.npz") for c in ids]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    loaded = [np.load(p, allow_pickle=True) for p in paths]

    # --- contiguity check -------------------------------------------------
    print("contiguity check:")
    ok = True
    for i in range(len(loaded) - 1):
        a, b = loaded[i], loaded[i + 1]
        a_start = int(a["clip_start_index"])
        a_end = a_start + len(a["smpl_joints"])
        b_start = int(b["clip_start_index"])
        gap = b_start - a_end
        status = "adjacent" if gap == 0 else f"GAP of {gap} frames"
        print(f"  clip_{ids[i]} ends @{a_end}  ->  clip_{ids[i+1]} starts @{b_start}   [{status}]")
        if gap != 0:
            ok = False
    if not ok and not force:
        raise SystemExit("non-contiguous clips; pass --force to merge anyway")

    # --- concatenate ------------------------------------------------------
    keys = [k for k in loaded[0].files if k not in META_KEYS]
    merged = {}
    for k in keys:
        merged[k] = np.concatenate([d[k] for d in loaded], axis=0)

    merged["clip_start_index"] = np.int64(int(loaded[0]["clip_start_index"]))
    t = merged["timestamp_monotonic"].reshape(-1).astype(np.float64)
    dur = float(t[-1] - t[0])
    merged["clip_duration_s"] = np.float32(dur)
    merged["merged_from"] = np.array([f"clip_{c}" for c in ids])

    out_name = out_name or f"clip_{ids[0]}_{ids[-1]}_merged.npz"
    out_path = os.path.join(clips_dir, out_name)
    np.savez_compressed(out_path, **merged)

    n = len(merged["smpl_joints"])
    print(f"\nmerged {len(ids)} clips -> {n} frames, {dur:.2f}s")
    print(f"wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dir")
    ap.add_argument("--clips", nargs="+", required=True, help="clip ids in order, e.g. 005 006")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="merge even if not contiguous")
    a = ap.parse_args()
    merge(a.clips_dir, a.clips, a.out, a.force)


if __name__ == "__main__":
    main()
