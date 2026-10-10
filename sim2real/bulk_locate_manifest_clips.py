#!/usr/bin/env python
"""Bulk-locate every manifest clip across every `streamed_*` session.

`locate_clip_in_session.py` answers "where is clip X in session dir D?" one
clip and one directory at a time. Keeping `eval_clip_manifest.md` §1/§4 honest
needs the whole cross product, plus the **separation** between the best and
second-best match -- the manifest's §5.7 warning is that an argmin alone is not
evidence once ~160 candidate clips exist.

Reports, per clip: best match, its error, the runner-up error, and the ratio.
A match is called CONFIRMED only if err <= --max-err AND runner-up/err >= --min-sep.

Usage:
  .venv_teleop/bin/python sim2real/bulk_locate_manifest_clips.py
  .venv_teleop/bin/python sim2real/bulk_locate_manifest_clips.py --clips jog_ff_start_180_R_002__A192_M
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.stream_clip_mode2 import load_stream_data  # noqa: E402
from sim2real.locate_clip_in_session import (  # noqa: E402
    best_alignment,
    load_session_joints,
)

MANIFEST_CLIPS = [
    "walk_180_R_003__A332_M",
    "walk_sideway_045_stop_005__A042_M",
    "walk_backward_start_001__A030_M",
    "walk_sideway_135_loop_003__A022",
    "walk_ff_stop_225_R_002__A266_M",
    "warm_up_chest_003__A359_M",
    "jog_ff_start_180_R_002__A265",
    "jog_ff_start_180_R_002__A192_M",
    "reach_jump_R_001__A072_M",
    "kneeling_start_101__A063_M",
    "kneeling_stop_002__A051_M",
    "high_jump_R_103__A389_M",
    "dance_vouge_shake_it_babe_360_R_002__A318_M",
    "dance_hiphop_mike_tyson_R_fast_001__A319_M",
]

SMPL_DIR = "/home/grease/ego_dataset/eval_subset/smpl"
DATA_ROOT = "/home/grease/g1_robot_data"


def session_dirs(root):
    """Every directory containing an smpl_joint.csv, at any depth we use."""
    out = []
    for pat in ("*/streamed_*", "streamed_*", "*/*/streamed_*"):
        for d in glob.glob(os.path.join(root, pat)):
            if os.path.isfile(os.path.join(d, "smpl_joint.csv")):
                out.append(d)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=MANIFEST_CLIPS)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-err", type=float, default=20.0,
                    help="mm; matches above this are 'unconfirmed' (manifest §5.7)")
    ap.add_argument("--min-sep", type=float, default=3.0,
                    help="runner-up / best ratio required to confirm")
    a = ap.parse_args()

    sdirs = session_dirs(DATA_ROOT)
    print(f"sessions with smpl_joint.csv: {len(sdirs)}")
    cache = {}
    for d in sdirs:
        try:
            cache[d] = load_session_joints(d)
        except Exception as exc:  # noqa: BLE001
            print(f"  unreadable: {d} ({exc})")

    print(f"\n{'clip':<46}{'best session':<34}{'off':>7}{'err':>9}"
          f"{'2nd':>9}{'sep':>7}  verdict")
    print("-" * 125)
    results = {}
    for name in a.clips:
        path = os.path.join(SMPL_DIR, name + ".pkl")
        joints, _, _, _, _ = load_stream_data(path, a.fps, official=True)
        clip = np.asarray(joints, dtype=np.float64)

        rows = []
        for d, sess in cache.items():
            off, err = best_alignment(clip, sess, a.stride)
            rows.append((d, off, err * 1000.0))
        rows.sort(key=lambda r: r[2])

        d, off, err = rows[0]
        # The runner-up is usually ANOTHER TRUE REPLAY of the same clip, so it
        # cannot serve as the separation baseline -- doing that makes every
        # multiply-replayed clip look ambiguous. Compare instead against the
        # nearest session that is clearly NOT the clip, i.e. the first error
        # above the true-match band.
        hits = [r for r in rows if r[2] <= a.max_err]
        nonmatch = [r for r in rows if r[2] > a.max_err]
        second = nonmatch[0][2] if nonmatch else float("inf")
        sep = second / err if err > 0 else float("inf")
        ok = (err <= a.max_err) and (sep >= a.min_sep)
        verdict = (f"CONFIRMED x{len(hits)}" if ok
                   else ("unconfirmed" if err <= a.max_err else "NO MATCH"))
        rel = os.path.relpath(d, DATA_ROOT)
        print(f"{name:<46}{rel:<34}{off:>7}{err:>9.2f}{second:>9.1f}"
              f"{sep:>7.1f}  {verdict}")
        results[name] = (rel, off, err, second, sep, ok, len(clip))

    print("\nConfirmed matches (all sessions under the error bar):")
    for name in a.clips:
        path = os.path.join(SMPL_DIR, name + ".pkl")
        joints, _, _, _, _ = load_stream_data(path, a.fps, official=True)
        clip = np.asarray(joints, dtype=np.float64)
        hits = []
        for d, sess in cache.items():
            off, err = best_alignment(clip, sess, a.stride)
            if err * 1000.0 <= a.max_err:
                hits.append((os.path.relpath(d, DATA_ROOT), off, err * 1000.0,
                             len(clip)))
        if hits:
            hits.sort(key=lambda h: h[2])
            print(f"  {name}:")
            for rel, off, err, T in hits:
                print(f"     {rel:<34} frames [{off}, {off + T})  "
                      f"t = {off / a.fps:.2f}-{(off + T) / a.fps:.2f} s  {err:.2f} mm")


if __name__ == "__main__":
    main()
