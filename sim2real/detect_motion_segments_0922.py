#!/usr/bin/env python3
"""detect_motion_segments_0922.py — Stage 1+2 for g1_run_0922:

Stage 1 (this is the new part): within each run's `motion_name=="streamed"`
span, detect actual motion sub-episodes by finding settle-then-move-then-
settle patterns in the executed joint velocity (dq) -- mirroring what
`stream_clip_mode2.py --settle 2.0` actually produces on the robot (a ~2s
near-static hold before each clip's first real frame, per
eval_clip_manifest.md §4). This avoids treating one long "streamed" span as
a single giant sliding-window search space, which is what produced the
poor/ambiguous alignments in the previous pass.

Stage 2: for each detected motion episode, align it against BOTH
walk_180_R_003__A332_M and walk_backward_start_001__A030_M (only these two,
per the current ask), and report which one it best matches (if either).
"""
import glob

import numpy as np

from locate_and_eval_20260922 import load_zlib_joblib, robust_load_csv, get_streamed_index_ranges


EVAL_ROBOT_DIR = "/home/grease/ego_dataset/eval_subset/robot"
TARGET_CLIPS = ["walk_180_R_003__A332_M", "walk_backward_start_001__A030_M"]
G1_RUNS_GLOB = "/home/grease/g1_robot_data/g1_run_0922/*"

SETTLE_VEL_THRESH = 0.05     # rad/s: below this counts as "settled/static"
MIN_SETTLE_FRAMES = 40       # 0.8s @ 50Hz: minimum static run to count as a settle boundary
MIN_EPISODE_FRAMES = 60      # 1.2s: ignore tiny blips


def resample_ref(dof, ref_fps):
    t_orig = np.arange(dof.shape[0]) / ref_fps
    t_50 = np.arange(0, t_orig[-1], 1.0 / 50.0)
    return np.stack([np.interp(t_50, t_orig, dof[:, j]) for j in range(dof.shape[1])], axis=1)


def detect_motion_episodes(dq_seg: np.ndarray):
    """dq_seg: (T, 29) joint velocities within one streamed span.
    Returns list of (start, end) frame indices (relative to dq_seg) where the
    robot is actually moving, bounded by settle (near-static) periods on
    both sides."""
    speed = np.max(np.abs(dq_seg), axis=1)  # (T,) -- max over joints per frame
    is_moving = speed > SETTLE_VEL_THRESH

    # Find contiguous "moving" runs.
    episodes = []
    i = 0
    T = len(is_moving)
    while i < T:
        if is_moving[i]:
            j = i
            while j < T and is_moving[j]:
                j += 1
            # extend backward/forward slightly to catch the ramp-in/out
            # (a real clip's very first/last frames may be slow enough to
            # dip under threshold right at the edges).
            episodes.append((i, j))
            i = j
        else:
            i += 1

    # Merge episodes that are separated by a settle gap shorter than
    # MIN_SETTLE_FRAMES (i.e. not a real settle-hold, just a brief slow
    # moment mid-motion).
    merged = []
    for ep in episodes:
        if merged and ep[0] - merged[-1][1] < MIN_SETTLE_FRAMES:
            merged[-1] = (merged[-1][0], ep[1])
        else:
            merged.append(list(ep))
    merged = [tuple(e) for e in merged if e[1] - e[0] >= MIN_EPISODE_FRAMES]
    return merged


def align_episode_to_clip(episode_q: np.ndarray, ref: np.ndarray, search_pad: int = 30):
    """episode_q: (Te, 29) executed joint angles for one detected episode.
    ref: (Tr, 29) reference dof, resampled to 50Hz.
    Since the episode should already correspond to (approximately) one full
    clip playback, only search a small +/-`search_pad`-frame window around
    a length-matched alignment (rather than a full unconstrained slide),
    padding/trimming as needed."""
    te, tr = episode_q.shape[0], ref.shape[0]

    if te >= tr:
        # episode is longer than (or equal to) the reference -- likely
        # includes some settle frames at the start/end; search within it.
        n_off = min(te - tr + 1, tr + 2 * search_pad)
        offs = range(0, te - tr + 1)
    else:
        # episode is shorter than reference -- can't fully contain it;
        # compare against the closest-length prefix of the reference
        # instead (partial-clip case).
        ref = ref[:te]
        tr = te
        offs = [0]

    best_off, best_err = None, np.inf
    for off in offs:
        window = episode_q[off:off + tr]
        err = np.mean(np.linalg.norm(window - ref, axis=-1))
        if err < best_err:
            best_off, best_err = off, err

    range_corr = np.corrcoef(
        ref.max(0) - ref.min(0),
        episode_q[best_off:best_off + tr].max(0) - episode_q[best_off:best_off + tr].min(0),
    )[0, 1]
    return best_off, best_err, range_corr, tr


def main():
    refs = {}
    for name in TARGET_CLIPS:
        d = load_zlib_joblib(f"{EVAL_ROBOT_DIR}/{name}.pkl")
        inner = d[name]
        refs[name] = resample_ref(inner["dof"], inner["fps"])
        print(f"ref '{name}': {refs[name].shape[0]} frames @ 50Hz "
              f"({refs[name].shape[0]/50.0:.2f}s)")

    run_dirs = sorted(d for d in glob.glob(G1_RUNS_GLOB) if d.split("/")[-1][0].isdigit())

    for run_dir in run_dirs:
        run_name = run_dir.split("/")[-1]
        try:
            q_full = robust_load_csv(f"{run_dir}/q.csv", 34)
            dq_full = robust_load_csv(f"{run_dir}/dq.csv", 34)
        except Exception as e:
            print(f"\n[skip run] {run_name}: {e}")
            continue
        if q_full.shape[0] == 0 or dq_full.shape[0] == 0:
            print(f"\n[skip run] {run_name}: empty q/dq")
            continue
        q, dq = q_full[:, 5:5 + 29], dq_full[:, 5:5 + 29]

        streamed_ranges = get_streamed_index_ranges(f"{run_dir}/motion_name.csv")
        print(f"\n{'='*90}\nrun {run_name}: streamed span(s) {streamed_ranges}")

        for seg_start, seg_end in streamed_ranges:
            dq_seg = dq[seg_start:seg_end]
            q_seg = q[seg_start:seg_end]
            episodes = detect_motion_episodes(dq_seg)
            print(f"  streamed span [{seg_start}:{seg_end}] ({(seg_end-seg_start)/50.0:.1f}s) "
                  f"-> {len(episodes)} motion episode(s) detected:")

            for k, (e_start, e_end) in enumerate(episodes):
                abs_start, abs_end = seg_start + e_start, seg_start + e_end
                ep_q = q_seg[e_start:e_end]
                dur = (e_end - e_start) / 50.0
                print(f"    episode {k}: frames[{abs_start}:{abs_end}] duration={dur:.2f}s")

                best_clip, best_err, best_corr, best_off = None, np.inf, None, None
                results = []
                for name, ref in refs.items():
                    off, err, corr, matched_len = align_episode_to_clip(ep_q, ref)
                    results.append((name, off, err, corr, matched_len))
                    if err < best_err:
                        best_clip, best_err, best_corr, best_off = name, err, corr, off

                for name, off, err, corr, matched_len in results:
                    marker = " <== BEST" if name == best_clip else ""
                    print(f"      vs '{name}': off={off} matched_len={matched_len} "
                          f"err={err:.3f} rad range_corr={corr:.3f}{marker}")

                confidence = "HIGH" if best_corr is not None and best_corr > 0.85 and best_err < 0.4 else \
                             "MEDIUM" if best_corr is not None and best_corr > 0.7 else "LOW"
                print(f"      => best guess: '{best_clip}' (confidence={confidence})")


if __name__ == "__main__":
    main()
