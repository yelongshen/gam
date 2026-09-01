"""
Auto-detect a few representative "concrete action" clips (walking, jumping,
waving, etc.) from a large, mostly-static continuous PICO SMPL streaming
capture (e.g. `logs/smpl_raw_real_robot/`, `reuben_testclip_0/`).

Implements the plan discussed for cooking a long raw stream down into a
handful of labeled motion segments:

  1. Load the full sequence (frame 0 of each 4-frame `pose_*.npz` chunk,
     same convention as `visualize_pico.py`).
  2. Compute a SLIDING-WINDOW version of `classify_motions.extract_features()`
     (root_speed, limb_energy, airborne_frac, upper_lower_ratio,
     pelvis_low_frac) -- reusing the exact same feature definitions already
     validated all session for AMASS/LAFAN1 classification, just applied
     per-window instead of per-whole-clip.
  3. Threshold `limb_energy` to split the stream into "active" vs.
     "static/idle" frames, merge nearby active frames into segments (with a
     gap-tolerance to bridge brief pauses), and drop segments that are too
     short (noise) or too long (need splitting).
  4. Classify each segment with `classify_motions.classify()` (coarse
     category), then apply extra periodicity/heuristic checks for a finer
     action label (walking vs jumping vs waving vs static).
  5. Print (and optionally save) the frame-range + label for each detected
     clip -- this script does NOT yet export .pkl clips, just detects and
     reports ranges (see --export flag for a follow-up).

Usage:
  .venv_sim/bin/python data_process/detect_action_clips.py --dir reuben_testclip_0
  .venv_sim/bin/python data_process/detect_action_clips.py --dir logs/smpl_raw_real_robot \
      --window_s 1.5 --energy_pct 60 --min_dur_s 1.0 --max_dur_s 8.0
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling imports
import classify_motions as C  # reuse LOWER_JOINTS/UPPER_JOINTS/FOOT_JOINTS/PELVIS


def load_sequence(d):
    """Load frame-0-of-each-chunk smpl_joints + fps, same convention as
    visualize_pico.py's load_sequence() (but only what we need here)."""
    files = sorted(glob.glob(os.path.join(d, "pose_*.npz")))
    if not files:
        raise SystemExit(f"no pose_*.npz in {d}")
    joints = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        joints.append(z['smpl_joints'][0])  # (24,3), first of the 4-frame chunk
    fps = float(np.load(files[0])['pico_fps'][0])
    return np.asarray(joints), fps, len(files)


def sliding_window_features(joints, fps, window_s):
    """Compute classify_motions.extract_features()-style signals, but per
    SLIDING WINDOW (stride = 1 frame) instead of once for a whole clip, so
    we get a continuous per-frame TIME SERIES of each feature to threshold
    against.

    Returns a dict of (T,) arrays, one value per frame (using a centered
    window; edges are padded by clamping to the nearest valid window)."""
    T = len(joints)
    win = max(4, int(round(window_s * fps)))
    half = win // 2
    dt = 1.0 / fps

    limb_energy = np.zeros(T)
    root_speed = np.zeros(T)
    airborne_frac = np.zeros(T)
    upper_lower_ratio = np.zeros(T)
    pelvis_low_frac = np.zeros(T)

    # Global floor estimate (5th percentile of foot heights across the WHOLE
    # stream) -- more robust than per-window floor for a long, mostly-static
    # capture where a short window might have zero foot-height variation.
    floor = np.percentile(joints[:, C.FOOT_JOINTS, 2], 5)

    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        seg = joints[lo:hi]
        if len(seg) < 4:
            continue

        pelvis = seg[:, C.PELVIS]
        vel = np.diff(seg, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt if len(vel) > 1 else np.zeros((1, 24, 3))

        limb_energy[t] = float(np.linalg.norm(acc, axis=2).mean()) if len(acc) else 0.0

        root_vel = np.diff(pelvis, axis=0) / dt
        root_speed[t] = (float(np.percentile(np.linalg.norm(root_vel[:, :2], axis=1), 75))
                         if len(root_vel) else 0.0)

        z = seg[:, :, 2] - floor
        feet_h = z[:, C.FOOT_JOINTS].min(axis=1)
        airborne_frac[t] = float((feet_h > 0.15).mean())
        pelvis_low_frac[t] = float((z[:, C.PELVIS] < 0.55).mean())

        upper_e = np.linalg.norm(vel[:, C.UPPER_JOINTS], axis=2).mean() if len(vel) else 0.0
        lower_e = np.linalg.norm(vel[:, C.LOWER_JOINTS], axis=2).mean() + 1e-6 if len(vel) else 1e-6
        upper_lower_ratio[t] = float(upper_e / lower_e)

    return dict(limb_energy=limb_energy, root_speed=root_speed,
                airborne_frac=airborne_frac, upper_lower_ratio=upper_lower_ratio,
                pelvis_low_frac=pelvis_low_frac)


def segment_active_spans(active_mask, fps, gap_s, min_dur_s, max_dur_s):
    """Merge contiguous (with gap-tolerance) True runs in `active_mask` into
    (start, end) frame-index spans, dropping too-short spans and splitting
    too-long ones into equal-ish chunks capped at max_dur_s."""
    T = len(active_mask)
    gap_frames = max(1, int(round(gap_s * fps)))
    min_frames = max(1, int(round(min_dur_s * fps)))
    max_frames = max(min_frames, int(round(max_dur_s * fps)))

    # Merge active runs, bridging gaps <= gap_frames.
    spans = []
    i = 0
    while i < T:
        if not active_mask[i]:
            i += 1
            continue
        start = i
        j = i
        while j < T:
            if active_mask[j]:
                j += 1
                continue
            # look ahead across the gap
            k = j
            while k < T and not active_mask[k] and (k - j) < gap_frames:
                k += 1
            if k < T and active_mask[k]:
                j = k
                continue
            break
        end = j
        spans.append((start, end))
        i = end

    # Drop too-short, split too-long.
    out = []
    for start, end in spans:
        dur = end - start
        if dur < min_frames:
            continue
        if dur <= max_frames:
            out.append((start, end))
        else:
            n_chunks = int(np.ceil(dur / max_frames))
            chunk = int(np.ceil(dur / n_chunks))
            for c0 in range(start, end, chunk):
                c1 = min(end, c0 + chunk)
                if (c1 - c0) >= min_frames:
                    out.append((c0, c1))
    return out


def _periodicity_score(signal, fps, band_hz=(0.4, 3.0)):
    """Autocorrelation-based periodicity strength in a target frequency band
    (used to distinguish rhythmic motion like walking/waving from a single
    isolated event like a jump). Returns the peak normalized autocorrelation
    value within the lag range corresponding to band_hz."""
    signal = signal - signal.mean()
    if np.allclose(signal, 0) or len(signal) < 8:
        return 0.0
    acf = np.correlate(signal, signal, mode='full')
    acf = acf[len(acf) // 2:]
    acf = acf / (acf[0] + 1e-9)
    lag_lo = max(1, int(round(fps / band_hz[1])))
    lag_hi = min(len(acf) - 1, int(round(fps / band_hz[0])))
    if lag_hi <= lag_lo:
        return 0.0
    return float(np.max(acf[lag_lo:lag_hi]))


def label_segment(joints, feats, start, end, fps):
    """Coarse category via classify_motions.classify(), refined into a
    concrete action label using periodicity + simple heuristics."""
    seg_joints = joints[start:end]
    feat_dict = C.extract_features(seg_joints)
    coarse = C.classify(feat_dict) if feat_dict else "Uncategorized"

    mean_root_speed = float(feats['root_speed'][start:end].mean())
    mean_airborne = float(feats['airborne_frac'][start:end].mean())
    mean_ulr = float(feats['upper_lower_ratio'][start:end].mean())

    # Foot-height periodicity (gait signal) and wrist-position periodicity
    # (waving signal), each checked in the walking/waving-relevant band.
    foot_h = joints[start:end, C.FOOT_JOINTS, 2].mean(axis=1)
    gait_periodicity = _periodicity_score(foot_h, fps, band_hz=(0.5, 2.5))
    wrist_x = joints[start:end, 20:22, 0].mean(axis=1)  # wrists, horizontal
    wave_periodicity = _periodicity_score(wrist_x, fps, band_hz=(0.5, 3.0))

    if mean_airborne > 0.08:
        label = "jumping"
    elif mean_root_speed > 0.5 and gait_periodicity > 0.3:
        label = "walking"
    elif mean_ulr > 1.8 and mean_root_speed < 0.3 and wave_periodicity > 0.25:
        label = "waving"
    elif coarse == "Basic Locomotion":
        label = "walking"
    elif coarse == "Agility / High-Dynamic":
        label = "agile"
    elif coarse == "Upper-Body Manipulation":
        label = "gesture"
    elif coarse == "Unstructured / OOD":
        label = "unstructured"
    else:
        label = "misc"

    return label, coarse, dict(
        mean_root_speed=mean_root_speed, mean_airborne=mean_airborne,
        mean_ulr=mean_ulr, gait_periodicity=gait_periodicity,
        wave_periodicity=wave_periodicity)


def snap_to_base(start, end, base, n_total):
    """Snap a (start, end) frame range to clean multiples of `base` -- floor
    `start` DOWN to the nearest multiple, ceil `end` UP to the nearest
    multiple (so the snapped range always fully covers the original one),
    clamped to [0, n_total]."""
    snapped_start = (start // base) * base
    snapped_end = int(np.ceil(end / base)) * base
    return max(0, snapped_start), min(n_total, snapped_end)


def merge_overlapping(spans):
    """Merge any (start, end) ranges that now overlap/touch after snapping
    (snapping can push previously-separate segments into contact)."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='directory of pose_*.npz files')
    ap.add_argument('--window_s', type=float, default=1.5, help='sliding window size (s)')
    ap.add_argument('--energy_pct', type=float, default=60.0,
                     help='percentile of limb_energy above which a frame is "active"')
    ap.add_argument('--gap_s', type=float, default=0.3, help='bridge gaps <= this (s)')
    ap.add_argument('--min_dur_s', type=float, default=1.0, help='drop segments shorter than this')
    ap.add_argument('--max_dur_s', type=float, default=8.0, help='split segments longer than this')
    ap.add_argument('--top_n_per_label', type=int, default=3,
                     help='keep at most N clips per detected label')
    ap.add_argument('--frame_base', type=int, default=100,
                     help='snap output clip start/end frame indices to multiples of this '
                          '(e.g. 100 -> ranges like [100,200), [400,1200))')
    args = ap.parse_args()

    print(f"Loading {args.dir} ...")
    joints, fps, n = load_sequence(args.dir)
    print(f"  {n} frames @ {fps:.1f} fps  ({n / fps:.1f}s)  joints={joints.shape}")

    print(f"Computing sliding-window features (window={args.window_s}s) ...")
    feats = sliding_window_features(joints, fps, args.window_s)

    threshold = np.percentile(feats['limb_energy'], args.energy_pct)
    active = feats['limb_energy'] > threshold
    print(f"  limb_energy threshold (p{args.energy_pct:.0f}): {threshold:.4f}  "
          f"({active.mean() * 100:.1f}% of frames flagged active)")

    spans = segment_active_spans(active, fps, args.gap_s, args.min_dur_s, args.max_dur_s)
    print(f"\nFound {len(spans)} candidate segments before dedup/ranking:")

    results = []
    for start, end in spans:
        label, coarse, extra = label_segment(joints, feats, start, end, fps)
        dur_s = (end - start) / fps
        score = feats['limb_energy'][start:end].mean()  # quality/rank proxy
        results.append(dict(start=start, end=end, dur_s=dur_s, label=label,
                             coarse=coarse, score=score, **extra))

    # Rank + keep top_n_per_label per detected label.
    from collections import defaultdict
    by_label = defaultdict(list)
    for r in results:
        by_label[r['label']].append(r)

    selected = []
    for label, items in by_label.items():
        items.sort(key=lambda r: -r['score'])
        selected.extend(items[:args.top_n_per_label])
    selected.sort(key=lambda r: r['start'])

    # Snap each selected clip's frame range to clean multiples of
    # --frame_base (e.g. base=100 -> [100,200), [400,1200), ...), so
    # downstream tooling gets round, human-friendly boundaries instead of
    # the raw detection-derived ones.
    for r in selected:
        r['start'], r['end'] = snap_to_base(r['start'], r['end'], args.frame_base, n)
        r['dur_s'] = (r['end'] - r['start']) / fps

    print(f"\n{'=' * 90}")
    print(f"SELECTED CLIPS (top {args.top_n_per_label} per label, {len(selected)} total, "
          f"frame_base={args.frame_base}):")
    print(f"{'=' * 90}")
    print(f"{'frame_range':>18s}  {'dur(s)':>7s}  {'label':12s} {'coarse_category':24s} "
          f"{'root_v':>7s} {'air%':>6s} {'ulr':>5s} {'gait_p':>7s} {'wave_p':>7s}")
    for r in selected:
        print(f"[{r['start']:6d},{r['end']:6d})  {r['dur_s']:7.2f}  {r['label']:12s} "
              f"{r['coarse']:24s} {r['mean_root_speed']:7.3f} {r['mean_airborne'] * 100:6.1f} "
              f"{r['mean_ulr']:5.2f} {r['gait_periodicity']:7.3f} {r['wave_periodicity']:7.3f}")
    print(f"{'=' * 90}")

    print(f"\nAll {len(results)} candidate segments (before top-N filtering):")
    for r in results:
        print(f"  [{r['start']:6d},{r['end']:6d})  dur={r['dur_s']:5.2f}s  "
              f"label={r['label']:12s}  coarse={r['coarse']}")


if __name__ == '__main__':
    main()
