"""Cut the 2026-10-15 PICO sessions (20261015_160754, 20261015_161659) into clips and tag the special events.

1. Default chunking = gear_sonic/scripts/chunk_pico_session.py (motion-activity segmentation, rest cuts, max 20 s).
2. Special events are detected from the skeleton in the Z-up world frame (body_pos_w is PICO Y-up -> (x, -z, y)):
     crawl         pelvis < CRAWL_PELVIS m and torso inclined > CRAWL_INCL deg from vertical (runs merged if closer than CRAWL_MERGE s)
     jump cluster  both feet > JUMP_FOOT m above the floor with the pelvis rising > JUMP_RISE m above its standing level;
                   hops closer than JUMP_MERGE s are one cluster (at least 2 hops)
   Each event is padded by PAD s and cut as its OWN clip, tagged in `event`; every default clip that overlaps an event is replaced by
   [remainder before, event clip, remainder after] (remainders shorter than MIN_LEN are dropped), so no frame is in two clips.
3. Output per session (like chunk_pico_session.py): <session>_clips/clip_XXX.npz + clips.csv (extra column `event`) and events.csv.

usage: python cut_picoset_20261015_clips.py [--dry]
"""
import argparse
import csv
import importlib.util
import os

import numpy as np

RAW = "/home/grease/g1_robot_data/pico_raw"
SESSIONS = ["20261015_160754", "20261015_161659"]
CRAWL_PELVIS, CRAWL_INCL, CRAWL_MERGE = 0.5, 50.0, 6.0
JUMP_FOOT, JUMP_RISE, JUMP_MERGE, MIN_HOPS = 0.12, 0.08, 4.0, 3
PAD = 2.0
MIN_LEN, MAX_LEN = 3.0, 20.0
MAX_EVENT_LEN = 60.0

spec = importlib.util.spec_from_file_location("chunk_pico_session", "/home/grease/gam/gear_sonic/scripts/chunk_pico_session.py")
cps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cps)


def yup_to_zup(p):
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=-1)


def qapply(q, v):
    w = q[..., 0:1]
    u = q[..., 1:]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def runs(mask, t, min_len):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(a, b) for a, b in zip(s, e) if t[b - 1] - t[a] >= min_len]


def merge(ivals, t, gap):
    out = []
    for a, b in ivals:
        if out and t[a] - t[out[-1][1] - 1] <= gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def detect_events(d):
    """d from load_session (newest frame of each file).  Returns [(kind, lo, hi, info)] with frame indices into d."""
    t = d["_t"]
    sj = d["smpl_joints"].astype(np.float64)
    quat = d["body_quat_w"].astype(np.float64)
    pos = yup_to_zup(d["body_pos_w"].astype(np.float64))
    q = np.repeat(quat[:, None, :], sj.shape[1], axis=1)
    world = qapply(q, sj) + pos[:, None, :]
    foot = world[:, [10, 11], 2]
    floor = float(np.percentile(foot.min(1), 5))
    pel = world[:, 0, 2] - floor
    low_foot = foot.min(1) - floor
    spine = world[:, 12] - world[:, 0]
    incl = np.degrees(np.arccos(np.clip(spine[:, 2] / np.linalg.norm(spine, axis=1), -1, 1)))
    events = []
    crawl = merge(runs((pel < CRAWL_PELVIS) & (incl > CRAWL_INCL), t, 0.5), t, CRAWL_MERGE)
    for a, b in crawl:
        events.append(("crawl", a, b, f"pelvis {pel[a:b].min():.2f}-{np.median(pel[a:b]):.2f} m, torso incl median {np.median(incl[a:b]):.0f} deg"))
    hops = []
    for a, b in runs(low_foot > JUMP_FOOT, t, 0.1):
        pre = np.median(pel[max(a - 50, 0) : a]) if a > 5 else np.median(pel)
        rise = pel[a:b].max() - pre
        if rise > JUMP_RISE:
            hops.append((a, b))
    clusters = [c for c in _cluster(hops, t, JUMP_MERGE) if len(c) >= MIN_HOPS]
    # feet leaving the floor while crawling is not a jump: drop clusters that overlap a crawl event
    clusters = [c for c in clusters if not any(c[0][0] < b and c[-1][1] > a for a, b in crawl)]
    for i, c in enumerate(clusters):
        a, b = c[0][0], c[-1][1]
        events.append((f"jump_{'AB'[i] if i < 2 else i}", a, b, f"{len(c)} hops, peak lowest-foot {max(low_foot[x:y].max() for x, y in c):.2f} m"))
    return events, floor, pel, incl


def _cluster(hops, t, gap):
    out = []
    for h in hops:
        if out and t[h[0]] - t[out[-1][-1][1] - 1] <= gap:
            out[-1].append(h)
        else:
            out.append([h])
    return out


def pad(lo, hi, t, T):
    return int(np.searchsorted(t, t[lo] - PAD)), min(int(np.searchsorted(t, t[hi - 1] + PAD)) + 1, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    for s in SESSIONS:
        sdir = f"{RAW}/{s}"
        out_dir = f"{sdir}_clips"
        d, _ = cps.load_session(sdir)
        t = d["_t"]
        T = len(t)
        act = cps.activity_signal(d)
        cuts, thr = cps.find_rest_cuts(act, t, 15.0, 0.8)
        bounds = [0] + cuts + [T]
        segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
        base = []
        for sg in segs:
            base.extend(cps.split_long(sg, t, act, MAX_LEN, MIN_LEN))
        base = [x for x in base if t[x[1] - 1] - t[x[0]] >= MIN_LEN]
        events, floor, pel, incl = detect_events(d)
        print("=" * 90, f"\n{s}: {T} frames, {t[-1]:.0f} s, floor {floor:+.2f} m, default clips {len(base)}")
        ev_iv = []
        for kind, a, b, info in events:
            lo, hi = pad(a, b, t, T)
            if t[hi - 1] - t[lo] > MAX_EVENT_LEN:
                print(f"  WARNING {kind} {t[lo]:.1f}-{t[hi - 1]:.1f} s longer than {MAX_EVENT_LEN} s")
            ev_iv.append((kind, lo, hi, info))
            print(f"  event {kind:8s} {t[a]:7.1f}-{t[b - 1]:7.1f} s  (clip {t[lo]:.1f}-{t[hi - 1]:.1f} s, {t[hi - 1] - t[lo]:.1f} s)   {info}")
        # carve: event clips are first-class; remaining base clips are trimmed around them
        final = []  # (lo, hi, event)
        covered = [(lo, hi) for _, lo, hi, _ in ev_iv]
        for lo, hi in base:
            parts = [(lo, hi)]
            for elo, ehi in covered:
                nxt = []
                for a, b in parts:
                    if ehi <= a or elo >= b:
                        nxt.append((a, b))
                    else:
                        if elo > a:
                            nxt.append((a, elo))
                        if ehi < b:
                            nxt.append((ehi, b))
                parts = nxt
            for a, b in parts:
                if t[b - 1] - t[a] >= MIN_LEN:
                    final.append((a, b, ""))
        for kind, lo, hi, info in ev_iv:
            final.append((lo, hi, kind))
        final.sort()
        print(f"  clips after carving: {len(final)}  ({sum(1 for f in final if f[2])} event clips)")
        if args.dry:
            continue
        os.makedirs(out_dir, exist_ok=True)
        for old in os.listdir(out_dir):
            if old.startswith("clip_") and old.endswith(".npz"):
                os.remove(os.path.join(out_dir, old))
        save_keys = [k for k in d if not k.startswith("_")]
        rows = []
        for i, (lo, hi, ev) in enumerate(final):
            name = f"clip_{i:03d}.npz"
            dur = float(t[hi - 1] - t[lo])
            clip = {k: d[k][lo:hi] for k in save_keys}
            clip["clip_start_index"] = np.int64(lo)
            clip["clip_duration_s"] = np.float32(dur)
            np.savez_compressed(os.path.join(out_dir, name), **clip)
            P = d["body_pos_w"][lo:hi].astype(np.float64)
            rows.append(dict(clip=name, start_frame=lo, end_frame=hi, frames=hi - lo, start_s=round(float(t[lo]), 2), dur_s=round(dur, 2),
                             path_m=round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), 2), disp_m=round(float(np.linalg.norm(P[-1] - P[0])), 2),
                             mean_act=round(float(act[lo:hi].mean()), 3), event=ev))
        with open(os.path.join(out_dir, "clips.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {len(rows)} clips -> {out_dir}")


if __name__ == "__main__":
    main()
