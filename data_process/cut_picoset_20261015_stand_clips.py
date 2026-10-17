"""Cut the 2026-10-15 PICO sessions (20261015_160754, 20261015_161659) into clips so that
  (1) every clip STARTS and ENDS standing (upright, pelvis at standing height, both feet on the floor, quiet), and
  (2) a crawl / crouch / jump (hop) episode is never split: it lies completely inside one clip, with a standing margin on both sides.

Method (Z-up world skeleton; body_pos_w is PICO Y-up -> (x, -z, y), floor = 5th percentile of the lowest foot):
  stand_pose   pelvis > STAND_H above the floor, torso within STAND_INCL deg of vertical, both feet < FOOT_LOW above the floor
  stand_ok     stand_pose and activity (chunk_pico_session.activity_signal) below the QPCT percentile of the standing frames
  candidates   the quietest frame of every run of stand_ok lasting >= MIN_QUIET s  -> the only places a cut may go
  protected    episodes that must not be split: low posture (pelvis < LOW_H or torso > LOW_INCL, >= 0.5 s) and real hops (both feet
               > HOP_FOOT above the floor, pelvis rising > HOP_RISE); episodes closer than MERGE_GAP s are merged, then each side is
               extended to the nearest candidate (so the episode starts/ends standing).  No cut is allowed inside a protected episode.
  segments     consecutive cut points; protected episodes become their own clip; too-short pieces are merged into a neighbour, pieces
               longer than MAX_LEN are split at the quietest candidate that leaves >= MIN_LEN on both sides.
  labels       event = crawl / jump_A, jump_B ... (crawl: pelvis < 0.5 m and torso > 50 deg; jump cluster: >= 3 hops, not during a crawl)
Every clip is verified to start and end in stand_pose (clip ends are trimmed to the first/last standing frame); the script prints the check.
Writes <session>_clips/clip_XXX.npz (same consolidated format as chunk_pico_session.py) and clips.csv (+ event, start_stand, end_stand,
protected); a previous <session>_clips is moved to <session>_clips_v1.

usage: python cut_picoset_20261015_stand_clips.py [--dry]
"""
import argparse
import csv
import importlib.util
import os
import shutil

import numpy as np

RAW = "/home/grease/g1_robot_data/pico_raw"
SESSIONS = ["20261015_160754", "20261015_161659"]
STAND_H, STAND_INCL, FOOT_LOW = 0.85, 20.0, 0.12
QPCT, MIN_QUIET = 40.0, 0.2
LOW_H, LOW_INCL = 0.80, 35.0
HOP_FOOT, HOP_RISE = 0.12, 0.08
MERGE_GAP = 4.0
EXTEND_MAX = 8.0
MIN_LEN, MAX_LEN = 3.0, 20.0
CRAWL_PELVIS, CRAWL_INCL = 0.5, 50.0
MIN_HOPS = 3

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
    return [(int(a), int(b)) for a, b in zip(s, e) if t[b - 1] - t[a] >= min_len]


def merge(iv, t, gap):
    out = []
    for a, b in sorted(iv):
        if out and t[a] - t[out[-1][1] - 1] <= gap:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def analyse(d):
    t = d["_t"]
    sj = d["smpl_joints"].astype(np.float64)
    quat = d["body_quat_w"].astype(np.float64)
    pos = yup_to_zup(d["body_pos_w"].astype(np.float64))
    q = np.repeat(quat[:, None, :], sj.shape[1], axis=1)
    world = qapply(q, sj) + pos[:, None, :]
    foot = world[:, [10, 11], 2]
    floor = float(np.percentile(foot.min(1), 5))
    pel = world[:, 0, 2] - floor
    foot_h = foot - floor
    spine = world[:, 12] - world[:, 0]
    incl = np.degrees(np.arccos(np.clip(spine[:, 2] / np.linalg.norm(spine, axis=1), -1, 1)))
    return dict(t=t, pel=pel, incl=incl, foot_h=foot_h, floor=floor)


def plan(d, a):
    t, pel, incl, foot_h = a["t"], a["pel"], a["incl"], a["foot_h"]
    T = len(t)
    act = cps.activity_signal(d)
    stand_pose = (pel > STAND_H) & (incl < STAND_INCL) & (foot_h.max(1) < FOOT_LOW)
    qthr = float(np.percentile(act[stand_pose], QPCT))
    stand_ok = stand_pose & (act < qthr)
    cand = np.array([lo + int(np.argmin(act[lo:hi])) for lo, hi in runs(stand_ok, t, MIN_QUIET)], dtype=int)
    low = (pel < LOW_H) | (incl > LOW_INCL)
    epis = [(lo, hi) for lo, hi in runs(low, t, 0.5)]
    hops = []
    for lo, hi in runs(foot_h.min(1) > HOP_FOOT, t, 0.1):
        pre = np.median(pel[max(lo - 50, 0) : lo]) if lo > 5 else np.median(pel)
        if pel[lo:hi].max() - pre > HOP_RISE:
            hops.append((lo, hi))
    epis += hops
    prot = merge(epis, t, MERGE_GAP)
    ext = []
    for lo, hi in prot:  # extend outwards to the nearest quiet standing frame on each side
        before = cand[cand < lo]
        after = cand[cand >= hi]
        a0 = int(before[-1]) if len(before) and t[lo] - t[before[-1]] <= EXTEND_MAX else lo
        b0 = int(after[0]) + 1 if len(after) and t[after[0]] - t[hi - 1] <= EXTEND_MAX else hi
        ext.append((a0, b0))
    prot2 = []
    for lo, hi in sorted(ext):
        if prot2 and lo < prot2[-1][1]:
            prot2[-1] = (prot2[-1][0], max(hi, prot2[-1][1]))
        else:
            prot2.append((lo, hi))
    return dict(act=act, stand_pose=stand_pose, stand_ok=stand_ok, cand=cand, prot=prot2, hops=hops, qthr=qthr, T=T)


def segment(p, t):
    T = p["T"]
    prot = p["prot"]
    dur = lambda a, b: t[b - 1] - t[a]  # noqa: E731
    inside = lambda i: any(lo < i < hi for lo, hi in prot)  # noqa: E731
    cuts = sorted({0, T} | {int(c) for c in p["cand"] if not inside(int(c))} | {lo for lo, _ in prot} | {hi for _, hi in prot})
    cuts = [c for c in cuts if 0 <= c < T] + [T]
    pieces = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] > 1]
    is_prot = lambda lo, hi: any(plo <= lo and hi <= phi for plo, phi in prot)  # noqa: E731
    final, buf = [], None
    for lo, hi in pieces:
        if is_prot(lo, hi):
            if buf:
                final.append((buf[0], buf[1], False))
                buf = None
            if final and final[-1][2] and final[-1][1] >= lo:  # continuation of the same protected episode
                final[-1] = (final[-1][0], hi, True)
            else:
                final.append((lo, hi, True))
            continue
        buf = (lo, hi) if buf is None else (buf[0], hi)
        if dur(*buf) >= MIN_LEN:
            final.append((buf[0], buf[1], False))
            buf = None
    if buf:
        if final and not final[-1][2]:
            final[-1] = (final[-1][0], buf[1], False)
        elif dur(*buf) >= MIN_LEN:
            final.append((buf[0], buf[1], False))

    def split(lo, hi):
        if dur(lo, hi) <= MAX_LEN:
            return [(lo, hi)]
        c = [int(x) for x in p["cand"] if lo < x < hi and dur(lo, int(x)) >= MIN_LEN and dur(int(x), hi) >= MIN_LEN]
        if not c:  # no quiet standing frame: fall back to the least active standing frame (still upright, feet down)
            c = [i for i in range(lo, hi) if p["stand_pose"][i] and dur(lo, i) >= MIN_LEN and dur(i, hi) >= MIN_LEN and not inside(i)]
        if not c:
            return [(lo, hi)]
        k = min(c, key=lambda x: p["act"][x])
        return split(lo, k) + split(k, hi)

    # a protected clip shorter than MIN_LEN is merged into a neighbour (the episode stays whole, only the standing margin grows)
    i = 0
    while i < len(final):
        lo, hi, pr = final[i]
        if pr and dur(lo, hi) < MIN_LEN and len(final) > 1:
            j = i + 1 if i + 1 < len(final) else i - 1
            a, b = sorted((i, j))
            final[a] = (final[a][0], final[b][1], True)
            del final[b]
            i = max(a - 1, 0)
            continue
        i += 1

    out = []
    for lo, hi, pr in final:
        out.extend([(lo, hi, True)] if pr else [(a, b, False) for a, b in split(lo, hi)])
    return [x for x in out if dur(x[0], x[1]) >= MIN_LEN]


def trim_to_stand(lo, hi, stand_pose):
    while lo < hi and not stand_pose[lo]:
        lo += 1
    while hi > lo and not stand_pose[hi - 1]:
        hi -= 1
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    for s in SESSIONS:
        sdir = f"{RAW}/{s}"
        out_dir = f"{sdir}_clips"
        d, _ = cps.load_session(sdir)
        a = analyse(d)
        t = a["t"]
        p = plan(d, a)
        T = len(t)
        print("=" * 100, f"\n{s}: {T} frames, {t[-1]:.0f} s, floor {a['floor']:+.2f}, stand frames {100 * p['stand_pose'].mean():.0f}%, quiet-stand candidates {len(p['cand'])} (act thr {p['qthr']:.2f})")
        crawl = merge(runs((a["pel"] < CRAWL_PELVIS) & (a["incl"] > CRAWL_INCL), t, 0.5), t, 6.0)
        clusters, cur = [], []
        for h in p["hops"]:
            if cur and t[h[0]] - t[cur[-1][1] - 1] <= MERGE_GAP:
                cur.append(h)
            else:
                if cur:
                    clusters.append(cur)
                cur = [h]
        if cur:
            clusters.append(cur)
        clusters = [c for c in clusters if len(c) >= MIN_HOPS and not any(c[0][0] < b and c[-1][1] > aa for aa, b in crawl)]
        events = [("crawl", lo, hi) for lo, hi in crawl] + [(f"jump_{'ABCDEFGH'[i]}", c[0][0], c[-1][1]) for i, c in enumerate(clusters)]
        for e in events:
            print(f"  event {e[0]:8s} {t[e[1]]:7.1f}-{t[e[2] - 1]:7.1f} s")
        print(f"  protected episodes (low posture / hops, merged, extended to standing): {len(p['prot'])}")
        clips = []
        for lo, hi, pr in segment(p, t):
            lo2, hi2 = trim_to_stand(lo, hi, p["stand_pose"])
            if t[hi2 - 1] - t[lo2] < (2.0 if pr else MIN_LEN):  # a protected episode (crawl/jump/crouch) is kept even if short
                continue
            ev = [n for n, elo, ehi in events if elo < hi2 and ehi > lo2]
            clips.append(dict(lo=lo2, hi=hi2, prot=pr, event="+".join(ev)))
        split_prot = []
        for plo, phi in p["prot"]:
            hit = [c for c in clips if plo < c["hi"] and phi > c["lo"]]
            if len(hit) != 1 or not (hit[0]["lo"] <= plo and hit[0]["hi"] >= phi):
                split_prot.append((round(float(t[plo]), 1), round(float(t[phi - 1]), 1), len(hit)))
        bad_ev = []
        for n, elo, ehi in events:
            hit = [c for c in clips if elo < c["hi"] and ehi > c["lo"]]
            if len(hit) != 1 or not (hit[0]["lo"] <= elo and hit[0]["hi"] >= ehi):
                bad_ev.append(n)
        s_ok = sum(bool(p["stand_pose"][c["lo"]]) for c in clips)
        e_ok = sum(bool(p["stand_pose"][c["hi"] - 1]) for c in clips)
        durs = np.array([t[c["hi"] - 1] - t[c["lo"]] for c in clips])
        print(f"  clips {len(clips)}  total {durs.sum() / 60:.1f} min  (min {durs.min():.1f} s, median {np.median(durs):.1f}, max {durs.max():.1f})   coverage {100 * durs.sum() / t[-1]:.0f}%")
        print(f"  start standing {s_ok}/{len(clips)}   end standing {e_ok}/{len(clips)}   labelled events split: {bad_ev or 'none'}   protected episodes split: {split_prot or 'none'}")
        for c in clips:
            if c["event"] or c["prot"]:
                print(f"    {'event ' + c['event'] if c['event'] else 'protected':18s} {t[c['lo']]:7.1f}-{t[c['hi'] - 1]:7.1f} s ({t[c['hi'] - 1] - t[c['lo']]:.1f} s)")
        if args.dry:
            continue
        if os.path.isdir(out_dir) and not os.path.isdir(out_dir + "_v1"):
            shutil.move(out_dir, out_dir + "_v1")
        os.makedirs(out_dir, exist_ok=True)
        for old in os.listdir(out_dir):
            if old.startswith("clip_") or old.endswith(".csv"):
                os.remove(os.path.join(out_dir, old))
        save_keys = [k for k in d if not k.startswith("_")]
        rows = []
        for i, c in enumerate(clips):
            lo, hi = c["lo"], c["hi"]
            name = f"clip_{i:03d}.npz"
            clip = {k: d[k][lo:hi] for k in save_keys}
            clip["clip_start_index"] = np.int64(lo)
            clip["clip_duration_s"] = np.float32(t[hi - 1] - t[lo])
            np.savez_compressed(os.path.join(out_dir, name), **clip)
            P = d["body_pos_w"][lo:hi].astype(np.float64)
            rows.append(dict(clip=name, start_frame=lo, end_frame=hi, frames=hi - lo, start_s=round(float(t[lo]), 2), dur_s=round(float(t[hi - 1] - t[lo]), 2),
                             path_m=round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), 2), disp_m=round(float(np.linalg.norm(P[-1] - P[0])), 2),
                             mean_act=round(float(p["act"][lo:hi].mean()), 3), event=c["event"], start_stand=int(p["stand_pose"][lo]), end_stand=int(p["stand_pose"][hi - 1]),
                             protected=int(c["prot"])))
        with open(os.path.join(out_dir, "clips.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {len(rows)} clips -> {out_dir}")


if __name__ == "__main__":
    main()
