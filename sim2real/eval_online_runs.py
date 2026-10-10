#!/usr/bin/env python3
"""eval_online_runs.py — generalized online clip evaluation for the sessions whose
robot logs and stream logs cannot be aligned by wall clock.

`eval_runs_0922.py` pairs a run with its stream session by timestamp. That only
works for `g1_run_0919` / `g1_run_0922`, whose run directories are named
`<YYYYMMDD>_<HHMMSS>` in the *streaming PC's* clock. For `g1_run_0909`, `0912`
and `0915` the run directories are named `g1_deploy_run_<MMDDYYYY>_run<N>` and
their only absolute clock (`time_realtime_ms`) is the robot's, which is ~15 h
away from the stream session names -- so wall-clock pairing is meaningless.

This script therefore pairs by **content**:

  1. Every `streamed_*` session is identified independently from its SMPL
     stream (yaw-invariant FFT sliding SSD vs all `eval_subset/smpl` clips,
     restricted to the detected motion window, with a clip-length plausibility
     filter) -- see `eval_runs_0922.py` for why each of those is needed.
  2. Every `"streamed"` span in every run is identified independently from the
     executed `q` (FFT sliding SSD vs all `eval_subset/robot` `dof` refs).
  3. A run span and a session are PAIRED when they name the *same* clip. When
     several candidates exist, they are matched greedily in chronological order
     (runs by `time_realtime_ms`, sessions by name), each session used once.
  4. Paired + agreeing episodes are scored with `online_eval_clips.evaluate`.

Usage:
    .venv_sim/bin/python sim2real/eval_online_runs.py \
        --runs /home/grease/g1_robot_data/g1_run_0909 \
        --sessions /home/grease/g1_robot_data/20260909 \
        --out sim2real/online_eval_results_20260909.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import online_eval_clips as OEC  # noqa: E402
from eval_runs_0922 import (  # noqa: E402
    FPS, NUM_DOF, Q_COL0, _ref_for_eval, _yaw_corrected_loader, best_matches,
    best_matches_yaw, load_robot_refs, load_smpl_refs, motion_window,
    robust_csv, sliding_ssd, streamed_ranges, window_mask,
)

MIN_MOTION_S = 1.5        # ignore streamed spans with less than this much motion
MAX_RMS_DEG = 12.0        # robot-side identification must be at least this good
MAX_RMS_MM = 90.0         # SMPL-side identification must be at least this good


def find_sessions(sessions_root):
    """`<root>/streamed_*` or the nested `<root>/<date>/streamed_*` layout."""
    s = sorted(glob.glob(os.path.join(sessions_root, "streamed_*")))
    if s:
        return sessions_root, s
    for sub in sorted(glob.glob(os.path.join(sessions_root, "*"))):
        s = sorted(glob.glob(os.path.join(sub, "streamed_*")))
        if s:
            return sub, s
    return sessions_root, []


def identify_sessions(session_dirs, smpl_refs):
    """[(name, dir, clip, off, rms_mm, margin, yaw)] in chronological order."""
    out = []
    for sdir in session_dirs:
        name = os.path.basename(sdir)
        path = os.path.join(sdir, "smpl_joint.csv")
        if not os.path.isfile(path):
            continue
        _, arr = robust_csv(path, ncols=72)
        if len(arr) < 50:
            print(f"  [session {name}] only {len(arr)} frames -- skipped")
            continue
        sj = arr.reshape(-1, 24, 3)
        win = motion_window(sj, 0.02)
        if (win[1] - win[0]) < MIN_MOTION_S * FPS:
            print(f"  [session {name}] motion window {(win[1]-win[0])/FPS:.1f}s "
                  f"-- too short, skipped")
            continue
        top = best_matches_yaw(sj - sj[:, [0]], smpl_refs, win=win,
                               len_tol=(0.6, 1.6))
        if not top:
            print(f"  [session {name}] no plausible clip")
            continue
        clip, off, rms, yaw = top[0]
        margin = top[1][2] / rms if len(top) > 1 and rms > 0 else np.inf
        flag = "" if rms * 1000 <= MAX_RMS_MM else "  (WEAK)"
        print(f"  [session {name}] {clip}   {rms * 1000:6.1f} mm  x{margin:.2f}  "
              f"off={off}  yaw={np.degrees(yaw):+.1f} deg{flag}")
        out.append(dict(name=name, dir=sdir, clip=clip, off=off,
                        rms_mm=rms * 1000, margin=float(margin), yaw=float(yaw),
                        used=False))
    return out


def identify_runs(run_dirs, robot_refs):
    """[(run, span, clip, off, rms_deg, margin, t_ms)] sorted by robot clock."""
    out = []
    for rdir in run_dirs:
        rname = os.path.basename(rdir)
        qcsv = os.path.join(rdir, "q.csv")
        if not os.path.isfile(qcsv):
            continue
        hdr, arr = robust_csv(qcsv)
        if len(arr) < 100:
            print(f"  [run {rname}] {len(arr)} frames -- aborted, skipped")
            continue
        t_ms = arr[:, hdr.index("time_ms")]
        t0_epoch = (arr[0, hdr.index("time_realtime_ms")]
                    if "time_realtime_ms" in hdr else 0.0)
        q = arr[:, Q_COL0:Q_COL0 + NUM_DOF]
        mn = os.path.join(rdir, "motion_name.csv")
        spans = streamed_ranges(mn) if os.path.isfile(mn) else [(0, len(q))]
        for s0, s1 in spans:
            seg = q[s0:min(s1, len(q))]
            if len(seg) < MIN_MOTION_S * FPS:
                continue
            win = motion_window(seg, 0.01)
            if (win[1] - win[0]) < MIN_MOTION_S * FPS:
                continue
            top = best_matches(seg, robot_refs, win=win)
            if not top:
                continue
            clip, off, rms = top[0]
            margin = top[1][2] / rms if len(top) > 1 and rms > 0 else np.inf
            rms_deg = float(np.degrees(rms))
            flag = "" if rms_deg <= MAX_RMS_DEG else "  (WEAK)"
            print(f"  [run {rname}] span [{s0}:{s1}] -> {clip}   "
                  f"{rms_deg:5.2f} deg  x{margin:.2f}  off={off}{flag}")
            out.append(dict(run=rname, dir=rdir, span=(s0, s1), clip=clip,
                            off=off, rms_deg=rms_deg, margin=float(margin),
                            t_ms=t_ms, q=q, s0=s0, epoch=float(t0_epoch),
                            used=False))
    out.sort(key=lambda r: (r["epoch"], r["s0"]))
    return out


def pair_by_clip(run_eps, sessions):
    """Pair run spans with sessions.

    Pass 1 -- *agreement*: both sides independently name the same clip. This is
    the strong evidence case and it is matched greedily in chronological order.

    Pass 2 -- *order*: whatever is left is zipped chronologically. This exists
    because requiring agreement silently deletes the most interesting episodes:
    when the robot fails to track a clip (e.g. the 09-12 `high_jump_R_103`
    reps), the executed `q` no longer resembles the reference, so the
    robot-side vote wanders off to a generic clip with a ~1.15 margin (i.e. no
    discrimination). Those episodes are still evaluated, but flagged
    `confirmed=False` / `pairing="order"`, and the SMPL-identified clip's rank
    on the robot side is reported so the weakness is visible.
    """
    pairs = []
    for ep in run_eps:
        cands = [s for s in sessions if not s["used"] and s["clip"] == ep["clip"]]
        if cands:
            cands[0]["used"] = True
            ep["used"] = True
            pairs.append((ep, cands[0], "agreement"))
    left_eps = [e for e in run_eps if not e["used"]]
    left_sess = [s for s in sessions if not s["used"]]
    for ep, s in zip(left_eps, left_sess):
        s["used"] = True
        ep["used"] = True
        pairs.append((ep, s, "order"))
    for ep in run_eps:
        if not ep["used"]:
            pairs.append((ep, None, "none"))
    pairs.sort(key=lambda p: (p[0]["epoch"], p[0]["s0"]))
    return pairs


def locate_clip_in_span(seg, ref):
    """Best offset of a *given* reference inside a run span (+ its rms, deg)."""
    win = motion_window(seg, 0.01)
    ssd = sliding_ssd(seg, ref)
    if ssd is None:
        return None, np.inf
    ssd = window_mask(ssd, len(ref), win)
    off = int(np.argmin(ssd))
    if not np.isfinite(ssd[off]):
        return None, np.inf
    return off, float(np.degrees(np.sqrt(max(ssd[off], 0.0) / ref.size)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--smpl-ref", default="/home/grease/ego_dataset/eval_subset/smpl")
    ap.add_argument("--robot-ref", default="/home/grease/ego_dataset/eval_subset/robot")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("[refs] loading SMPL + robot reference clips ...")
    smpl_refs = load_smpl_refs(args.smpl_ref)
    robot_refs, _ = load_robot_refs(args.robot_ref)

    sess_root, session_dirs = find_sessions(args.sessions)
    run_dirs = sorted(d for d in glob.glob(os.path.join(args.runs, "*"))
                      if os.path.isdir(d))
    print(f"[data] {len(run_dirs)} run dirs, {len(session_dirs)} stream sessions\n")

    print("=== SMPL-side identification (what was streamed) ===")
    sessions = identify_sessions(session_dirs, smpl_refs)
    print("\n=== robot-side identification (what was executed) ===")
    run_eps = identify_runs(run_dirs, robot_refs)

    print("\n=== pairing (same clip on both sides, chronological) ===")
    fk = OEC.FK()
    OEC.load_reference = _ref_for_eval
    results, skipped = [], []

    for ep, sess, how in pair_by_clip(run_eps, sessions):
        tag = f"{ep['run']} span[{ep['span'][0]}:{ep['span'][1]}]"
        if sess is None:
            print(f"  [unpaired] {tag}: robot says '{ep['clip']}' "
                  f"({ep['rms_deg']:.2f} deg) but no stream session is left")
            skipped.append({**{k: ep[k] for k in ("run", "clip", "rms_deg")},
                            "reason": "no stream session to pair with"})
            continue
        if sess["rms_mm"] > MAX_RMS_MM:
            print(f"  [weak] {tag} / {sess['name']}: SMPL id "
                  f"'{sess['clip']}' only {sess['rms_mm']:.1f} mm -- skipped")
            skipped.append({"run": ep["run"], "session": sess["name"],
                            "clip": sess["clip"], "smpl_rms_mm": sess["rms_mm"],
                            "reason": "SMPL identification below bar"})
            continue

        clip = sess["clip"]          # the stream is the authority on identity
        if how == "agreement":
            off, rms_deg = ep["off"], ep["rms_deg"]
            print(f"  [CONFIRMED] {tag} / {sess['name']}: '{clip}' "
                  f"(robot {rms_deg:.2f} deg x{ep['margin']:.2f}, "
                  f"SMPL {sess['rms_mm']:.1f} mm x{sess['margin']:.2f})")
        else:
            # Robot side voted differently -> re-locate the streamed clip inside
            # the span and report how badly the executed motion diverged.
            seg = ep["q"][ep["span"][0]:ep["span"][1]]
            off, rms_deg = locate_clip_in_span(seg, robot_refs[clip])
            if off is None:
                skipped.append({"run": ep["run"], "session": sess["name"],
                                "clip": clip, "reason": "clip does not fit span"})
                print(f"  [skip] {tag} / {sess['name']}: '{clip}' does not fit")
                continue
            print(f"  [order-paired, NOT confirmed] {tag} / {sess['name']}: "
                  f"streamed '{clip}' (SMPL {sess['rms_mm']:.1f} mm "
                  f"x{sess['margin']:.2f}); executed motion instead votes "
                  f"'{ep['clip']}' ({ep['rms_deg']:.2f} deg x{ep['margin']:.2f}) "
                  f"-- streamed clip fits at {rms_deg:.2f} deg")

        OEC.load_session_smpl = _yaw_corrected_loader(-sess["yaw"])
        n = len(smpl_refs[clip])
        a0 = ep["span"][0] + off
        t_ms = ep["t_ms"]
        edef = dict(clip=clip, category="", run=ep["run"],
                    session=sess["name"], offset=sess["off"], nframes=n,
                    t0_ms=float(t_ms[min(a0, len(t_ms) - 1)]),
                    t1_ms=float(t_ms[min(a0 + n, len(t_ms) - 1)]))
        try:
            res = OEC.evaluate(edef, fk, sess_root, args.runs, args.robot_ref)
        except Exception as exc:                                  # noqa: BLE001
            print(f"      [error] {exc}")
            skipped.append({"run": ep["run"], "clip": clip,
                            "reason": f"evaluate failed: {exc}"})
            continue
        res["match"] = {"pairing": how, "confirmed": how == "agreement",
                        "robot_rms_deg": rms_deg, "robot_vote": ep["clip"],
                        "robot_vote_rms_deg": ep["rms_deg"],
                        "robot_margin": ep["margin"],
                        "smpl_rms_mm": sess["rms_mm"], "smpl_margin": sess["margin"],
                        "yaw_deg": float(np.degrees(sess["yaw"]))}
        res["episode_def"] = edef
        results.append(res)

    for r in results:
        print(OEC.fmt(r))

    confirmed = [r for r in results if r["match"]["confirmed"]]
    agg = OEC.aggregate(confirmed) if confirmed else {}
    if agg:
        print("\n=== per-clip aggregate, CONFIRMED episodes only "
              "(primary imitation metric, mean +/- std) ===")
        for clip, a in agg.items():
            print(f"  {clip}  (n={a['n_episodes']}, "
                  f"non-fall {a['non_fall_rate'] * 100:.0f}%)")
            print(f"     mpjpe_l  {a['mpjpe_l'][0]:6.2f} +/- {a['mpjpe_l'][1]:5.2f} mm  "
                  f"  mpjpe_pa {a['mpjpe_pa'][0]:6.2f} +/- {a['mpjpe_pa'][1]:5.2f} mm")
            print(f"     vel_dist {a['vel_dist'][0]:6.2f} +/- {a['vel_dist'][1]:5.2f}   "
                  f"   joint err {a['joint_err_deg_mean'][0]:5.2f} +/- "
                  f"{a['joint_err_deg_mean'][1]:4.2f} deg")

    unused = [s["name"] for s in sessions if not s["used"]]
    if unused:
        print(f"\n[orphan sessions] {len(unused)} streamed with no matching run span: "
              + ", ".join(unused))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"episodes": results, "aggregate": agg, "skipped": skipped,
                   "sessions": [{k: v for k, v in s.items() if k != "dir"}
                                for s in sessions],
                   "orphan_sessions": unused}, fh, indent=2, default=str)
    print(f"\n[saved] {args.out}  ({len(results)} episodes, "
          f"{len(confirmed)} confirmed)")


if __name__ == "__main__":
    main()
