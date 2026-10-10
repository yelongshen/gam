#!/usr/bin/env python3
"""eval_runs_0922.py — one-clip-per-run pipeline for the 2026-09-22 session.

Each run in `g1_run_0922/` replays exactly ONE eval_subset motion clip, and the
corresponding `20260922/20260922/streamed_*` session holds the SMPL stream that
was sent to the robot for that replay. This script:

  1. IDENTIFIES the clip from the **SMPL side** (independent of any label):
     the session's `smpl_joint.csv` (24x3 @50 Hz, root-relative) is matched
     against all 160 `eval_subset/smpl/<clip>.pkl` -> `smpl_joints` references
     with an FFT sliding-window SSD search; best + runner-up margin reported.

  2. CONFIRMS it from the **robot side**: within the run's
     `motion_name == "streamed"` span, the executed `q` (29 dof @50 Hz) is
     matched against all 160 `eval_subset/robot/<clip>.pkl` -> `dof`
     (resampled 30 -> 50 Hz). The clip is "confirmed" only when the robot-side
     winner equals the SMPL-side winner.

  3. EVALUATES the located episode with the standard online metric suite
     (`online_eval_clips.evaluate`): mpjpe_l / mpjpe_pa / vel_dist / accel_dist
     vs the retargeted G1 reference, the cmd-vs-q control-loop diagnostics,
     the raw-SMPL diagnostics, and the robustness terms.

Runs and sessions are paired by wall-clock: a session whose start time falls
inside a run's [start, start+duration] window belongs to that run.

Usage:
    .venv_sim/bin/python sim2real/eval_runs_0922.py
"""
import argparse
import datetime as dt
import glob
import io
import json
import os
import sys
import zlib

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import online_eval_clips as OEC  # noqa: E402

FPS = 50.0
NUM_DOF = 29
Q_COL0 = 5          # q.csv / dq.csv: 5 leading bookkeeping columns


# ----------------------------------------------------------------- loading --

def load_pkl(path):
    """eval_subset pkls are zlib-compressed joblib dumps (plain joblib fallback)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        return joblib.load(io.BytesIO(zlib.decompress(raw)))
    except zlib.error:
        return joblib.load(io.BytesIO(raw))


def _unwrap(d, clip):
    return d[clip] if isinstance(d, dict) and clip in d else d


def robust_csv(path, ncols=None):
    rows = []
    with open(path) as fh:
        try:
            header = next(fh).rstrip("\n").split(",")
        except StopIteration:
            return [], np.zeros((0, 0))
        n = ncols or len(header)
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) != n:
                continue
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                continue
    return header, np.asarray(rows, dtype=np.float64)


def streamed_ranges(motion_name_csv):
    names = []
    with open(motion_name_csv) as fh:
        next(fh)
        for line in fh:
            names.append(line.rstrip("\n").split(",")[-1].strip('"'))
    out, start = [], None
    for i, n in enumerate(names):
        if n == "streamed" and start is None:
            start = i
        elif n != "streamed" and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(names)))
    return out


def resample_to_50hz(arr, fps):
    t_src = np.arange(len(arr)) / fps
    t_dst = np.arange(0, t_src[-1], 1.0 / FPS)
    flat = arr.reshape(len(arr), -1)
    out = np.stack([np.interp(t_dst, t_src, flat[:, c]) for c in range(flat.shape[1])], 1)
    return out.reshape(len(t_dst), *arr.shape[1:])


# ------------------------------------------------------- sliding-window SSD --

def sliding_ssd(run, ref):
    """SSD of `ref` (M,D) over every offset of `run` (N,D), via FFT.

    Returns (offsets_ssd,) with length N-M+1; ssd[o] = sum||run[o:o+M]-ref||^2.
    """
    n, d = run.shape
    m = ref.shape[0]
    if n < m:
        return None
    # windowed sum of squares of the run
    sq = np.concatenate([[0.0], np.cumsum((run ** 2).sum(1))])
    win = sq[m:] - sq[:-m]                       # (N-M+1,)
    # cross-correlation via FFT (linear conv with time-reversed ref)
    L = 1 << int(np.ceil(np.log2(n + m)))
    F = np.fft.rfft(run, n=L, axis=0) * np.conj(np.fft.rfft(ref, n=L, axis=0))
    corr = np.fft.irfft(F, n=L, axis=0)[:n - m + 1].sum(1)
    return win - 2 * corr + (ref ** 2).sum()


def motion_window(x, thresh):
    """[start, end) of the moving part of a (T,D) signal (frame-to-frame L2)."""
    mv = np.linalg.norm(np.diff(x.reshape(len(x), -1), axis=0), axis=1)
    idx = np.where(mv > thresh)[0]
    if len(idx) == 0:
        return 0, len(x)
    return int(idx.min()), int(idx.max()) + 2


def window_mask(ssd, m, win, pad=60, min_cover=0.8):
    """Reject offsets whose [off, off+m) window does not cover the motion window.

    Without this, long mostly-static sessions let a *static* clip win by fitting
    the settle/idle tail, which is exactly how `wiping_shoes` out-scored the real
    clip in the first pass.
    """
    ws, we = win
    offs = np.arange(len(ssd))
    ov = np.minimum(offs + m, we) - np.maximum(offs, ws)
    cover = np.clip(ov, 0, None) / max(min(m, we - ws), 1)
    ok = (cover >= min_cover) & (offs >= ws - pad) & (offs + m <= we + pad)
    if not ok.any():                      # clip longer/shorter than the window
        ok = cover >= min_cover
    out = np.full_like(ssd, np.inf)
    out[ok] = ssd[ok]
    return out


def best_matches(run, refs, topk=3, win=None):
    """refs: {name: (M,D)}. Returns topk [(name, offset, rms_per_coord)] sorted."""
    out = []
    for name, ref in refs.items():
        ssd = sliding_ssd(run, ref)
        if ssd is None:
            continue
        if win is not None:
            ssd = window_mask(ssd, len(ref), win)
        off = int(np.argmin(ssd))
        if not np.isfinite(ssd[off]):
            continue
        rms = float(np.sqrt(max(ssd[off], 0.0) / ref.size))
        out.append((name, off, rms))
    out.sort(key=lambda x: x[2])
    return out[:topk]


def _fft_corr(run, ref):
    """Sum over channels of the sliding cross-correlation; (N-M+1,)."""
    n, m = run.shape[0], ref.shape[0]
    L = 1 << int(np.ceil(np.log2(n + m)))
    F = np.fft.rfft(run, n=L, axis=0) * np.conj(np.fft.rfft(ref, n=L, axis=0))
    return np.fft.irfft(F, n=L, axis=0)[:n - m + 1].sum(1)


def sliding_ssd_yaw(run, ref):
    """Yaw-invariant sliding SSD for (N,24,3)/(M,24,3) joint clouds.

    The streamed SMPL is emitted in the *sender's* world frame, which differs
    from the reference clip's frame by an arbitrary constant rotation about z
    (verified: identical poses, ~90 deg apart). For each offset the optimal yaw
    has a closed form: with A = sum(sx*rx + sy*ry), B = sum(sy*rx - sx*ry) and
    C = sum(sz*rz), max_theta cross = sqrt(A^2 + B^2) + C at theta = atan2(B, A).

    Returns (ssd, yaw) arrays over offsets.
    """
    n, m = run.shape[0], ref.shape[0]
    if n < m:
        return None, None
    sq = np.concatenate([[0.0], np.cumsum((run ** 2).sum((1, 2)))])
    win = sq[m:] - sq[:-m]
    sx, sy, sz = run[..., 0], run[..., 1], run[..., 2]
    rx, ry, rz = ref[..., 0], ref[..., 1], ref[..., 2]
    A = _fft_corr(sx, rx) + _fft_corr(sy, ry)
    B = _fft_corr(sy, rx) - _fft_corr(sx, ry)
    C = _fft_corr(sz, rz)
    cross = np.sqrt(A ** 2 + B ** 2) + C
    return win - 2 * cross + (ref ** 2).sum(), np.arctan2(B, A)


def best_matches_yaw(run, refs, topk=3, win=None, len_tol=None):
    """As best_matches but yaw-invariant; returns [(name, off, rms, yaw_rad)].

    `len_tol=(lo,hi)` additionally keeps only clips whose length is within
    [lo,hi] x the motion-window length -- the session streams exactly one clip,
    so a candidate far shorter/longer than the moving part cannot be it.
    """
    out = []
    for name, ref in refs.items():
        ssd, yaw = sliding_ssd_yaw(run, ref)
        if ssd is None or yaw is None:
            continue
        if win is not None and len_tol is not None:
            ratio = len(ref) / max(win[1] - win[0], 1)
            if not (len_tol[0] <= ratio <= len_tol[1]):
                continue
        if win is not None:
            ssd = window_mask(ssd, len(ref), win)
        off = int(np.argmin(ssd))
        if not np.isfinite(ssd[off]):
            continue
        out.append((name, off, float(np.sqrt(max(ssd[off], 0.0) / ref.size)),
                    float(yaw[off])))
    out.sort(key=lambda x: x[2])
    return out[:topk]


# ------------------------------------------------------------- reference db --

def load_smpl_refs(smpl_dir):
    refs = {}
    for p in sorted(glob.glob(os.path.join(smpl_dir, "*.pkl"))):
        clip = os.path.basename(p)[:-4]
        d = _unwrap(load_pkl(p), clip)
        j = np.asarray(d["smpl_joints"], dtype=np.float64)
        fps = float(d.get("fps", 50.0))
        if abs(fps - FPS) > 1e-6:
            j = resample_to_50hz(j, fps)
        refs[clip] = j - j[:, [0]]                        # root-relative (M,24,3)
    return refs


def load_robot_refs(robot_dir):
    refs, meta = {}, {}
    for p in sorted(glob.glob(os.path.join(robot_dir, "*.pkl"))):
        clip = os.path.basename(p)[:-4]
        d = _unwrap(load_pkl(p), clip)
        dof = np.asarray(d["dof"], dtype=np.float64)
        fps = float(d.get("fps", 30.0))
        refs[clip] = resample_to_50hz(dof, fps) if abs(fps - FPS) > 1e-6 else dof
        meta[clip] = {"fps": fps, "frames": len(dof)}
    return refs, meta


# ------------------------------------------------------------------ pairing --

def parse_ts(name):
    """'20260922_083033' or 'streamed_083353' -> datetime (date from the run dirs)."""
    tail = name.split("_")[-1]
    return dt.datetime.strptime(tail, "%H%M%S")


def pair_runs_sessions(runs, sessions):
    """runs: [(name, dir, duration_s)]; sessions: [(name, dir)] -> {run_name: sess}."""
    pairs = {}
    for rname, rdir, dur in runs:
        t0 = parse_ts(rname)
        t1 = t0 + dt.timedelta(seconds=dur)
        cands = [s for s in sessions if t0 <= parse_ts(s[0]) <= t1]
        pairs[rname] = cands[0] if len(cands) >= 1 else None
        if len(cands) > 1:
            print(f"  [warn] run {rname} overlaps {len(cands)} sessions, using first")
    return pairs


# --------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/grease/g1_robot_data/g1_run_0922")
    ap.add_argument("--sessions",
                    default="/home/grease/g1_robot_data/20260922/20260922")
    ap.add_argument("--smpl-ref", default="/home/grease/ego_dataset/eval_subset/smpl")
    ap.add_argument("--robot-ref", default="/home/grease/ego_dataset/eval_subset/robot")
    ap.add_argument("--out", default="sim2real/online_eval_results_20260922.json")
    args = ap.parse_args()

    print("[refs] loading 160 SMPL + 160 robot references ...")
    smpl_refs = load_smpl_refs(args.smpl_ref)
    robot_refs, robot_meta = load_robot_refs(args.robot_ref)

    # --- enumerate runs ------------------------------------------------------
    runs = []
    for rdir in sorted(glob.glob(os.path.join(args.runs, "*"))):
        rname = os.path.basename(rdir)
        qcsv = os.path.join(rdir, "q.csv")
        if not os.path.isfile(qcsv):
            continue
        hdr, arr = robust_csv(qcsv)
        if len(arr) < 100:
            print(f"[skip] {rname}: only {len(arr)} logged frames (aborted run)")
            continue
        runs.append((rname, rdir, arr[:, hdr.index("time_ms")].max() / 1000.0))

    sessions = [(os.path.basename(s), s)
                for s in sorted(glob.glob(os.path.join(args.sessions, "streamed_*")))]
    print(f"[data] {len(runs)} usable runs, {len(sessions)} streamed sessions")
    pairs = pair_runs_sessions(runs, sessions)

    fk = OEC.FK()
    OEC.load_reference = lambda root, clip: _ref_for_eval(root, clip)  # zlib-aware
    episodes, results = [], []

    for rname, rdir, dur in runs:
        print("\n" + "=" * 92)
        sess = pairs.get(rname)
        print(f"run {rname}  ({dur:.1f} s)   session: {sess[0] if sess else 'NONE'}")
        print("=" * 92)

        # --- 1. identify from the SMPL stream --------------------------------
        smpl_top = None
        if sess:
            _, sj = robust_csv(os.path.join(sess[1], "smpl_joint.csv"), ncols=72)
            sj = sj.reshape(-1, 24, 3)
            stream = sj - sj[:, [0]]
            swin = motion_window(sj, 0.02)
            print(f"  stream motion window: frames [{swin[0]}:{swin[1]}] "
                  f"({(swin[1] - swin[0]) / FPS:.1f} s of {len(sj) / FPS:.1f} s)")
            smpl_top = best_matches_yaw(stream, smpl_refs, win=swin,
                                        len_tol=(0.6, 1.6))
            print("  SMPL-side clip ID (session smpl_joint.csv vs eval_subset/smpl, "
                  "yaw-invariant):")
            for i, (c, o, r, y) in enumerate(smpl_top):
                tag = "<== best" if i == 0 else ""
                print(f"    {r * 1000:8.2f} mm rms  off={o:5d}  "
                      f"yaw={np.degrees(y):+7.1f} deg  {c} {tag}")

        # --- 2. confirm from the executed robot state ------------------------
        hdr_q, q_full = robust_csv(os.path.join(rdir, "q.csv"))
        t_ms = q_full[:, hdr_q.index("time_ms")]
        q = q_full[:, Q_COL0:Q_COL0 + NUM_DOF]
        spans = streamed_ranges(os.path.join(rdir, "motion_name.csv"))
        if not spans:
            print("  [skip] no 'streamed' span in motion_name.csv")
            continue
        s0, s1 = max(spans, key=lambda s: s[1] - s[0])
        seg = q[s0:s1]
        rwin = motion_window(seg, 0.01)
        print(f"  streamed span: frames [{s0}:{s1}] "
              f"({(s1 - s0) / FPS:.1f} s at t={t_ms[s0] / 1000:.1f}s), "
              f"motion [{rwin[0]}:{rwin[1]}]")
        robot_top = best_matches(seg, robot_refs, win=rwin)
        print("  robot-side clip ID (executed q vs eval_subset/robot dof):")
        for i, (c, o, r) in enumerate(robot_top):
            tag = "<== best" if i == 0 else ""
            print(f"    {np.degrees(r):8.2f} deg rms  off={o:5d}  {c} {tag}")

        if not smpl_top:
            print("  [skip] no paired SMPL session -> cannot confirm clip")
            continue
        clip_s, off_s, rms_s, yaw_s = smpl_top[0]
        clip_r, off_r, rms_r = robot_top[0]
        confirmed = clip_s == clip_r
        margin_s = smpl_top[1][2] / rms_s if len(smpl_top) > 1 and rms_s > 0 else np.inf
        print(f"  --> SMPL says '{clip_s}' (x{margin_s:.2f} margin over runner-up, "
              f"yaw {np.degrees(yaw_s):+.1f} deg), robot says '{clip_r}'  ==> "
              f"{'CONFIRMED MATCH' if confirmed else 'MISMATCH'}")
        if not confirmed:
            print("  [skip] clip identity not confirmed; not computing metrics")
            continue

        # --- 3. evaluate ------------------------------------------------------
        # Undo the sender-frame yaw so the raw-SMPL diagnostics are frame-aligned.
        OEC.load_session_smpl = _yaw_corrected_loader(-yaw_s)
        nframes = len(smpl_refs[clip_s])          # clip length at 50 Hz
        a0 = s0 + off_r
        ep = dict(clip=clip_s, category="", run=rname, session=sess[0],
                  offset=off_s, nframes=nframes,
                  t0_ms=float(t_ms[a0]),
                  t1_ms=float(t_ms[min(a0 + nframes, len(t_ms) - 1)]),
                  smpl_rms_mm=rms_s * 1000, robot_rms_deg=float(np.degrees(rms_r)),
                  smpl_margin=float(margin_s))
        episodes.append(ep)
        res = OEC.evaluate(ep, fk, args.sessions, args.runs, args.robot_ref)
        res["match"] = {"smpl_rms_mm": ep["smpl_rms_mm"],
                        "robot_rms_deg": ep["robot_rms_deg"],
                        "smpl_margin": ep["smpl_margin"], "confirmed": True}
        results.append(res)
        print(OEC.fmt(res))

    # --- orphan sessions: streamed but with no (usable) robot log ------------
    used = {r["session"] for r in results}
    for sname, sdir in sessions:
        if sname in used:
            continue
        _, sj = robust_csv(os.path.join(sdir, "smpl_joint.csv"), ncols=72)
        if len(sj) == 0:
            continue
        sj = sj.reshape(-1, 24, 3)
        top = best_matches_yaw(sj - sj[:, [0]], smpl_refs,
                               win=motion_window(sj, 0.02), len_tol=(0.6, 1.6))
        c, o, r, y = top[0]
        print(f"\n[orphan session] {sname}: streamed '{c}' "
              f"({r * 1000:.1f} mm rms, off={o}) but no matching robot log "
              f"-> no metrics")

    if results:
        agg = OEC.aggregate(results)
        print("\n=== per-clip aggregate (primary imitation metric, mean +/- std) ===")
        for clip, a in agg.items():
            print(f"  {clip}  (n={a['n_episodes']}, "
                  f"non-fall {a['non_fall_rate'] * 100:.0f}%)")
            print(f"     mpjpe_l  {a['mpjpe_l'][0]:6.2f} +/- {a['mpjpe_l'][1]:5.2f} mm  "
                  f"  mpjpe_pa {a['mpjpe_pa'][0]:6.2f} +/- {a['mpjpe_pa'][1]:5.2f} mm")
            print(f"     vel_dist {a['vel_dist'][0]:6.2f} +/- {a['vel_dist'][1]:5.2f}   "
                  f"   joint err {a['joint_err_deg_mean'][0]:5.2f} +/- "
                  f"{a['joint_err_deg_mean'][1]:4.2f} deg")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"episodes": results, "aggregate": agg,
                       "episode_defs": episodes}, fh, indent=2, default=str)
        print(f"\n[saved] {args.out}")
    else:
        print("\n[!] no confirmed episodes")


def _yaw_corrected_loader(yaw):
    """OEC.load_session_smpl replacement: rotates the stream by `yaw` about z."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def loader(session_dir, off, nframes):
        _, arr = robust_csv(os.path.join(session_dir, "smpl_joint.csv"), ncols=72)
        return arr[off:off + nframes].reshape(-1, 24, 3) @ R.T

    return loader


def _ref_for_eval(robot_root, clip):
    d = _unwrap(load_pkl(os.path.join(robot_root, f"{clip}.pkl")), clip)
    return {
        "dof": np.asarray(d["dof"], dtype=np.float64),
        "root_rot": np.asarray(d["root_rot"], dtype=np.float64),
        "root_trans": np.asarray(d["root_trans_offset"], dtype=np.float64),
        "smpl_joints": np.asarray(d["smpl_joints"], dtype=np.float64),
        "fps": float(d.get("fps", 30.0)),
    }


if __name__ == "__main__":
    main()
