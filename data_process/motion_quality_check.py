"""
Motion data quality check (QC) for AMASS / LAFAN1 SMPL sequences.

Runs a physical-plausibility + integrity pass over every sequence listed in
`motion_categories_full.csv`, adds QC metrics, and writes:

    motion_qc.csv          - per-sequence QC metrics + qc_pass flag
    motion_qc_report.txt    - human-readable summary (pass rates, worst offenders)

Reuses the forward-kinematics / loader from classify_motions.py so the geometry
is identical to the categorisation pass.

QC metrics (joints in meters, Z-up):
    n_frames        - sequence length
    has_nan         - any NaN/Inf in joints
    foot_skate      - mean horizontal speed of the *grounded* foot (m/s).
                      High => retargeting foot-sliding artefact.
    ground_pen      - how far the lowest foot dips BELOW the floor (m, +ve = bad)
    float_gap       - median lowest-foot height ABOVE floor (m). Large => floating
    max_jerk        - 99th-pct joint jerk magnitude (m/s^3). Spikes => sensor glitch
    max_speed       - max joint speed (m/s). Huge => teleport/frame glitch
    bone_cv         - mean coefficient-of-variation of key bone lengths (should ~0)
    static_flag     - whole clip is basically motionless
    outlier_flag    - feature-space outlier within its category (robust z-score)

qc_pass = passes ALL hard gates (integrity + physical limits + not static).
Outliers are reported but NOT auto-failed (they may be legit rare motions);
they're excluded from the *test* pool later by the split script if desired.

Usage:
    .venv-1/bin/python motion_quality_check.py
    .venv-1/bin/python motion_quality_check.py --limit 500   # quick sample
"""
import os
import sys
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_process"))
import classify_motions as C  # reuse FK, loader, joint groups, FPS

# ── Hard-gate thresholds (human-plausible limits) ───────────────────────────
MIN_FRAMES = 30          # < ~1 s @30fps: too short to evaluate
MAX_SPEED = 20.0         # m/s: above this is a teleport glitch
MAX_JERK = 4000.0        # m/s^3 (99th pct): above this is sensor spiking
MAX_FOOT_SKATE = 1.2     # m/s: grounded-foot sliding tolerance
MAX_GROUND_PEN = 0.25    # m: foot allowed below floor before flagged
MAX_FLOAT_GAP = 0.30     # m: whole-clip float above floor before flagged
MAX_BONE_CV = 0.15       # bone-length coefficient of variation
STATIC_ENERGY = 0.02     # m/s mean joint speed below which clip is "static"

# bone pairs (child, parent) used for length-consistency check
BONE_PAIRS = [(4, 1), (7, 4), (5, 2), (8, 5), (18, 16), (20, 18),
              (19, 17), (21, 19), (15, 12)]

GROUND_THRESH = 0.08     # m above floor considered "in contact"


def compute_qc(joints):
    """joints: (T,24,3) meters Z-up. Returns dict of QC metrics (no pass gate)."""
    T = joints.shape[0]
    out = dict(n_frames=T, has_nan=0, foot_skate=0.0, ground_pen=0.0,
               float_gap=0.0, max_jerk=0.0, max_speed=0.0, bone_cv=0.0,
               static_flag=0, mean_speed=0.0)
    if not np.isfinite(joints).all():
        out['has_nan'] = 1
        return out
    if T < 3:
        return out

    dt = 1.0 / C.FPS
    floor = np.percentile(joints[:, C.FOOT_JOINTS, 2], 5)
    z = joints[:, :, 2] - floor

    vel = np.diff(joints, axis=0) / dt            # (T-1,24,3)
    speed = np.linalg.norm(vel, axis=2)           # (T-1,24)
    out['max_speed'] = float(speed.max())
    out['mean_speed'] = float(speed.mean())

    if T >= 4:
        acc = np.diff(vel, axis=0) / dt
        jerk = np.diff(acc, axis=0) / dt
        out['max_jerk'] = float(np.percentile(np.linalg.norm(jerk, axis=2), 99))

    # foot-skate: horizontal speed of a foot while it is grounded
    skates = []
    for fj in C.FOOT_JOINTS:
        h = z[1:, fj]                              # align with vel frames
        grounded = h < GROUND_THRESH
        if grounded.sum() > 0:
            horiz = np.linalg.norm(vel[:, fj, :2], axis=1)
            skates.append(horiz[grounded].mean())
    out['foot_skate'] = float(np.mean(skates)) if skates else 0.0

    # ground penetration / floating
    low_foot = z[:, C.FOOT_JOINTS].min(axis=1)
    out['ground_pen'] = float(max(0.0, -np.percentile(low_foot, 1)))
    out['float_gap'] = float(max(0.0, np.median(low_foot)))

    # bone-length consistency (coefficient of variation over time)
    cvs = []
    for child, parent in BONE_PAIRS:
        L = np.linalg.norm(joints[:, child] - joints[:, parent], axis=1)
        m = L.mean()
        if m > 1e-6:
            cvs.append(L.std() / m)
    out['bone_cv'] = float(np.mean(cvs)) if cvs else 0.0

    out['static_flag'] = int(out['mean_speed'] < STATIC_ENERGY)
    return out


def hard_pass(q):
    """Apply hard gates -> bool qc_pass, plus list of failure reasons."""
    reasons = []
    if q['has_nan']:
        reasons.append('nan')
    if q['n_frames'] < MIN_FRAMES:
        reasons.append('too_short')
    if q['max_speed'] > MAX_SPEED:
        reasons.append('teleport')
    if q['max_jerk'] > MAX_JERK:
        reasons.append('jitter')
    if q['foot_skate'] > MAX_FOOT_SKATE:
        reasons.append('foot_skate')
    if q['ground_pen'] > MAX_GROUND_PEN:
        reasons.append('ground_pen')
    if q['float_gap'] > MAX_FLOAT_GAP:
        reasons.append('floating')
    if q['bone_cv'] > MAX_BONE_CV:
        reasons.append('bone_inconsistent')
    if q['static_flag']:
        reasons.append('static')
    return (len(reasons) == 0), reasons


def subject_id(source, path):
    """Best-effort subject/take-family id for leakage-safe splitting."""
    base = os.path.basename(path)
    if source == 'LAFAN1':
        # e.g. fightAndSports1_subject4.bvh -> subject4
        for tok in base.replace('.bvh', '').split('_'):
            if tok.startswith('subject'):
                return 'LAFAN1_' + tok
        return 'LAFAN1_' + base
    # AMASS: <dataset>/<subject>/<seq> -> dataset+subject folder
    parts = path.replace('\\', '/').split('/')
    try:
        idx = parts.index('extracted')
        return '/'.join(parts[idx + 1: idx + 3])   # dataset/subject
    except ValueError:
        return os.path.dirname(path)


def robust_z_outliers(values):
    """Return boolean outlier mask for EXTREME upper-tail values only.

    Uses median/MAD robust z-score and flags only the high side (|z|>6 and
    z>0), since for our QC metrics (skate, jerk, speed, bone_cv) *large*
    values are the suspicious ones. Low values are normal (calm motions).
    """
    v = np.asarray(values, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-9
    z = 0.6745 * (v - med) / mad
    return z > 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_csv', default='data_analysis/motion_categories/motion_categories_full.csv')
    ap.add_argument('--out_csv', default='data_analysis/motion_qc/motion_qc.csv')
    ap.add_argument('--report', default='data_analysis/motion_qc/motion_qc_report.txt')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.in_csv)))
    if args.limit:
        rows = rows[:args.limit]
    print(f"QC over {len(rows):,} sequences from {args.in_csv} ...")

    results = []
    for i, r in enumerate(rows):
        path = r['path']
        # CSV stored relpaths; make absolute if needed
        abspath = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        rec = dict(source=r['source'], path=path, category=r['category'])
        try:
            j = C.load_joints(abspath)
            if j is None:
                q = dict(n_frames=0, has_nan=1, foot_skate=0, ground_pen=0,
                         float_gap=0, max_jerk=0, max_speed=0, bone_cv=0,
                         static_flag=0, mean_speed=0)
            else:
                q = compute_qc(j)
        except Exception:
            q = dict(n_frames=0, has_nan=1, foot_skate=0, ground_pen=0,
                     float_gap=0, max_jerk=0, max_speed=0, bone_cv=0,
                     static_flag=0, mean_speed=0)
        passed, reasons = hard_pass(q)
        rec.update(q)
        rec['qc_pass'] = int(passed)
        rec['fail_reasons'] = ';'.join(reasons)
        rec['subject_id'] = subject_id(r['source'], path)
        results.append(rec)
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(rows)}")

    # ── feature-space outlier flag within each category ─────────────────────
    feat_keys = ['foot_skate', 'max_jerk', 'max_speed', 'bone_cv', 'mean_speed']
    by_cat = {}
    for idx, rec in enumerate(results):
        by_cat.setdefault(rec['category'], []).append(idx)
    for cat, idxs in by_cat.items():
        if len(idxs) < 20:
            for k in idxs:
                results[k]['outlier_flag'] = 0
            continue
        masks = []
        for fk in feat_keys:
            masks.append(robust_z_outliers([results[k][fk] for k in idxs]))
        combined = np.any(masks, axis=0)
        for j2, k in enumerate(idxs):
            results[k]['outlier_flag'] = int(combined[j2])

    # ── write CSV ───────────────────────────────────────────────────────────
    cols = ['source', 'path', 'category', 'subject_id', 'n_frames', 'has_nan',
            'foot_skate', 'ground_pen', 'float_gap', 'max_jerk', 'max_speed',
            'bone_cv', 'mean_speed', 'static_flag', 'outlier_flag',
            'qc_pass', 'fail_reasons']
    with open(args.out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rec in results:
            w.writerow({c: rec.get(c, '') for c in cols})

    # ── report ──────────────────────────────────────────────────────────────
    n = len(results)
    n_pass = sum(r['qc_pass'] for r in results)
    from collections import Counter
    reason_counts = Counter()
    for r in results:
        for rs in r['fail_reasons'].split(';'):
            if rs:
                reason_counts[rs] += 1
    cat_stats = {}
    for r in results:
        c = r['category']
        s = cat_stats.setdefault(c, [0, 0, 0])  # total, pass, outlier
        s[0] += 1
        s[1] += r['qc_pass']
        s[2] += r.get('outlier_flag', 0)

    lines = []
    lines.append("=" * 60)
    lines.append("MOTION DATA QUALITY REPORT")
    lines.append("=" * 60)
    lines.append(f"Total sequences        : {n:,}")
    lines.append(f"Passed QC (hard gates) : {n_pass:,}  ({100*n_pass/n:.1f}%)")
    lines.append(f"Failed QC              : {n-n_pass:,}  ({100*(n-n_pass)/n:.1f}%)")
    lines.append(f"Flagged outliers       : {sum(r.get('outlier_flag',0) for r in results):,}")
    lines.append("")
    lines.append("Failure reasons (a clip can have several):")
    for rs, c in reason_counts.most_common():
        lines.append(f"  {rs:20s} {c:6d}  ({100*c/n:4.1f}%)")
    lines.append("")
    lines.append("Per-category QC pass rate (clean pool available for split):")
    lines.append(f"  {'category':28s} {'total':>7s} {'pass':>7s} {'pass%':>7s} {'outlier':>8s}")
    for c, (t, p, o) in sorted(cat_stats.items()):
        lines.append(f"  {c:28s} {t:7d} {p:7d} {100*p/t:6.1f}% {o:8d}")
    lines.append("")
    # worst offenders per hard metric
    def worst(metric, k=8):
        srt = sorted(results, key=lambda r: r.get(metric, 0), reverse=True)[:k]
        return [(round(r.get(metric, 0), 2), r['category'], os.path.basename(r['path'])) for r in srt]
    for metric in ['foot_skate', 'max_jerk', 'max_speed', 'ground_pen', 'bone_cv']:
        lines.append(f"Worst {metric}:")
        for val, cat, name in worst(metric):
            lines.append(f"    {val:>10}  [{cat[:18]:18s}] {name[:45]}")
        lines.append("")
    lines.append("=" * 60)
    lines.append(f"Clean pool (qc_pass==1, outlier_flag==0) per category:")
    for c, (t, p, o) in sorted(cat_stats.items()):
        clean = sum(1 for r in results
                    if r['category'] == c and r['qc_pass'] and not r.get('outlier_flag', 0))
        flag = "  <-- BELOW 400!" if clean < 400 else ""
        lines.append(f"  {c:28s} {clean:6d}{flag}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print("\n" + report)
    with open(args.report, 'w') as fh:
        fh.write(report + "\n")
    print(f"\nSaved: {args.out_csv}  and  {args.report}")


if __name__ == "__main__":
    main()
