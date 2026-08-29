"""
Train/test split for the categorized + QC'd motion dataset.

Supports three leakage-control policies:

  --leakage sequence   (policy B, DEFAULT)
        Sample the test set at the sequence level, then remove from TRAIN any
        clip that is a NEAR-DUPLICATE of a test clip (same subject_id + same
        category + very similar length/kinematic signature). Keeps most of the
        training data while preventing the same/near-same motion from leaking.

  --leakage subject    (policy A, strictest)
        No subject_id may appear in both TRAIN and TEST. Safest but very costly
        for AMASS (one subject owns hundreds of clips).

  --leakage source     (policy C)
        Hold out whole sub-datasets (e.g. SFU, HumanEva) for TEST.

Test pool is restricted to clean clips: qc_pass==1 AND outlier_flag==0
(unless --allow_outliers_in_test).

OOD handling:
  --ood_test_only      Put clean OOD clips into TEST (capped at --per_category)
                       and exclude OOD from TRAIN entirely. Recommended because
                       OOD is tiny (~401 clips) and is meant to measure
                       generalization, not to be trained on.

Outputs:
    split_test.csv, split_train.csv, split_summary.txt

Usage:
    .venv_sim/bin/python split_train_test.py                       # policy B, 400/cat
    .venv_sim/bin/python split_train_test.py --ood_test_only
    .venv_sim/bin/python split_train_test.py --leakage subject
"""
import csv
import argparse
import random
from collections import defaultdict, Counter

import numpy as np

CATEGORIES = [
    "Basic Locomotion",
    "Agility / High-Dynamic",
    "Upper-Body Manipulation",
    "Unstructured / OOD",
]
OOD = "Unstructured / OOD"

# features (from motion_qc.csv) used to build a near-duplicate signature
SIG_KEYS = ['n_frames', 'mean_speed', 'max_speed', 'max_jerk',
            'foot_skate', 'float_gap']


def robust_scale(rows):
    """Return (median, mad) per SIG_KEY for normalizing signatures."""
    scale = {}
    for k in SIG_KEYS:
        v = np.array([float(r[k]) for r in rows], dtype=float)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-6
        scale[k] = (med, mad)
    return scale


def signature(r, scale):
    return np.array([(float(r[k]) - scale[k][0]) / scale[k][1] for k in SIG_KEYS])


# ── environment-support motions ────────────────────────────────────────────
# These rely on props/geometry that do not exist in the flat-ground MuJoCo
# scene (chairs, stairs, handrails, tables, treadmills, ...). The robot has
# nothing to sit on or climb, so the reference is physically untrackable and
# any tracking error measured on such a clip is meaningless.
ENV_SUPPORT_PAT = re.compile(
    r'(sit(?:ting|down|_|\b)|chair|stool|sofa|bench|seat|couch'
    r'|stair|upstair|downstair|escalator|ladder|climb'
    r'|handrail|railing|banister'
    r'|table|desk|counter|shelf|cupboard|drawer'
    r'|lean|treadmill|bicycl|bike|cycling'
    r'|door)', re.I)
# 'sit' is a substring of these ordinary words - don't let them trigger
ENV_SUPPORT_FALSE = re.compile(r'(position|transit|visit|opposite|deposit)', re.I)


def needs_env_support(path):
    name = os.path.basename(path)
    return bool(ENV_SUPPORT_PAT.search(name)) and not ENV_SUPPORT_FALSE.search(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qc_csv', default='motion_qc.csv')
    ap.add_argument('--per_category', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--leakage', choices=['sequence', 'subject', 'source'],
                    default='sequence')
    ap.add_argument('--dup_tol', type=float, default=0.6,
                    help='near-dup L2 distance threshold in scaled feature space '
                         '(smaller = stricter; sequence mode only)')
    ap.add_argument('--dup_frame_tol', type=float, default=0.05,
                    help='relative frame-count difference to still count as dup')
    ap.add_argument('--ood_test_only', action='store_true')
    ap.add_argument('--keep_env_support', action='store_true',
                    help='keep motions that need chairs/stairs/handrails/etc. '
                         '(excluded by default: the flat-ground sim has no such '
                         'geometry, so those references are untrackable)')
    ap.add_argument('--allow_outliers_in_test', action='store_true')
    ap.add_argument('--include_failed_in_train', action='store_true')
    ap.add_argument('--test_out', default='split_test.csv')
    ap.add_argument('--train_out', default='split_train.csv')
    ap.add_argument('--summary', default='split_summary.txt')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.qc_csv)))
    for r in rows:
        r['qc_pass'] = int(r['qc_pass'])
        r['outlier_flag'] = int(r.get('outlier_flag', 0) or 0)

    # drop environment-support motions from BOTH splits unless asked to keep
    n_env = 0
    if not args.keep_env_support:
        before = len(rows)
        rows = [r for r in rows if not needs_env_support(r['path'])]
        n_env = before - len(rows)
        print(f"[filter] removed {n_env} environment-support clips "
              f"(chairs/stairs/handrails/tables/treadmills/...)")
    random.seed(args.seed)
    scale = robust_scale(rows)

    def test_eligible(r):
        if r['category'] not in CATEGORIES:
            return False
        if not r['qc_pass']:
            return False
        if r['outlier_flag'] and not args.allow_outliers_in_test:
            return False
        return True

    # ── choose TEST set ─────────────────────────────────────────────────────
    test_ids = set()
    test_rows = []
    warnings = []

    if args.leakage == 'source':
        by_cat_src = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if test_eligible(r):
                src = r['subject_id'].split('/')[0]  # dataset name
                by_cat_src[r['category']][src].append(r)
        for c in CATEGORIES:
            srcs = list(by_cat_src[c].keys())
            random.shuffle(srcs)
            picked = []
            for s in srcs:
                if len(picked) >= args.per_category:
                    break
                picked.extend(by_cat_src[c][s])
            picked = picked[:args.per_category]
            if len(picked) < args.per_category:
                warnings.append(f"{c}: only {len(picked)} clips (source mode)")
            for r in picked:
                test_ids.add(id(r)); test_rows.append(r)
    else:
        # sequence & subject modes: sample sequences per category
        by_cat = defaultdict(list)
        for r in rows:
            if test_eligible(r):
                by_cat[r['category']].append(r)
        for c in CATEGORIES:
            pool = by_cat[c]
            random.shuffle(pool)
            take = min(args.per_category, len(pool))
            if take < args.per_category:
                warnings.append(f"{c}: only {take} clean clips (< {args.per_category})")
            for r in pool[:take]:
                test_ids.add(id(r)); test_rows.append(r)

    test_subjects = set(r['subject_id'] for r in test_rows)

    # ── near-dup index for sequence mode ────────────────────────────────────
    dup_index = defaultdict(list)
    if args.leakage == 'sequence':
        for r in test_rows:
            key = (r['subject_id'], r['category'])
            dup_index[key].append((signature(r, scale), float(r['n_frames'])))

    def is_near_dup_of_test(r):
        key = (r['subject_id'], r['category'])
        if key not in dup_index:
            return False
        sig = signature(r, scale)
        nf = float(r['n_frames'])
        for tsig, tnf in dup_index[key]:
            frame_rel = abs(nf - tnf) / max(tnf, 1.0)
            if frame_rel <= args.dup_frame_tol and \
               np.linalg.norm(sig - tsig) <= args.dup_tol:
                return True
        return False

    # ── build TRAIN set ─────────────────────────────────────────────────────
    train_rows = []
    dropped_dup = dropped_subject = dropped_ood = dropped_source = 0
    test_sources = set(r['subject_id'].split('/')[0] for r in test_rows)

    for r in rows:
        if id(r) in test_ids:
            continue
        if not args.include_failed_in_train and not r['qc_pass']:
            continue
        if args.ood_test_only and r['category'] == OOD:
            dropped_ood += 1
            continue
        if args.leakage == 'subject' and r['subject_id'] in test_subjects:
            dropped_subject += 1
            continue
        if args.leakage == 'source' and \
                r['subject_id'].split('/')[0] in test_sources:
            dropped_source += 1
            continue
        if args.leakage == 'sequence' and is_near_dup_of_test(r):
            dropped_dup += 1
            continue
        train_rows.append(r)

    # ── write CSVs ──────────────────────────────────────────────────────────
    fieldnames = list(rows[0].keys()) + ['split']
    with open(args.test_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in test_rows:
            rr = dict(r); rr['split'] = 'test'; w.writerow(rr)
    with open(args.train_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in train_rows:
            rr = dict(r); rr['split'] = 'train'; w.writerow(rr)

    # ── summary ─────────────────────────────────────────────────────────────
    test_cat = Counter(r['category'] for r in test_rows)
    train_cat = Counter(r['category'] for r in train_rows)
    test_src = Counter(r['source'] for r in test_rows)
    train_src = Counter(r['source'] for r in train_rows)
    train_subjects = set(r['subject_id'] for r in train_rows)
    overlap = test_subjects & train_subjects

    L = []
    L.append("=" * 58)
    L.append(f"TRAIN / TEST SPLIT SUMMARY   (leakage policy: {args.leakage})")
    L.append("=" * 58)
    L.append(f"Target per-category test size : {args.per_category}")
    L.append(f"Seed                          : {args.seed}")
    L.append(f"OOD test-only                 : {args.ood_test_only}")
    L.append(f"Env-support clips removed     : {n_env}"
             + ("  (kept: --keep_env_support)" if args.keep_env_support else ""))
    if args.leakage == 'sequence':
        L.append(f"Near-dup tol (L2 / frame%)    : {args.dup_tol} / "
                 f"{args.dup_frame_tol*100:.0f}%")
    L.append("")
    L.append(f"TEST  total : {len(test_rows):,}   sources: {dict(test_src)}")
    for c in CATEGORIES:
        L.append(f"    {c:28s} {test_cat.get(c,0):5d}")
    L.append("")
    L.append(f"TRAIN total : {len(train_rows):,}   sources: {dict(train_src)}")
    for c in CATEGORIES:
        L.append(f"    {c:28s} {train_cat.get(c,0):5d}")
    L.append("")
    L.append("Clips excluded from TRAIN:")
    if args.leakage == 'sequence':
        L.append(f"    near-duplicates of test : {dropped_dup}")
    if args.leakage == 'subject':
        L.append(f"    subject leakage         : {dropped_subject}")
    if args.leakage == 'source':
        L.append(f"    source leakage          : {dropped_source}")
    if args.ood_test_only:
        L.append(f"    OOD (test-only policy)  : {dropped_ood}")
    L.append("")
    L.append(f"Subjects in TEST  : {len(test_subjects)}")
    L.append(f"Subjects in TRAIN : {len(train_subjects)}")
    if args.leakage == 'subject':
        L.append(f"Subject overlap   : {len(overlap)}  "
                 f"{'(LEAKAGE!)' if overlap else '(clean, disjoint)'}")
    else:
        L.append(f"Subject overlap   : {len(overlap)}  "
                 f"(expected for {args.leakage} mode; exact/near dups removed)")
    if warnings:
        L.append("")
        L.append("WARNINGS:")
        for w2 in warnings:
            L.append("  - " + w2)
    L.append("=" * 58)

    report = "\n".join(L)
    print("\n" + report)
    with open(args.summary, 'w') as fh:
        fh.write(report + "\n")
    print(f"\nSaved: {args.test_out}, {args.train_out}, {args.summary}")


if __name__ == "__main__":
    main()
