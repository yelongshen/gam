"""
Leakage-safe train/test split for the categorized + QC'd motion dataset.

Reads `motion_qc.csv` (produced by motion_quality_check.py) and:
  1. Restricts the TEST pool to clean clips: qc_pass==1 AND outlier_flag==0.
  2. Randomly selects N (default 400) sequences per category for the TEST set,
     with a SUBJECT-DISJOINT guarantee: no subject_id that appears in TEST may
     appear in TRAIN (prevents same-subject/take-family leakage).
  3. Everything else (including QC-failed and outlier clips, optionally) becomes
     the additional TRAINING pool.

Outputs:
    split_test.csv          - the held-out test set (N per category)
    split_train.csv          - the remaining training pool
    split_summary.txt        - counts + provenance breakdown

Usage:
    .venv_sim/bin/python split_train_test.py                 # 400/category
    .venv_sim/bin/python split_train_test.py --per_category 300
    .venv_sim/bin/python split_train_test.py --include_failed_in_train
"""
import os
import csv
import argparse
import random
from collections import defaultdict, Counter

CATEGORIES = [
    "Basic Locomotion",
    "Agility / High-Dynamic",
    "Upper-Body Manipulation",
    "Unstructured / OOD",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qc_csv', default='data_analysis/motion_qc/motion_qc.csv')
    ap.add_argument('--per_category', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--test_out', default='data_analysis/split/split_test.csv')
    ap.add_argument('--train_out', default='data_analysis/split/split_train.csv')
    ap.add_argument('--summary', default='data_analysis/split/split_summary.txt')
    ap.add_argument('--allow_outliers_in_test', action='store_true',
                    help='permit outlier_flag==1 clips into the test pool')
    ap.add_argument('--include_failed_in_train', action='store_true',
                    help='keep QC-failed clips in the training pool')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.qc_csv)))
    for r in rows:
        r['qc_pass'] = int(r['qc_pass'])
        r['outlier_flag'] = int(r.get('outlier_flag', 0) or 0)
    random.seed(args.seed)

    # ── build clean TEST-eligible pool, grouped by category then subject ────
    def test_eligible(r):
        if not r['qc_pass']:
            return False
        if r['outlier_flag'] and not args.allow_outliers_in_test:
            return False
        return True

    # group subjects -> clips, per category
    cat_subj = {c: defaultdict(list) for c in CATEGORIES}
    for r in rows:
        c = r['category']
        if c in cat_subj and test_eligible(r):
            cat_subj[c][r['subject_id']].append(r)

    test_rows = []
    test_subjects = set()
    warnings = []

    for c in CATEGORIES:
        subjects = list(cat_subj[c].keys())
        random.shuffle(subjects)
        picked = []
        # greedily add whole subjects until we reach per_category, so subjects
        # are never split across train/test
        for s in subjects:
            if len(picked) >= args.per_category:
                break
            clips = cat_subj[c][s]
            # if adding this whole subject overshoots a lot, still add (subject
            # disjointness > exact count); we trim at the end by dropping extra
            picked.extend(clips)
            test_subjects.add(s)
        # trim to exactly per_category if we overshot, but keep subject-whole:
        # remove clips from the last-added subjects until <= target while
        # keeping each remaining subject fully in test.
        if len(picked) > args.per_category:
            # sort picked by subject so we can drop trailing whole subjects
            random.shuffle(picked)
            picked = picked[:args.per_category]
            # NOTE: trimming mid-subject can reintroduce that subject into train.
            # To preserve strict disjointness we record which subjects remain.
        actual = len(picked)
        if actual < args.per_category:
            warnings.append(f"{c}: only {actual} clean clips available "
                            f"(< {args.per_category})")
        test_rows.extend((c, r) for r in picked)

    # recompute the true set of subjects fully represented in test
    test_ids = set(id(r) for _, r in test_rows)
    test_subjects = set(r['subject_id'] for _, r in test_rows)

    # ── TRAIN pool = everything not chosen for test, minus leaking subjects ──
    train_rows = []
    leak_dropped = 0
    for r in rows:
        if id(r) in test_ids:
            continue
        if not args.include_failed_in_train and not r['qc_pass']:
            continue
        if r['subject_id'] in test_subjects:
            # subject leaks into test -> drop from train to stay disjoint
            leak_dropped += 1
            continue
        train_rows.append(r)

    # ── write CSVs ──────────────────────────────────────────────────────────
    fieldnames = list(rows[0].keys()) + ['split']
    with open(args.test_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for _, r in test_rows:
            rr = dict(r); rr['split'] = 'test'
            w.writerow(rr)
    with open(args.train_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in train_rows:
            rr = dict(r); rr['split'] = 'train'
            w.writerow(rr)

    # ── summary ─────────────────────────────────────────────────────────────
    test_cat = Counter(c for c, _ in test_rows)
    train_cat = Counter(r['category'] for r in train_rows)
    test_src = Counter(r['source'] for _, r in test_rows)
    train_src = Counter(r['source'] for r in train_rows)
    # leakage assertion
    train_subjects = set(r['subject_id'] for r in train_rows)
    overlap = test_subjects & train_subjects

    lines = []
    lines.append("=" * 56)
    lines.append("TRAIN / TEST SPLIT SUMMARY")
    lines.append("=" * 56)
    lines.append(f"Target per-category test size : {args.per_category}")
    lines.append(f"Seed                          : {args.seed}")
    lines.append("")
    lines.append(f"TEST  total : {len(test_rows):,}   sources: {dict(test_src)}")
    for c in CATEGORIES:
        lines.append(f"    {c:28s} {test_cat.get(c,0):5d}")
    lines.append("")
    lines.append(f"TRAIN total : {len(train_rows):,}   sources: {dict(train_src)}")
    for c in CATEGORIES:
        lines.append(f"    {c:28s} {train_cat.get(c,0):5d}")
    lines.append("")
    lines.append(f"Subjects in TEST  : {len(test_subjects)}")
    lines.append(f"Subjects in TRAIN : {len(train_subjects)}")
    lines.append(f"Subject overlap   : {len(overlap)}  "
                 f"{'(LEAKAGE!)' if overlap else '(clean, disjoint)'}")
    lines.append(f"Train clips dropped to prevent leakage : {leak_dropped}")
    if warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for w2 in warnings:
            lines.append("  - " + w2)
    lines.append("=" * 56)

    report = "\n".join(lines)
    print("\n" + report)
    with open(args.summary, 'w') as fh:
        fh.write(report + "\n")
    print(f"\nSaved: {args.test_out}, {args.train_out}, {args.summary}")


if __name__ == "__main__":
    main()
