"""
Build a `lafan1_evalset/` or `amass_evalset/` directory with the SAME flat
layout as the existing `ego_dataset/eval_subset/` (i.e. just two
subdirectories, `smpl/<name>.pkl` and `robot/<name>.pkl`, no per-category
subfolders), populated from a `classify_amass_split.py`-produced test-split
CSV (columns: category,name,smpl_path,robot_path).

Uses symlinks (not copies) to avoid duplicating multi-MB/GB motion_lib
files on disk -- same pattern used for the ad-hoc `/tmp/debug_eval_subset_motion/`
dirs used earlier in this session.

Robot motion_lib files are named "<name>_motion_lib.pkl" in their source
directories; the symlink itself is created as "<name>.pkl" (matching
eval_subset/robot/'s naming, which has no such suffix) so downstream tools
that expect eval_subset's exact naming convention work unmodified.

Usage:
    .venv_sim/bin/python build_evalset_dir.py \
        --split_csv data_analysis/split/amass_test_split.csv \
        --out_dir /home/grease/ego_dataset/amass_evalset

    .venv_sim/bin/python build_evalset_dir.py \
        --split_csv data_analysis/split/lafan1_test_split.csv \
        --out_dir /home/grease/ego_dataset/lafan1_evalset
"""
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_csv', required=True,
                     help='CSV with columns: category,name,smpl_path,robot_path')
    ap.add_argument('--out_dir', required=True,
                     help='destination root; will contain smpl/ and robot/ subdirs')
    args = ap.parse_args()

    smpl_dir = os.path.join(args.out_dir, 'smpl')
    robot_dir = os.path.join(args.out_dir, 'robot')
    os.makedirs(smpl_dir, exist_ok=True)
    os.makedirs(robot_dir, exist_ok=True)

    n_ok = 0
    n_missing = 0
    with open(args.split_csv) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        name = r['name']
        smpl_src = os.path.abspath(r['smpl_path'])
        robot_src = os.path.abspath(r['robot_path'])

        if not (os.path.exists(smpl_src) and os.path.exists(robot_src)):
            print(f"  [!] missing source for {name}: "
                  f"smpl_exists={os.path.exists(smpl_src)} "
                  f"robot_exists={os.path.exists(robot_src)}")
            n_missing += 1
            continue

        smpl_link = os.path.join(smpl_dir, f'{name}.pkl')
        robot_link = os.path.join(robot_dir, f'{name}.pkl')  # strip any _motion_lib suffix

        for link, src in ((smpl_link, smpl_src), (robot_link, robot_src)):
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(src, link)
        n_ok += 1

    print(f"\nBuilt {args.out_dir}:")
    print(f"  smpl/  -> {n_ok} symlinks")
    print(f"  robot/ -> {n_ok} symlinks")
    if n_missing:
        print(f"  [!] {n_missing} clips skipped (missing source file)")

    # Per-category breakdown, matching classify_amass_split.py's own report.
    from collections import Counter
    cats = Counter(r['category'] for r in rows)
    print("\nPer-category counts in this evalset:")
    for cat, n in sorted(cats.items()):
        print(f"  {cat:28s} {n:4d}")


if __name__ == '__main__':
    main()
