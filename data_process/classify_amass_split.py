"""
Classify the AMASS smpl_filtered<->retargeted-robot pairs into the 4
SONIC-paper-aligned categories already implemented in
`classify_motions.py` (Unstructured/OOD, Agility/High-Dynamic,
Basic Locomotion, Upper-Body Manipulation), then randomly sample a fixed
number of clips per category into a test split.

Only clips that exist in BOTH:
    smpl_dir  : /home/grease/ego_dataset/amass_smpl_filtered_v2/<name>.pkl
    robot_dir : /home/grease/gamc/storage/test/amass_retarget/<name>.npz
are eligible, since a usable test clip needs both the human SMPL reference
AND its retargeted G1 counterpart.

Classification reuses `classify_motions.extract_features()` +
`.classify()` directly on each pkl's own `smpl_joints` (T,24,3 meters,
Z-up) -- no BVH re-parsing needed, since smpl_filtered pkls already store
joints in that exact convention.

Usage:
    .venv_sim/bin/python classify_amass_split.py \
        --smpl_dir /home/grease/ego_dataset/amass_smpl_filtered_v2 \
        --robot_dir /home/grease/gamc/storage/test/amass_retarget \
        --per_category 40 --seed 0 \
        --out_csv data_analysis/split/amass_test_split.csv
"""
import argparse
import glob
import os
import random
import sys
from collections import defaultdict

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_motions as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smpl_dir', required=True)
    ap.add_argument('--robot_dir', required=True)
    ap.add_argument('--per_category', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--full_csv', default=None,
                     help='optional: dump ALL classified pairs (not just the sampled '
                          'test split) here, for inspection/reuse')
    args = ap.parse_args()

    smpl_names = {os.path.splitext(f)[0] for f in os.listdir(args.smpl_dir)
                  if f.endswith('.pkl')}
    # Robot motion_lib pkls are named "<name>_motion_lib.pkl" (as of the
    # amass_retarget_motion_lib / lafan1_all_retarget_motion_lib update) --
    # strip that suffix so names line up 1:1 with the smpl_dir stems.
    robot_names = set()
    robot_stem_to_file = {}
    for f in os.listdir(args.robot_dir):
        if not (f.endswith('.pkl') or f.endswith('.npz')):
            continue
        stem = os.path.splitext(f)[0]
        if stem.endswith('_motion_lib'):
            stem = stem[:-len('_motion_lib')]
        robot_names.add(stem)
        robot_stem_to_file[stem] = f
    paired = sorted(smpl_names & robot_names)
    print(f"smpl-only: {len(smpl_names)}  robot-only: {len(robot_names)}  "
          f"paired (usable): {len(paired)}")

    by_cat = defaultdict(list)
    n_fail = 0
    t0_report = max(1, len(paired) // 20)
    for i, name in enumerate(paired):
        pkl_path = os.path.join(args.smpl_dir, f'{name}.pkl')
        try:
            data = joblib.load(pkl_path)
            joints = data['smpl_joints']
            feat = C.extract_features(joints)
            cat = C.classify(feat)
        except Exception as e:
            n_fail += 1
            cat = None
        if cat is not None:
            by_cat[cat].append(name)
        if (i + 1) % t0_report == 0 or i + 1 == len(paired):
            print(f"  classified {i + 1}/{len(paired)}...", flush=True)

    print(f"\nFailed to classify: {n_fail}")
    print("Category counts (all paired clips):")
    for cat in sorted(by_cat.keys()):
        print(f"  {cat:28s} {len(by_cat[cat]):5d}")

    if args.full_csv:
        os.makedirs(os.path.dirname(args.full_csv) or '.', exist_ok=True)
        with open(args.full_csv, 'w') as f:
            f.write('category,name,smpl_path,robot_path\n')
            for cat, names in by_cat.items():
                for name in names:
                    f.write(f"{cat},{name},"
                            f"{os.path.join(args.smpl_dir, name + '.pkl')},"
                            f"{os.path.join(args.robot_dir, robot_stem_to_file[name])}\n")
        print(f"Full classified list saved -> {args.full_csv}")

    # Random sample per category for the test split.
    random.seed(args.seed)
    selected = []
    print(f"\nSampling {args.per_category}/category (seed={args.seed}):")
    for cat in sorted(by_cat.keys()):
        names = by_cat[cat][:]
        random.shuffle(names)
        take = names[:args.per_category]
        print(f"  {cat:28s} requested {args.per_category:3d}  "
              f"available {len(names):5d}  took {len(take):3d}")
        selected.extend((cat, n) for n in take)

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w') as f:
        f.write('category,name,smpl_path,robot_path\n')
        for cat, name in selected:
            f.write(f"{cat},{name},"
                    f"{os.path.join(args.smpl_dir, name + '.pkl')},"
                    f"{os.path.join(args.robot_dir, robot_stem_to_file[name])}\n")
    print(f"\nTest split saved -> {args.out_csv}  ({len(selected)} clips total)")


if __name__ == '__main__':
    main()
