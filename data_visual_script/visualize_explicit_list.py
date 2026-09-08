"""Batch-visualize a specific, explicit list of clip names (unlike the
built-in --sample_n random-from-whole-folder mode), by calling process_one()
directly for each name in a supplied text file."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visualize_smpl_robot_pair as P

names_file = sys.argv[1]
smpl_dir = sys.argv[2]
robot_dir = sys.argv[3]
out_dir = sys.argv[4]
frame_step = int(sys.argv[5]) if len(sys.argv) > 5 else 4

with open(names_file) as f:
    names = [l.strip() for l in f if l.strip()]

print(f"{len(names)} explicit clips to render")
n_ok, n_fail = 0, 0
for i, name in enumerate(names):
    print(f"\n[{i + 1}/{len(names)}]", end="")
    out_path = os.path.join(out_dir, f"{name}.gif")
    try:
        P.process_one(name, smpl_dir, robot_dir, out_path, frame_step)
        n_ok += 1
    except Exception as e:
        print(f"  [!] FAILED: {name}: {e}")
        n_fail += 1

print(f"\n\nDone: {n_ok} succeeded, {n_fail} failed, out of {len(names)} total")
