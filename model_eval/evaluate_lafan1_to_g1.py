import os
import glob
import numpy as np

lafan_dir = "/home/grease/egodata/downloads/lafan1_extracted"

print(f"Scanning LAFAN1 raw dataset at {lafan_dir}...")
lafan_files = glob.glob(os.path.join(lafan_dir, "*.bvh"))

categories = {
    "Basic_Locomotion": [f for f in lafan_files if "walk" in f.lower() or "run" in f.lower()],
    "Agility_HighDynamic": [f for f in lafan_files if "jump" in f.lower() or "falling" in f.lower() or "aiming" in f.lower()],
    "UpperBody_Manipulation": [f for f in lafan_files if "dance" in f.lower() or "fight" in f.lower()]
}

print(f"Total LAFAN1 .bvh files found: {len(lafan_files)}\n")
for cat, files in categories.items():
    print(f"[{cat}] - Found {len(files)} files")
    for f in files[:4]:  
        print(f"  - {os.path.basename(f)}")
    print()
