import os
import glob
import joblib

# From our earlier check, the user has downloaded AMASS. 
# But it takes hours to retarget SMPL betas to G1 kinematics. 
# Let's leverage the existing filter set from GR00T-WholeBodyControl to quickly assemble our 4 chosen categories!
# GR00T-WBC smpl_filtered already contains the 131k AMASS sequences converted into exact G1 kinematics!

source_dir = "/home/grease/GR00T-WholeBodyControl/data/smpl_filtered"
output_dir = "gear_sonic_deploy/reference/evaluation_set"

os.makedirs(output_dir, exist_ok=True)

# 1. We identify a few standard filenames for each category requested
eval_mapping = {
    "Basic_Locomotion": [
        "walking",
        "jogging",
        "run",
        "backward",
        "crouch"
    ],
    "Agility_HighDynamic": [
        "box",
        "kick",
        "jump",
        "dance"
    ],
    "UpperBody_Manipulation": [
        "reach",
        "wave",
        "place",
        "pick"
    ]
}

print(f"Scanning {source_dir} for evaluation candidates...")
all_files = glob.glob(os.path.join(source_dir, "*.pkl"))

selected_set = {k: [] for k in eval_mapping.keys()}

# We just want ~5 files per category so it's manageable for simulation testing.
for f in all_files:
    fname = os.path.basename(f).lower()
    for cat, keywords in eval_mapping.items():
        if len(selected_set[cat]) < 5:
            if any(k in fname for k in keywords):
                selected_set[cat].append(f)
                break

for cat, files in selected_set.items():
    print(f"\n--- {cat} ---")
    for f in files:
         print("  ", os.path.basename(f))

# Let's write out a shell script to use the provided convert_motions.py to make them C++ compatible
with open("build_eval_set.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    for cat, files in selected_set.items():
        cat_dir = os.path.join(output_dir, cat)
        f.write(f"mkdir -p {cat_dir}\n")
        f.write(f"echo 'Processing category: {cat}'\n")
        for pkl in files:
            f.write(f".venv_sim/bin/python gear_sonic_deploy/reference/convert_motions.py {pkl} {cat_dir}\n")

print("\nGenerated build_eval_set.sh to convert these into C++ tracker format.")
