import os
import glob
import numpy as np

amass_dir = "/home/grease/egodata/downloads/amass/extracted/ACCAD"

print(f"Scanning AMASS (ACCAD) raw dataset at {amass_dir}...")
amass_files = glob.glob(os.path.join(amass_dir, "**/*.npz"), recursive=True)

# Select an appropriate mapping 
categories = {
    "Basic_Locomotion": [f for f in amass_files if "Walking" in f or "Stand" in f or "Sway" in f],
    "Agility_HighDynamic": [f for f in amass_files if "MartialArts" in f or "CartWheel" in f or "Running" in f],
    "UpperBody_Manipulation": [f for f in amass_files if "Gestures" in f or "pick" in f or "lift" in f or "Convers" in f]
}

# The target output folder
eval_output_dir = "gear_sonic_deploy/reference/evaluation_set_raw_smpl"
os.makedirs(eval_output_dir, exist_ok=True)

# Print selection metrics
print(f"Total ACCAD .npz files found: {len(amass_files)}\n")
for cat, files in categories.items():
    print(f"[{cat}] - Found {len(files)} files")
    for f in files[:4]:  # Show just the first 4 for brevity
        print(f"  - {os.path.basename(f)}")
    print()

print("To use these directly in the SONIC pipeline, the raw AMASS poses (which are 156-dim SMPL sequences) must be converted into the strict SMPL tracking payload:")
print("A script would normally load 'poses' and 'trans' via `smplx` to generate Cartesian `smpl_joints` [24x3], saving it into `data/smpl_filtered` formatting.")

# Since writing the full generic SMPLx logic script from scratch implies many external dependencies, 
# let's just create a list output file script so we know perfectly what we grabbed.
