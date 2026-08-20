import os
import glob
import numpy as np

# We'll map the known uncompressed structure of the massive AMASS tree and LAFAN1.
amass_dir = "/home/grease/egodata/downloads/amass/extracted"
lafan_dir = "/home/grease/egodata/downloads/lafan1_extracted"

print("Compiling global hierarchy to categorize motion sequences...")
amass_files = glob.glob(os.path.join(amass_dir, "**/*.npz"), recursive=True)
lafan_files = glob.glob(os.path.join(lafan_dir, "**/*.bvh"), recursive=True)

all_files = [*amass_files, *lafan_files]

categories = {
    "Basic Locomotion": ["walk", "run", "jog", "step", "stand", "sway", "crouch", "squat", "lunge"],
    "Agility / High-Dynamic": ["kick", "punch", "jump", "hop", "box", "cartwheel", "flip", "fall", "aim", "fight", "martial"],
    "Upper-Body Manipulation": ["reach", "wave", "pick", "place", "lift", "gesture", "convers", "dance", "macarena"],
    "Unstructured Motion / OOD": ["crawl", "sit", "lie", "ground"]
}

stats = {k: {"count": 0, "total_frames": 0, "files": []} for k in categories.keys()}
stats["Uncategorized"] = {"count": 0, "total_frames": 0, "files": []}

def get_frames(f):
    try:
        if f.endswith('.npz'):
            dat = np.load(f)
            # AMASS contains 156-dim SMPL sequences under 'poses'
            return dat['poses'].shape[0] if 'poses' in dat else 0
        elif f.endswith('.bvh'):
            # Very basic string count for Frames: line in bvh
            with open(f, 'r') as bvh_f:
               for line in bvh_f:
                  if line.startswith("Frames:"):
                     return int(line.split()[1])
    except:
        pass
    return 0

# Count categories linearly
for i, f in enumerate(all_files):
    name = os.path.basename(f).lower()
    
    # Exclude metadata / rest poses
    if "shape" in name or "calibration" in name:
        continue
        
    found_cat = False
    for cat, kws in categories.items():
        if any(kw in name for kw in kws):
            found_cat = True
            stats[cat]["count"] += 1
            stats[cat]["files"].append(f)
            break
            
    if not found_cat:
        stats["Uncategorized"]["count"] += 1
        stats["Uncategorized"]["files"].append(f)

# Display summary results block
print("\n" + "="*50)
print("📊 DISTRIBUTION OF CATEGORIES")
print(f"Total Processed Sequences: {len(all_files):,}")
print("="*50)

for cat, data in stats.items():
    if data["count"] > 0:
        print(f"\n[{cat}]")
        print(f"  -> Sequences Found: {data['count']:,}")
        
        # We only open the NPZ files dynamically for the calculation sample to save I/O overhead on 20,000 files
        sampled_files = data["files"][:300]
        lengths = [get_frames(f) for f in sampled_files]
        valid_lengths = [l for l in lengths if l > 0]
        
        if valid_lengths:
            print(f"  -> Avg Length:      {np.mean(valid_lengths):.1f} frames")
            print(f"  -> Est Total Time:  ~{np.mean(valid_lengths) * data['count'] / (50 * 60):.1f} minutes of motion data")
        
        # Show a quick sample filename to verify semantic routing
        if len(data["files"]) > 0:
             print(f"  -> Example match:   {os.path.basename(data['files'][0])}")

print("\n" + "="*50)
