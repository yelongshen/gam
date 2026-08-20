import numpy as np

# We can't trivially load .bvh without a parser like bvhpy, but we can look for the string 'Hand' or 'Finger' inside the file.
bvh_file = '/home/grease/egodata/downloads/lafan1_extracted/aiming1_subject1.bvh'
with open(bvh_file, 'r') as f:
    hierarchy = []
    for line in f:
        if line.strip() == "MOTION":
            break
        if "JOINT" in line or "ROOT" in line:
            hierarchy.append(line.strip().split()[-1])

print("Total joints in LAFAN1 BVH:", len(hierarchy))
hands_fingers = [j for j in hierarchy if 'hand' in j.lower() or 'finger' in j.lower() or 'thumb' in j.lower()]
print("Hand/Finger joints found:", hands_fingers)
