import os
import csv

data_dir = '/home/grease/ego_dataset/smpl_filtered_to_bvh_csv_robot_filtered/smpl_filtered_to_bvh_csv'
files = os.listdir(data_dir)

csv_path = '/home/grease/gam/model_eval/dataset_categories.csv'

def categorize(filename):
    f = filename.lower()
    if 'flip' in f or 'jump' in f or 'cartwheel' in f or 'kick' in f or 'martial' in f:
        return "Agility & Acrobatics"
    elif 'dance' in f or 'gestures' in f or 'sway' in f or 'swing' in f or 'triumph' in f or 'bow' in f or 'yawn' in f or 'itching' in f or 'show_bicep' in f or 'eureka' in f or 'making_friedeggs' in f or 'tasty' in f or 'wipe' in f or 'sip' in f or 'button' in f or 'valve' in f:
        return "Complex Gestures / Dance"
    elif 'crawl' in f or 'heavy' in f or 'light_two' in f or 'light_one' in f or 'kneel' in f or 'crouch' in f or 'stoop' in f or 'door' in f or 'object_' in f or 'box' in f:
        return "Unstructured Motion"
    else:
        return "Basic Locomotion"

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'category'])
    for f in sorted(files):
        if not f.endswith('.pkl'):
            continue
        cat = categorize(f)
        writer.writerow([f, cat])

print(f"Generated CSV with {len(files)} categorized entries at {csv_path}")
