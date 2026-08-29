import os
import json

files = os.listdir('/home/grease/ego_dataset/eval_subset/robot')
files = [f.replace('.pkl', '') for f in files]

categories = {
    "Agility & Acrobatics": [],
    "Complex Gestures / Dance": [],
    "Unstructured Motion": [],
    "Basic Locomotion (Baseline Repetition)": []
}

for f in files:
    # 1. Agility & Acrobatics
    if 'flip' in f.lower() or 'jump' in f.lower() or 'cartwheel' in f.lower() or 'kick' in f.lower() or 'martial' in f.lower():
        categories["Agility & Acrobatics"].append(f)
    # 2. Complex Gestures / Dance
    elif 'dance' in f.lower() or 'gestures' in f.lower() or 'sway' in f.lower() or 'swing' in f.lower() or 'triumph' in f.lower() or 'bow' in f.lower() or 'yawn' in f.lower() or 'itching' in f.lower() or 'show_bicep' in f.lower() or 'eureka' in f.lower() or 'making_friedeggs' in f.lower() or 'tasty' in f.lower() or 'wipe' in f.lower() or 'sip' in f.lower() or 'button' in f.lower() or 'valve' in f.lower():
        categories["Complex Gestures / Dance"].append(f)
    # 3. Unstructured Motion
    elif 'crawl' in f.lower() or 'heavy' in f.lower() or 'light_two' in f.lower() or 'light_one' in f.lower() or 'kneel' in f.lower() or 'crouch' in f.lower() or 'stoop' in f.lower() or 'door' in f.lower() or 'object_' in f.lower():
        categories["Unstructured Motion"].append(f)
    # 4. Basic Locomotion
    else:
        categories["Basic Locomotion (Baseline Repetition)"].append(f)

# Collect all failed keys across configs to split them
logs = {
    'sonic_release': '/home/grease/GR00T-WholeBodyControl/logs_eval/metrics/EVAL_SUBSET_OFFLINE/metrics_eval.json',
    'low_latency': '/home/grease/GR00T-WholeBodyControl/logs_eval/metrics/EVAL_SUBSET_LOW_LATENCY/metrics_eval.json',
    'sonic_pretrained': '/home/grease/GR00T-WholeBodyControl/logs_eval/metrics/EVAL_SUBSET_PRETRAINED/metrics_eval.json',
    'scratch_stable_lr': '/home/grease/GR00T-WholeBodyControl/logs_eval/metrics/EVAL_SUBSET_SCRATCH_STABLE_LR/metrics_eval.json'
}

failures_by_cat = {c: {} for c in categories}
for c in failures_by_cat:
    for ckpt in logs:
        failures_by_cat[c][ckpt] = []

for ckpt_name, path in logs.items():
    if not os.path.exists(path): continue
    with open(path) as f:
        data = json.load(f)
    failed_keys = data.get('failed_keys', [])
    for failed_key in failed_keys:
        for cat, items in categories.items():
            if failed_key in items:
                failures_by_cat[cat][ckpt_name].append(failed_key)
                break

print("=== Failure Breakdown by Category ===")
for cat, items in categories.items():
    print(f"\n{cat} (Total {len(items)} motions)")
    for ckpt_name in logs.keys():
        fail_list = failures_by_cat[cat][ckpt_name]
        print(f"  - {ckpt_name}: {len(fail_list)} failures ({len(fail_list)/len(items)*100:.1f}%)")
        if len(fail_list) > 0 and len(fail_list) <= 5:
            # Print names if only a few failed
            print(f"      {fail_list}")
