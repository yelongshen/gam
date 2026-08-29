import argparse
import csv
import glob
import os

def create_split_csv(input_dir, output_csv):
    files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    if not files:
        print(f"No .pkl files found in {input_dir}")
        return False
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'path'])
        for p in files:
            writer.writerow(['eval_subset', p])
    print(f"Created {output_csv} with {len(files)} clips.")
    return True

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', default='/home/grease/ego_dataset/eval_subset/smpl')
    ap.add_argument('--out_csv', default='data_analysis/split/split_eval_subset.csv')
    args = ap.parse_args()
    
    create_split_csv(args.input_dir, args.out_csv)
