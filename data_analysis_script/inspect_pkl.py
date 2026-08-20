import pickle
import sys

pkl_path = sys.argv[1]
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
print(f"File: {pkl_path}")
print("=" * 60)
if isinstance(data, dict):
    for key, val in data.items():
        if hasattr(val, 'shape'):
            print(f"Key: {key:<25} | Shape: {str(val.shape):<15} | Type: {val.dtype}")
        else:
            print(f"Key: {key:<25} | Type: {type(val)} | Value: {str(val)[:30]}")
print("=" * 60)
