import numpy as np
import sys

npz_path = sys.argv[1]
data = np.load(npz_path)
print(f"File: {npz_path}")
print("=" * 60)
for key in data.files:
    arr = data[key]
    print(f"Key: {key:<25} | Shape: {str(arr.shape):<15} | Type: {arr.dtype}")
print("=" * 60)
