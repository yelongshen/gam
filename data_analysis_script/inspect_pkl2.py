import pickle
import sys
import gzip

try:
    with open(sys.argv[1], 'rb') as f:
        data = pickle.load(f)
except Exception:
    import pandas as pd
    try:
        data = pd.read_pickle(sys.argv[1])
    except Exception:
        import joblib
        data = joblib.load(sys.argv[1])

print(f"File: {sys.argv[1]}")
print("=" * 60)
if isinstance(data, dict):
    for key, val in data.items():
        if hasattr(val, 'shape'):
            print(f"Key: {key:<25} | Shape: {str(val.shape):<15} | Type: {val.dtype}")
        else:
            print(f"Key: {key:<25} | Type: {type(val)} | Value: {str(val)[:30]}")
else:
    print(type(data))
print("=" * 60)
