import glob
import subprocess
import sys

import numpy as np

f = sorted(glob.glob(sys.argv[1]))[0]
raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "3", "-i", f, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                     capture_output=True).stdout
a = np.frombuffer(raw, dtype=np.uint8).reshape(480, 960, 3)[:, 480:]
low = a[380:470].reshape(-1, 3)
top = a[0:50].reshape(-1, 3)
print(f.split("/")[-1], "| unique colours  lower", len(np.unique(low, axis=0)), " top", len(np.unique(top, axis=0)),
      "| mean RGB lower", low.mean(0).round(0), " top", top.mean(0).round(0))
