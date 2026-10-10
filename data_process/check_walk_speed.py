import csv, os
import numpy as np, joblib
P = "/home/grease/g1_robot_data/pico_raw"
for ds, pref, cdir in (("picoset_20260929", "pico0929_", f"{P}/20260929_174102_clips"), ("picoset_20260930", "pico0930_", f"{P}/20260930_combined_clips")):
    real, dset = [], []
    for r in csv.DictReader(open(f"{cdir}/clips.csv")):
        path, dur = float(r["path_m"]), float(r["dur_s"])
        if path < 6.0:
            continue
        p = f"/home/grease/ego_dataset/{ds}/smpl/{pref}{r['clip'][:-4]}.pkl"
        if not os.path.exists(p): continue
        s = joblib.load(p); s = s[next(iter(s))] if "transl" not in s else s
        tr = np.asarray(s["transl"], float)
        sp = np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=1).sum()
        real.append(path / dur); dset.append(sp / (len(tr) / float(s["fps"])))
    print(f"{ds}: {len(real)} locomotion clips (path > 6 m): average root speed from the raw timestamps median {np.median(real):.2f} m/s | in the dataset {np.median(dset):.2f} m/s (ratio {np.median(np.array(dset)/np.array(real)):.2f})")
