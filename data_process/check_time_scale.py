import csv, glob, sys, os
import numpy as np, joblib
P = "/home/grease/g1_robot_data/pico_raw"
sets = {
 "picoset_20260928": ("pico0928_", f"{P}/20260928_162215_clips"),
 "picoset_20260929": ("pico0929_", f"{P}/20260929_174102_clips"),
 "picoset_20260930": ("pico0930_", f"{P}/20260930_combined_clips"),
 "picoset_20261002_sit": ("pico1002sit_", f"{P}/20261002_151150_sit_clips"),
}
for ds, (pref, cdir) in sets.items():
    rows = []
    for r in csv.DictReader(open(f"{cdir}/clips.csv")):
        name = pref + r["clip"][:-4]
        p = f"/home/grease/ego_dataset/{ds}/smpl/{name}.pkl"
        if not os.path.exists(p) or len(rows) >= 12:
            continue
        s = joblib.load(p); s = s[next(iter(s))] if "transl" not in s else s
        n = len(s["transl"]); fps = float(s["fps"])
        d = np.load(f"{cdir}/{r['clip']}", allow_pickle=True)
        t = d["timestamp_monotonic"].reshape(-1).astype(float)
        rows.append((r["clip"][:-4], float(r["dur_s"]), n / fps, len(t) / (t[-1] - t[0]), float(np.asarray(d["pico_fps"]).reshape(-1)[0])))
    a = np.array([[x[1], x[2], x[3], x[4]] for x in rows])
    print(f"{ds}: real duration (clips.csv) vs dataset smpl duration (frames/50): ratio median {np.median(a[:,1]/a[:,0]):.3f}  | file rate {np.median(a[:,2]):.1f} Hz | pico_fps field {np.median(a[:,3]):.1f}")
    print("   e.g.", [(x[0], round(x[1], 1), round(x[2], 1)) for x in rows[:4]])
