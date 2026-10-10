import glob, json, os, sys
import numpy as np
base = "/home/grease/GR00T-WholeBodyControl/logs_eval"
pat = sys.argv[1] if len(sys.argv) > 1 else "*EVAL_picoset1002sit_chair_sitmixv2_004k"
dirs = sorted(glob.glob(os.path.join(base, pat)))
print("dirs:", [os.path.basename(d) for d in dirs])
d = json.load(open(os.path.join(dirs[-1], "metrics_eval.json")))
for k in ("eval/success/success_rate", "eval/success/progress_rate", "eval/success/mpjpe_l", "eval/success/mpjpe_g", "eval/success/mpjpe_pa",
          "eval/all/mpjpe_l", "eval/all/mpjpe_g", "eval/all/mpjpe_pa"):
    if k in d: print(f"{k:34s} {d[k]}")
a = d["eval/all_metrics_dict"]
keys = [str(k) for k in a["motion_keys"]]
term = np.asarray(a["terminated"]).astype(bool)
prog = np.asarray(a["progress"]) if "progress" in a else None
n = len(keys)
print(f"clips {n}  terminated {int(term.sum())}  success {n - int(term.sum())} ({100 * (n - term.sum()) / n:.1f}%)")
if prog is not None:
    print("progress: mean %.3f  median %.3f  | terminated clips mean progress %.3f" % (prog.mean(), np.median(prog), prog[term].mean() if term.any() else float('nan')))
import csv
meta = {r["clip"][:-4]: r for r in csv.DictReader(open("/home/grease/g1_robot_data/pico_raw/20261002_151150_sit_clips/clips.csv"))}
ep = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob("/home/grease/ego_dataset/picoset_20261002_sit/envs/*.json")}
print("\nper-clip (sorted by progress):  clip  dur_s  sit_s  seat_top  progress  term")
rows = []
for i, k in enumerate(keys):
    cid = k.replace("pico1002sit_", "")
    m = meta.get(cid, {})
    rows.append((float(prog[i]) if prog is not None else 0.0, k, float(m.get("dur_s", 0)), float(m.get("sit_dur_s", 0)), ep[k]["seat_center"][2] if k in ep else 0.0, bool(term[i])))
for r in sorted(rows)[:30]:
    print(f"  {r[1][-8:]:>10s}  {r[2]:5.1f}  {r[3]:5.1f}  {r[4]:.3f}  {r[0]:.3f}  {r[5]}")
# what separates success from failure?
rs = np.array([[r[2], r[3], r[4], r[0]] for r in rows]); t = np.array([r[5] for r in rows])
for j, nm in ((0, "clip duration s"), (1, "sit duration s"), (2, "seat top height m")):
    print(f"{nm:20s} terminated median {np.median(rs[t, j]) if t.any() else float('nan'):.3f} | succeeded median {np.median(rs[~t, j]) if (~t).any() else float('nan'):.3f}")
print("progress percentiles of terminated clips 10/50/90:", np.round(np.percentile(rs[t, 3], [10, 50, 90]), 3) if t.any() else None)
