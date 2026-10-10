import glob, json, os, csv, sys
import numpy as np
base = "/home/grease/GR00T-WholeBodyControl/logs_eval/"
d = json.load(open(sorted(glob.glob(base + "*EVAL_picoset1002sit_chair_sitmixv2_004k"))[-1] + "/metrics_eval.json"))
a = d["eval/all_metrics_dict"]
keys = [str(k) for k in a["motion_keys"]]; term = np.asarray(a["terminated"]).astype(bool); prog = np.asarray(a["progress"])
meta = {r["clip"][:-4]: r for r in csv.DictReader(open("/home/grease/g1_robot_data/pico_raw/20261002_151150_sit_clips/clips.csv"))}
fixed = ["clip_%s" % x for x in ("001", "008", "009", "024", "027", "032", "044", "045", "046")]
rows = []
for i, k in enumerate(keys):
    c = k.replace("pico1002sit_", ""); m = meta[c]
    t = prog[i] * float(m["dur_s"]); s0 = float(m["sit_start_s"]); sd = float(m["sit_dur_s"])
    ph = "ok" if not term[i] else ("walk-in" if t < s0 else "seated" if t < s0 + sd - 1.5 else "stand/leave")
    rows.append((c, float(m["dur_s"]), ph, float(prog[i])))
pick = list(fixed)
for ph, n in (("ok", 3), ("walk-in", 3), ("seated", 3), ("stand/leave", 2)):
    cand = sorted([r for r in rows if r[2] == ph and r[0] not in fixed and r[1] <= 17.0], key=lambda r: r[1])
    pick += [r[0] for r in cand[:n]]
print("subset (%d clips):" % len(pick), pick)
print("longest clip in subset: %.1f s" % max(float(meta[c]["dur_s"]) for c in pick))
D = "/home/grease/ego_dataset/picoset_20261002_sit"; S = "/home/grease/ego_dataset/picoset_20261002_sit_short"
import shutil
shutil.rmtree(S, ignore_errors=True)
for sub in ("robot", "smpl", "envs"):
    os.makedirs(f"{S}/{sub}")
for c in pick:
    n = f"pico1002sit_{c}"
    for sub, ext in (("robot", ".pkl"), ("smpl", ".pkl"), ("envs", ".pkl"), ("envs", ".json")):
        os.symlink(f"{D}/{sub}/{n}{ext}", f"{S}/{sub}/{n}{ext}")
os.symlink(f"{D}/envs/chair_seat.usda", f"{S}/envs/chair_seat.usda")
json.dump(pick, open("/tmp/short_clips.json", "w"))
print("wrote", S)
