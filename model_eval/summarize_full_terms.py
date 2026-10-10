import glob, json, os, csv, sys, collections
import numpy as np
tag, jl = sys.argv[1], sys.argv[2]
d = json.load(open(sorted(glob.glob(f"/home/grease/GR00T-WholeBodyControl/logs_eval/*EVAL_picoset1002sit_chair_{tag}"))[-1] + "/metrics_eval.json"))
a = d["eval/all_metrics_dict"]; keys = [str(k) for k in a["motion_keys"]]; term = np.asarray(a["terminated"]).astype(bool); prog = np.asarray(a["progress"])
meta = {r["clip"][:-4]: r for r in csv.DictReader(open("/home/grease/g1_robot_data/pico_raw/20261002_151150_sit_clips/clips.csv"))}
rec = {}
for l in open(jl):
    r = json.loads(l); rec.setdefault(r["motion"], r)
fixed = {"clip_%s" % x for x in ("001", "008", "009", "024", "027", "032", "044", "045", "046")}
n = len(keys)
print(f"{tag}: success {d['eval/success/success_rate']:.3f} ({n - int(term.sum())}/{n})  progress {d['eval/success/progress_rate']:.3f}  mpjpe_l(success) {d['eval/success/mpjpe_l']:.1f} mm  mpjpe_g {d['eval/success/mpjpe_g']:.0f} mm")
ph = collections.Counter(); reasons = collections.Counter()
rows = []
for i, k in enumerate(keys):
    c = k.replace("pico1002sit_", ""); m = meta[c]
    if not term[i]: rows.append((c, "ok", 0.0, "")); continue
    t = prog[i] * float(m["dur_s"]); s0 = float(m["sit_start_s"]); sd = float(m["sit_dur_s"])
    p = "walk-in" if t < s0 else ("sit-down" if t < s0 + 1.5 else "seated hold" if t < s0 + sd - 1.5 else "stand-up" if t < s0 + sd + 0.5 else "leave")
    f = ",".join(rec.get(k, {}).get("fired", ["?"])); ph[p] += 1; reasons[f] += 1
    rows.append((c, p, t, f))
print("failure phase:", dict(ph)); print("termination reason:", dict(reasons))
fx = [r for r in rows if r[0] in fixed]
print("the 9 previously failing clips:", "; ".join(f"{r[0][-3:]} {r[1]}{'' if r[1] == 'ok' else ' @%.1fs' % r[2]}" for r in fx))
sit = np.array([float(meta[k.replace('pico1002sit_', '')]['sit_dur_s']) for k in keys])
for lo, hi in ((0, 4), (4, 8), (8, 15), (15, 40)):
    mk = (sit >= lo) & (sit < hi); print(f"sit {lo:2d}-{hi:2d} s: n={int(mk.sum()):2d} success {100 * (1 - term[mk].mean()):5.1f}%")
