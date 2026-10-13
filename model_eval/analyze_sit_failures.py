import glob, json, os, csv, sys
import numpy as np
pat = sys.argv[1] if len(sys.argv) > 1 else "*EVAL_picoset1002sit_chair_sitmixv2_004k"
d = json.load(open(sorted(glob.glob("/home/grease/GR00T-WholeBodyControl/logs_eval/" + pat))[-1] + "/metrics_eval.json"))
a = d["eval/all_metrics_dict"]
keys = [str(k) for k in a["motion_keys"]]; term = np.asarray(a["terminated"]).astype(bool); prog = np.asarray(a["progress"])
meta = {r["clip"][:-4]: r for r in csv.DictReader(open("/home/grease/g1_robot_data/pico_raw/20261002_151150_sit_clips/clips.csv"))}
phase = {"before sit (walk-in)": 0, "sit-down (0-1.5 s after sit start)": 0, "seated hold": 0, "stand-up (last 1.5 s of sit)": 0, "after stand-up (leave)": 0}
rows = []
for i, k in enumerate(keys):
    if not term[i]: continue
    m = meta[k.replace("pico1002sit_", "")]
    dur, s0, sd = float(m["dur_s"]), float(m["sit_start_s"]), float(m["sit_dur_s"])
    t = prog[i] * dur
    if t < s0: ph = "before sit (walk-in)"
    elif t < s0 + 1.5: ph = "sit-down (0-1.5 s after sit start)"
    elif t < s0 + sd - 1.5: ph = "seated hold"
    elif t < s0 + sd + 0.5: ph = "stand-up (last 1.5 s of sit)"
    else: ph = "after stand-up (leave)"
    phase[ph] += 1
    rows.append((k[-8:], round(t, 1), round(s0, 1), round(sd, 1), ph))
print("failure phase of the %d terminated clips:" % term.sum())
for k, v in phase.items(): print(f"  {k:38s} {v}")
print("early failures (<2 s):", [(r[0], r[1]) for r in rows if r[1] < 2.0])
sit = np.array([float(meta[k.replace('pico1002sit_', '')]['sit_dur_s']) for k in keys])
for lo, hi in ((0, 4), (4, 8), (8, 15), (15, 40)):
    m = (sit >= lo) & (sit < hi)
    print(f"sit duration {lo:2d}-{hi:2d} s: n={int(m.sum()):2d}  success {100 * (1 - term[m].mean()):5.1f}%  mean progress {prog[m].mean():.3f}")
