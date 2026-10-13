import glob, json, os, sys, csv, collections
import numpy as np
base = "/home/grease/GR00T-WholeBodyControl/logs_eval/"
meta = {r["clip"][:-4]: r for r in csv.DictReader(open("/home/grease/g1_robot_data/pico_raw/20261002_151150_sit_clips/clips.csv"))}
fixed = {"clip_%s" % x for x in ("001", "008", "009", "024", "027", "032", "044", "045", "046")}
tags = sys.argv[1:] or ["short_sitmixv2_004k", "short_novr050k"]
for tag in tags:
    mdir = sorted(glob.glob(base + f"*EVAL_picoset1002sit_chair_{tag}"))[-1]
    d = json.load(open(mdir + "/metrics_eval.json"))
    a = d["eval/all_metrics_dict"]
    keys = [str(k).replace("pico1002sit_", "") for k in a["motion_keys"]]
    term = np.asarray(a["terminated"]).astype(bool); prog = np.asarray(a["progress"])
    jl = "/tmp/terms_%s.jsonl" % ("sitmixv2" if "sitmixv2" in tag else "novr050k")
    ev = {}
    for ln in open(jl):
        r = json.loads(ln); ev.setdefault(r["motion"].replace("pico1002sit_", ""), r)
    print("=" * 100)
    print(f"{tag}:  success {d['eval/success/success_rate']:.3f}  progress {d['eval/success/progress_rate']:.3f}  mpjpe_l(success) {d['eval/success/mpjpe_l']:.1f} mm   [{len(keys)} clips, {int(term.sum())} terminated]")
    print(f"{'clip':9s}{'set':7s}{'dur':>6s}{'sit@':>6s}{'sit_s':>6s} {'result':9s}{'t_fail':>7s}  {'phase':12s} termination reason(s)")
    cnt = collections.Counter(); cnt_fixed = collections.Counter()
    for i, k in enumerate(keys):
        m = meta[k]; dur, s0, sd = float(m["dur_s"]), float(m["sit_start_s"]), float(m["sit_dur_s"])
        if not term[i]:
            print(f"{k:9s}{'FIXED' if k in fixed else 'other':7s}{dur:6.1f}{s0:6.1f}{sd:6.1f} {'ok':9s}"); continue
        r = ev.get(k, {}); t = r.get("step", -1) / 30.0
        ph = "walk-in" if t < s0 else "seated hold" if t < s0 + sd - 1.5 else "stand-up" if t < s0 + sd + 0.5 else "leave"
        fired = ",".join(r.get("fired", [])) or "(none / not logged)"
        print(f"{k:9s}{'FIXED' if k in fixed else 'other':7s}{dur:6.1f}{s0:6.1f}{sd:6.1f} {'TERM':9s}{t:7.2f}  {ph:12s} {fired}   root_z={r.get('root_z', float('nan')):.2f}")
        for f in r.get("fired", []): (cnt_fixed if k in fixed else cnt)[f] += 1
    print("fixed 9 clips: terminated %d/9 ; first-fired terms:" % sum(term[i] for i, k in enumerate(keys) if k in fixed), dict(cnt_fixed))
    print("other clips : terminated %d/%d ; terms:" % (sum(term[i] for i, k in enumerate(keys) if k not in fixed), sum(1 for k in keys if k not in fixed)), dict(cnt))
