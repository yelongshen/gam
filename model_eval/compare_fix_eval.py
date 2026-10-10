import glob, json, os, numpy as np
fixed = {ds: set(l.strip() for l in open(f"/tmp/fix_{ds}.txt") if l.strip()) for ds in ("picoset_20260928", "picoset_20260929", "picoset_20260930")}
def load(ds, tag):
    mf = sorted(glob.glob(f"/home/grease/GR00T-WholeBodyControl/logs_eval/*EVAL_early_{ds}_{tag}/metrics_eval.json"))[-1]
    d = json.load(open(mf)); a = d["eval/all_metrics_dict"]
    keys = [str(k) for k in a["motion_keys"]]; term = np.asarray(a["terminated"]).astype(bool)
    rec = {}
    for l in open(f"/tmp/terms_early_{ds}_{tag}.jsonl"):
        r = json.loads(l); rec.setdefault(r["motion"], r)
    t = {k: (rec[k]["step"] / 30.0 if term[i] else None) for i, k in enumerate(keys)}
    pr = {k: float(np.asarray(a["progress"])[i]) for i, k in enumerate(keys)}
    return d, keys, t, pr
tot = {"b": [0, 0, 0, 0], "a": [0, 0, 0, 0]}
print(f"{'set':18s}{'':8s}{'success':>10s}{'progress':>10s}{'early<1s':>10s}{'early<3s':>10s}")
for ds in ("picoset_20260928", "picoset_20260929", "picoset_20260930"):
    rb = load(ds, "novr050k"); ra = load(ds, "novr050k_fix")
    for nm, (d, keys, t, pr) in (("before", rb), ("after", ra)):
        n = len(keys); e1 = sum(1 for k in keys if t[k] is not None and t[k] < 1); e3 = sum(1 for k in keys if t[k] is not None and t[k] < 3)
        print(f"{ds:18s}{nm:8s}{d['eval/success/success_rate']:10.3f}{d['eval/success/progress_rate']:10.3f}{e1:10d}{e3:10d}   (n={n})")
    # clips that were re-retargeted
    fx = [k for k in rb[1] if k in fixed[ds]]
    for nm, (d, keys, t, pr) in (("before", rb), ("after", ra)):
        e3 = sum(1 for k in fx if t[k] is not None and t[k] < 3); ok = sum(1 for k in fx if t[k] is None)
        print(f"   re-retargeted clips ({len(fx)}) {nm:6s}: early<3s {e3:3d} | survive whole clip {ok:3d} | mean progress {np.mean([pr[k] for k in fx]):.3f}")
    ctl = [k for k in rb[1] if k not in fixed[ds]]
    for nm, (d, keys, t, pr) in (("before", rb), ("after", ra)):
        print(f"   untouched control clips ({len(ctl)}) {nm:6s}: early<3s {sum(1 for k in ctl if t[k] is not None and t[k] < 3):3d} | survive {sum(1 for k in ctl if t[k] is None):3d} | mean progress {np.mean([pr[k] for k in ctl]):.3f}")
