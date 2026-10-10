import glob, json, sys, os
from pathlib import Path
import numpy as np, joblib
tag = sys.argv[1] if len(sys.argv) > 1 else "novr050k"
screen = json.load(open("/tmp/screen_retarget_start.json"))
PELV = np.array([0.003, -0.351, 0.012])
def human_lean(ds, clip):
    p = Path(f"/home/grease/ego_dataset/{ds}/smpl/{clip}.pkl")
    s = joblib.load(p); s = s[next(iter(s))] if "smpl_joints" not in s else s
    J = np.asarray(s["smpl_joints"], float)[:100]
    v = J[:, 9] - J[:, 0]
    return float(np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))).mean())
for ds in ("picoset_20260928", "picoset_20260929", "picoset_20260930"):
    mf = sorted(glob.glob(f"/home/grease/GR00T-WholeBodyControl/logs_eval/*EVAL_early_{ds}_{tag}/metrics_eval.json"))
    jl = f"/tmp/terms_early_{ds}_{tag}.jsonl"
    if not mf or not os.path.exists(jl):
        print(ds, "not ready"); continue
    d = json.load(open(mf[-1])); a = d["eval/all_metrics_dict"]
    keys = [str(k) for k in a["motion_keys"]]; term = np.asarray(a["terminated"]).astype(bool); prog = np.asarray(a["progress"])
    rec = {}
    for l in open(jl):
        r = json.loads(l); rec.setdefault(r["motion"], r)
    st = {r["clip"]: r for r in screen[ds]}
    n = len(keys)
    print(f"== {ds} [{tag}]: success {d['eval/success/success_rate']:.3f} ({n - int(term.sum())}/{n})  progress {d['eval/success/progress_rate']:.3f}")
    rows = []
    for i, k in enumerate(keys):
        s = st.get(k)
        if s is None: continue
        t = rec[k]["step"] / 30.0 if (term[i] and k in rec) else None
        hl = human_lean(ds, k)
        rows.append(dict(clip=k, term=bool(term[i]), t=t, fired=",".join(rec.get(k, {}).get("fired", [])), lean=s["lean"], hlean=hl, f0=s["f0"], spd0=s["spd0"]))
    for lim in (1.0, 3.0):
        e = [r for r in rows if r["t"] is not None and r["t"] < lim]
        print(f"   early failures (< {lim:.0f} s): {len(e)} / {n}   reasons: { {f: sum(1 for r in e if r['fired'] == f) for f in set(r['fired'] for r in e)} }")
    e1 = [r for r in rows if r["t"] is not None and r["t"] < 1.0]
    print("   early (<1 s) clips: clip | t_fail | robot lean / human lean (first 2 s) | f0 foot z | first-frame speed")
    for r in sorted(e1, key=lambda r: r["t"]):
        print(f"      {r['clip'][-14:]:14s} {r['t']:5.2f}s  lean {r['lean']:5.1f} / {r['hlean']:5.1f}  f0 {r['f0']:.3f}  spd0 {r['spd0']:5.1f}  {r['fired']}")
    # signature = robot torso leans much more than the human does
    sig = [r for r in rows if r["lean"] - r["hlean"] > 20]
    print(f"   'bowed' signature (robot lean - human lean > 20 deg): {len(sig)} clips; of those terminated early (<3 s): {sum(1 for r in sig if r['t'] is not None and r['t'] < 3)}, terminated at all: {sum(1 for r in sig if r['term'])}")
    print("   signature clips:", [(r["clip"][-14:], round(r["lean"]), round(r["hlean"]), None if r["t"] is None else round(r["t"], 1)) for r in sorted(sig, key=lambda r: -(r['lean'] - r['hlean']))[:10]])
    nsig = [r for r in rows if r["lean"] - r["hlean"] <= 20]
    print(f"   early (<3 s) rate: with signature {sum(1 for r in sig if r['t'] is not None and r['t'] < 3)}/{len(sig)}  | without {sum(1 for r in nsig if r['t'] is not None and r['t'] < 3)}/{len(nsig)}")
