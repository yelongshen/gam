import glob, json, os, sys
from pathlib import Path
import numpy as np, joblib
tag = sys.argv[1] if len(sys.argv) > 1 else "novr050k"
screen = json.load(open("/tmp/screen_retarget_start.json"))
def human_lean(ds, clip):
    s = joblib.load(f"/home/grease/ego_dataset/{ds}/smpl/{clip}.pkl"); s = s[next(iter(s))] if "smpl_joints" not in s else s
    J = np.asarray(s["smpl_joints"], float)[:100]; v = J[:, 9] - J[:, 0]
    return float(np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))).mean())
allrows = []
for ds in ("picoset_20260928", "picoset_20260929", "picoset_20260930"):
    mf = sorted(glob.glob(f"/home/grease/GR00T-WholeBodyControl/logs_eval/*EVAL_early_{ds}_{tag}/metrics_eval.json"))[-1]
    a = json.load(open(mf))["eval/all_metrics_dict"]; keys = [str(k) for k in a["motion_keys"]]; term = np.asarray(a["terminated"]).astype(bool)
    rec = {}
    for l in open(f"/tmp/terms_early_{ds}_{tag}.jsonl"):
        r = json.loads(l); rec.setdefault(r["motion"], r)
    st = {r["clip"]: r for r in screen[ds]}
    for i, k in enumerate(keys):
        s = st[k]; t = rec[k]["step"] / 30.0 if term[i] else None
        allrows.append(dict(ds=ds, clip=k, t=t, early=(t is not None and t < 3.0), lean=s["lean"], hl=human_lean(ds, k), spd0=s["spd0"], f0=s["f0"], fired=",".join(rec.get(k, {}).get("fired", []))))
def rate(rows, f):
    sel = [r for r in rows if f(r)]; rest = [r for r in rows if not f(r)]
    return len(sel), sum(r["early"] for r in sel), len(rest), sum(r["early"] for r in rest)
tests = {"A  bowed IK (robot lean - human lean > 20 deg)": lambda r: r["lean"] - r["hl"] > 20,
         "B  first-frame ref speed > 10 rad/s": lambda r: r["spd0"] > 10,
         "B' first-frame ref speed > 6 rad/s": lambda r: r["spd0"] > 6,
         "C  frame-0 lowest ankle > 0.10 m": lambda r: r["f0"] > 0.10,
         "A or B": lambda r: (r["lean"] - r["hl"] > 20) or r["spd0"] > 10}
print(f"{'':52s}" + "".join(f"{ds[-5:]:>26s}" for ds in ("picoset_20260928", "picoset_20260929", "picoset_20260930")) + f"{'ALL':>26s}")
print(f"{'early failure (<3 s) = flagged: n_early/n_flagged | unflagged: n_early/n':52s}")
for nm, f in tests.items():
    cells = []
    for ds in ("picoset_20260928", "picoset_20260929", "picoset_20260930", None):
        rows = [r for r in allrows if ds is None or r["ds"] == ds]
        n, e, m, e2 = rate(rows, f)
        cells.append(f"{e:3d}/{n:3d} | {e2:3d}/{m:3d}")
    print(f"{nm:52s}" + "".join(f"{c:>26s}" for c in cells))
# what is left unexplained among early failures (<3 s)?
for ds in ("picoset_20260928", "picoset_20260929", "picoset_20260930"):
    rows = [r for r in allrows if r["ds"] == ds]; ear = [r for r in rows if r["early"]]
    unexp = [r for r in ear if not ((r["lean"] - r["hl"] > 20) or r["spd0"] > 10)]
    print(f"{ds}: early(<3 s) {len(ear)}/{len(rows)}; explained by A or B: {len(ear) - len(unexp)}; unexplained: {len(unexp)}")
    print("    unexplained:", [(r["clip"][-8:], round(r["t"], 2), round(r["spd0"], 1), round(r["f0"], 2), r["fired"]) for r in sorted(unexp, key=lambda r: r["t"])[:14]])
# reason of failure for first-frame spikes
sp = [r for r in allrows if r["early"] and r["spd0"] > 10]
import collections
print("early failures with first-frame speed > 10:", len(sp), "fired:", dict(collections.Counter(r["fired"] for r in sp)))
json.dump([{k: v for k, v in r.items()} for r in allrows], open("/tmp/early_join_rows.json", "w"))
