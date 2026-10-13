import glob, json, os, sys
import numpy as np, joblib
screen = json.load(open("/tmp/screen_retarget_start.json"))          # AFTER the fix
fixed = {ds: set(l.strip() for l in open(f"/tmp/fix_{ds}.txt") if l.strip()) for ds in ("picoset_20260928", "picoset_20260929", "picoset_20260930")}
def load(ds, tag):
    mf = sorted(glob.glob(f"/home/grease/GR00T-WholeBodyControl/logs_eval/*EVAL_early_{ds}_{tag}/metrics_eval.json"))[-1]
    a = json.load(open(mf))["eval/all_metrics_dict"]
    keys = [str(k) for k in a["motion_keys"]]; term = np.asarray(a["terminated"]).astype(bool)
    rec = {}
    for l in open(f"/tmp/terms_early_{ds}_{tag}.jsonl"):
        r = json.loads(l); rec.setdefault(r["motion"], r)
    return {k: ((rec[k]["step"] / 30.0, ",".join(rec[k]["fired"]).replace("foot_pos_xyz", "foot").replace("anchor_ori_full", "ori").replace("ee_body_pos", "ee").replace("anchor_pos", "anc")) if term[i] else None) for i, k in enumerate(keys)}
def hl(ds, clip):
    s = joblib.load(f"/home/grease/ego_dataset/{ds}/smpl/{clip}.pkl"); s = s[next(iter(s))] if "smpl_joints" not in s else s
    J = np.asarray(s["smpl_joints"], float)[:100]; v = J[:, 9] - J[:, 0]
    return float(np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))).mean())
for ds in ("picoset_20260928", "picoset_20260929", "picoset_20260930"):
    after = load(ds, "novr050k_fix"); before = load(ds, "novr050k")
    st = {r["clip"]: r for r in screen[ds]}
    rows = [(after[k][0], k) for k in after if after[k] is not None and after[k][0] < 3.0]
    rows.sort()
    n1 = sum(1 for t, _ in rows if t < 1.0)
    print(f"\n=== {ds}: early failures AFTER the fix: <1 s: {n1}   <3 s: {len(rows)}   (of {len(after)} clips) ===")
    print(f"{'clip':>16s} {'t_fail':>6s} {'<1s':>3s} {'reason':>8s} | {'robot lean/human lean':>21s} {'f0 foot z':>9s} {'1st-frame spd':>13s} | {'re-retargeted':>13s} {'t_fail before':>13s}")
    for t, k in rows:
        s = st[k]; b = before[k]
        print(f"{k.replace('pico', ''):>16s} {t:6.2f} {'*' if t < 1.0 else ' ':>3s} {after[k][1]:>8s} | {s['lean']:9.1f} /{hl(ds, k):6.1f} {s['f0']:9.3f} {s['spd0']:13.1f} | {'yes' if k in fixed[ds] else 'no':>13s} {('%.2f s' % b[0]) if b else 'survived':>13s}")
