import glob
import json
import os
from collections import Counter

os.chdir(os.path.expanduser("~/GR00T-WholeBodyControl/logs_eval"))
V = ["baseline", "anchor_pos", "anchor_ori", "ee_body_pos", "foot_pos_xyz", "all"]
man = {c["name"]: c for c in json.load(open(os.path.expanduser("~/ego_dataset/ICL_hardset/manifest.json")))["clips"]}
res = {}
for v in V:
    for d in sorted(glob.glob(f"*EVAL_iclabl_{v}"), reverse=True):
        try:
            j = json.load(open(d + "/metrics_eval.json"))
        except Exception:
            continue
        a = j["eval/all_metrics_dict"]
        res[v] = {"s": j["eval/success/success_rate"], "p": j["eval/success/progress_rate"],
                  "clip": {n: (int(a["terminated"][i]), float(a["progress"][i]), float(a["mpjpe_l"][i]), float(a["mpjpe_g"][i]))
                           for i, n in enumerate(a["motion_keys"])}}
        break
print(f"{'variant (term relaxed)':24s} {'success':>8s} {'progress':>9s}   clips rescued vs baseline")
base = res["baseline"]["clip"]
for v in V:
    if v not in res:
        print(f"{v:24s} (no metrics yet)")
        continue
    r = res[v]
    resc = [n for n, c in r["clip"].items() if base[n][0] and not c[0]]
    print(f"{v:24s} {r['s']:8.3f} {r['p']:9.3f}   {len(resc)}")
singles = ["anchor_pos", "anchor_ori", "ee_body_pos", "foot_pos_xyz"]
print("\nbinding termination per clip (failed at baseline; term whose relaxation lets the clip run longer):")
cnt = Counter()
rows = []
for n, c in sorted(base.items()):
    if not c[0]:
        continue
    gains = {v: res[v]["clip"][n][1] - c[1] for v in singles if v in res}
    best = max(gains, key=gains.get) if gains else None
    tag = best if gains and gains[best] > 0.05 else "none single (others/robustness)"
    cnt[tag] += 1
    rows.append((n, man[n]["dur_s"], c[1], tag, {k: round(g, 2) for k, g in gains.items()}))
for n, dur, p, tag, g in rows:
    print(f"  {n[:46]:46s} dur {dur:5.1f}s base progress {p:4.2f}  -> {tag:34s} {g}")
print("\ncount by binding term:", dict(cnt))
if "all" in res:
    a = res["all"]["clip"]
    print("\nall terminations relaxed: mean progress %.3f; clips still 'terminated': %d" % (res["all"]["p"], sum(c[0] for c in a.values())))
    big = sorted(a.items(), key=lambda kv: -kv[1][3])[:8]
    print("largest mpjpe_g with everything relaxed (robot wandered off):", [(n[:30], round(c[3])) for n, c in big])
