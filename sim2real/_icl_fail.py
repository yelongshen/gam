import glob
import json
import os

import numpy as np

os.chdir(os.path.expanduser("~/GR00T-WholeBodyControl/logs_eval"))
man = {c["name"]: c for c in json.load(open(os.path.expanduser("~/ego_dataset/ICL_hardset/manifest.json")))["clips"]}


def load(tag):
    d = sorted(glob.glob(f"*EVAL_icl_{tag}"))
    for x in reversed(d):
        try:
            return json.load(open(x + "/metrics_eval.json"))["eval/all_metrics_dict"]
        except Exception:
            pass


runs = {t: load(t) for t in ["base100k", "mixv3b_014000", "mixv3b_020000"]}
runs = {k: v for k, v in runs.items() if v}
names = sorted(man)
print("runs:", list(runs))
tab = {}
for t, a in runs.items():
    for i, n in enumerate(a["motion_keys"]):
        tab.setdefault(n, {})[t] = (int(a["terminated"][i]), float(a["progress"][i]), float(a["mpjpe_l"][i]))
fails_all = [n for n in names if n in tab and all(tab[n][t][0] for t in runs)]
fails_any = [n for n in names if n in tab and any(tab[n][t][0] for t in runs)]
print(f"clips failing in ALL runs: {len(fails_all)}   in ANY run: {len(fails_any)}   of {len(names)}")
print(f"\n{'clip':52s} {'dur':>5s}  " + "  ".join(f"{t[-9:]:>16s}" for t in runs) + "   fail time (s) base")
for n in fails_any:
    d = man[n]["dur_s"]
    cols = []
    for t in runs:
        term, prog, mp = tab[n][t]
        cols.append(f"{'X' if term else 'ok'} p={prog:4.2f} {mp:5.0f}mm")
    t0 = tab[n]["base100k"]
    print(f"{n[:52]:52s} {d:5.1f}  " + "  ".join(f"{c:>16s}" for c in cols) + f"   {t0[1] * d:5.1f}" if t0[0] else f"{n[:52]:52s} {d:5.1f}  " + "  ".join(f"{c:>16s}" for c in cols))
# failure time distribution (base)
ft = [(tab[n]["base100k"][1] * man[n]["dur_s"], man[n]["dur_s"], n) for n in names if n in tab and tab[n]["base100k"][0]]
t = np.array([x[0] for x in ft])
print(f"\nbase100k: {len(ft)} failed clips; failure time (s) min {t.min():.1f} median {np.median(t):.1f} max {t.max():.1f}; "
      f"fraction failing before 1 s: {(t < 1).mean():.2f}, before 2 s: {(t < 2).mean():.2f}, before 4 s: {(t < 4).mean():.2f}")
by = {}
for n in names:
    if n in tab:
        by.setdefault((man[n]["source"], man[n]["split"]), []).append(tab[n]["base100k"][0])
print("failure rate by source (base100k):", {f"{k[0]}/{k[1]}": f"{sum(v)}/{len(v)}" for k, v in by.items()})
