#!/usr/bin/env python3
"""Settle-test report: failure second on an ICL settle set (first frame held, then the motion).

Per-clip hold length comes from <SET>/settle_info.json (`motion_start_s`), so sets with mixed holds (e.g.
ICL_hardset_settle: 2 s for most clips, 0.2 s for the unholdable ones) are classified correctly.
Env: SET=<dir> (default ICL_hardset_settle)  PREFIX=<eval_name prefix> (default iclsettlev2; old 2 s set: SET=.../ICL_hardset_settle2 PREFIX=iclsettle)  OUT=<md file>.

For each clip: did the policy hold the start pose for the whole settle period?  If not it fails at t < 2 s (never reached the
motion). If it holds, failure second minus 2 = seconds into the real motion. Compared with the old failure second (no hold).
Usage: python sim2real/icl_settle_report.py
"""
import glob
import json
import os

import numpy as np

LOGS = os.path.expanduser("~/GR00T-WholeBodyControl/logs_eval")
SET = os.path.expanduser(os.environ.get("SET", "~/ego_dataset/ICL_hardset_settle"))
PREFIX = os.environ.get("PREFIX", "iclsettlev2")
HERE = os.path.dirname(os.path.abspath(__file__))
info = {i["name"]: i for i in json.load(open(f"{SET}/settle_info.json"))}
old = {}
for row in __import__("csv").DictReader(open(f"{HERE}/icl_hard_set/icl_failure_seconds.csv")):
    old[row["clip"]] = row
RUNS = {k: f"{PREFIX}_{k}_baseline" for k in ("base100k", "mixv3b_010000", "mixv3b_014000", "mixv3b_020000")}


def load(tag):
    for d in sorted(glob.glob(f"{LOGS}/*EVAL_{tag}"), reverse=True):
        try:
            a = json.load(open(d + "/metrics_eval.json"))["eval/all_metrics_dict"]
            return {n: (int(a["terminated"][i]), float(a["progress"][i])) for i, n in enumerate(a["motion_keys"])}
        except Exception:
            pass


res = {k: load(v) for k, v in RUNS.items()}
res = {k: v for k, v in res.items() if v}
print("runs with results:", list(res))
rows = []
for n, i in info.items():
    tot = i["robot_frames"] / 30.0
    SETTLE = float(i["motion_start_s"])
    r = {"clip": n, "motion_start_s": SETTLE}
    cls = []
    for k, d in res.items():
        term, prog = d[n]
        fs = prog * tot
        if not term:
            r[k] = "ok"
            cls.append("ok")
        elif fs < SETTLE:
            r[k] = f"{fs:.2f} (HOLD)"
            cls.append("hold")
        else:
            r[k] = f"{fs:.2f} (+{fs - SETTLE:.2f})"
            cls.append("motion")
    r["class"] = max(set(cls), key=cls.count)
    r["old_fail_s"] = old.get(n, {}).get("fail_s_base100k", "")
    rows.append(r)
cnt = {k: {"hold": 0, "motion": 0, "ok": 0} for k in res}
for r in rows:
    for k in res:
        v = r[k]
        cnt[k]["ok" if v == "ok" else ("hold" if "HOLD" in v else "motion")] += 1
print("\nper run: clips that fail DURING their hold / fail after the hold (in motion) / survive the whole clip")
for k, c in cnt.items():
    print(f"  {k:16s} hold-fail {c['hold']:2d}   motion-fail {c['motion']:2d}   ok {c['ok']:2d}")
order = {"hold": 0, "motion": 1, "ok": 2}
rows.sort(key=lambda r: (order[r["class"]], r["clip"]))
cols = ["clip", "motion_start_s", "class", "old_fail_s"] + list(res)
OUT = os.environ.get("OUT", f"{HERE}/icl_settle_results.md" if os.path.basename(SET) == "ICL_hardset_settle2"
                     else f"{HERE}/icl_settle_results_{os.path.basename(SET)}.md")
with open(OUT, "w") as fh:
    fh.write(f"# ICL hard set with settle (first frame held) - {os.path.basename(SET)}\n\n")
    fh.write(f"`{os.path.basename(SET)}` = `ICL_hardset_aligned` with the first frame held before the motion "
             "(`motion_start_s` column; built by `data_process/build_settle_set.py` / `sim2real/build_grounded_settle_set.py`). **(HOLD)** = the policy failed during the hold, i.e. it could not stay stable at the "
             "start pose; **(+x)** = failed x seconds into the real motion; `ok` = ran to the end. `old_fail_s` = failure second "
             "without the hold (base100k).\n\n")
    fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    for r in rows:
        fh.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
    fh.write("\n| run | fail during hold | fail in motion | ok |\n|---|---|---|---|\n")
    for k, c in cnt.items():
        fh.write(f"| {k} | {c['hold']} | {c['motion']} | {c['ok']} |\n")
print(f"\nwrote {OUT}")
for r in rows[:40]:
    print(f"  {r['clip'][:44]:44s} {r['class']:6s} old {str(r['old_fail_s']):>5s} | " + " | ".join(f"{r[k]:>14s}" for k in res))
