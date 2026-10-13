#!/usr/bin/env python3
"""Generate sim2real/icl_hardset_results.md from logs_eval/*EVAL_icl_* (ICL_hardset multi-attempt evals).
Usage: python sim2real/icl_report.py   (re-run after new evals finish)"""
import datetime
import glob
import json
import os
import re

import numpy as np

LOGS = os.path.expanduser("~/GR00T-WholeBodyControl/logs_eval")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icl_hardset_results.md")
BLOCK = 5


def load(d):
    try:
        return json.load(open(f"{LOGS}/{d}/metrics_eval.json"))
    except Exception:
        return None


def blocks(x):
    x = np.asarray(x, dtype=float)
    return [x[i:i + BLOCK].mean() for i in range(0, len(x), BLOCK)]


single, retry, retry3, seen = {}, {}, {}, {}
for d in sorted(os.listdir(LOGS)):
    m = re.match(r"^\d+_\d+-EVAL_(icl(?:rep3)?_.+)$", d)
    if not m or "VID" in d:
        continue
    name = m.group(1)
    seen.setdefault(name, False)
    j = load(d)
    if j is None:
        continue
    seen[name] = True
    short = re.sub(r"^icl_", "", name)
    if "retry" in j:
        (retry if j["retry"].get("attempts") == 30 else retry3)[short] = j["retry"]
    elif "eval/success/mpjpe_l" in j:
        single[name] = j
pending = sorted(n for n, ok in seen.items() if not ok)

L = []
w = L.append
w("# ICL hard set -- multi-attempt evaluation results\n")
w(f"_Generated {datetime.date.today()} by `sim2real/icl_report.py` from `GR00T-WholeBodyControl/logs_eval/*EVAL_icl_*`._\n")
w("## Setup\n")
w("- **Dataset:** `~/ego_dataset/ICL_hardset` -- 35 clips (13 LAFAN test windows, 11 LAFAN train windows, 9 AMASS clips that "
  "fail across checkpoints, 2 eval_subset flips). Built by `data_process/build_icl_hardset.py`. The 11 `lafan_train` clips come "
  "from `lafan1_trainset` (the policy may have trained on them).")
w("- **Protocol (`r30`):** every clip is attempted 30 times in a row in sim (headless, flat ground, no object, tracking-eval terminations). "
  "`keep` = the model's history carries over between attempts, `clr` = history cleared before each attempt.")
w("- **Metric:** success rate = fraction of the 35 clips that complete without early termination; progress rate = mean fraction of each clip "
  "completed. One clip = 2.9 percentage points of success rate.")
w("- **Checkpoint tags** are as named in the eval directories (`base10` is presumably the 100k base checkpoint; `hist25` / `hist50` are the "
  "history-conditioned models; `mixv3b` 14k / 20k are the mixpico v3b finetunes).\n")

w("## 1. Single-attempt evals (one rollout per clip)\n")
w("| run | success | progress | mpjpe_l (mm) | mpjpe_g (mm) |\n|---|---|---|---|---|")
for k, j in single.items():
    w(f"| {k} | {j['eval/success/success_rate']:.3f} | {j['eval/success/progress_rate']:.3f} | "
      f"{j.get('eval/success/mpjpe_l', float('nan')):.1f} | {j.get('eval/success/mpjpe_g', float('nan')):.1f} |")
w("\n`iclrep3_*` rows used 3 repeated attempts with the earlier harness (success counted differently, lower numbers); "
  "treat them separately.\n")

w("### 1b. 3-attempt runs (r3): success rate per attempt\n")
w("| run | attempt 1 | attempt 2 | attempt 3 |\n|---|---|---|---|")
for n, r in retry3.items():
    s = r["success_rate_per_attempt"]
    w(f"| {n} | " + " | ".join(f"{v:.3f}" for v in s) + " |")
w("")

for title, key in (("2. Success rate per block of 5 attempts (r30)", "success_rate_per_attempt"),
                   ("3. Progress rate per block of 5 attempts (r30)", "progress_rate_per_attempt")):
    w(f"## {title}\n")
    nb = len(blocks(next(iter(retry.values()))[key]))
    w("| run | " + " | ".join(f"{i * BLOCK + 1}-{(i + 1) * BLOCK}" for i in range(nb)) + " | last - first |")
    w("|---|" + "---|" * (nb + 1))
    for n, r in retry.items():
        b = blocks(r[key])
        w(f"| {n} | " + " | ".join(f"{v:.3f}" for v in b) + f" | {b[-1] - b[0]:+.3f} |")
    w("")

w("## 4. Keep minus clear (success rate, same checkpoint)\n")
nb = len(blocks(next(iter(retry.values()))["success_rate_per_attempt"]))
w("| checkpoint | " + " | ".join(f"{i * BLOCK + 1}-{(i + 1) * BLOCK}" for i in range(nb)) + " | mean |")
w("|---|" + "---|" * (nb + 1))
for n in retry:
    if n.endswith("_keep") and n[:-5] + "_clr" in retry:
        d = np.array(blocks(retry[n]["success_rate_per_attempt"])) - np.array(blocks(retry[n[:-5] + "_clr"]["success_rate_per_attempt"]))
        w(f"| {n[:-5]} | " + " | ".join(f"{v:+.3f}" for v in d) + f" | {d.mean():+.3f} |")

w("\n## Findings\n")
w("- **No checkpoint improves with attempts.** Within each run the 5-attempt block means stay within about +-0.03 of each other "
  "(about one clip) with no consistent direction; the last block minus the first block ranges from -0.034 to +0.011 in success rate.")
w("- **Keeping history does not help.** Kept runs are flat or falling and sit at or below the cleared run of the same checkpoint in most "
  "blocks; `hist50` keep falls from 0.057 to 0.023.")
w("- **Attempt 1 is a high draw.** For `base10` attempt 1 is 0.171 but the mean of attempts 1-5 is 0.126.")
w("- **The base checkpoint is the strongest on this set**; the history-conditioned models are at or below it.\n")
w("## Caveats\n")
w("- 35 clips: differences of one or two clips are within noise; blocks of 5 attempts average out sim noise but not clip selection.")
w("- The set was chosen from clips that fail in earlier evaluations, so absolute success rates are low by construction.")
w("- `lafan_train` clips may have been seen in training; split them out before drawing conclusions about generalization.")
w("- Per-clip outcomes are not stored in the r30 files (only per-attempt aggregates), so which clips ever succeed is not recoverable from them.\n")
if pending:
    w("## Runs without metrics yet (started, no `metrics_eval.json`)\n")
    for d in pending:
        w(f"- `{d}`")
    w("")
open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT}: {len(single)} single-attempt runs, {len(retry)} r30 runs, {len(pending)} pending")
