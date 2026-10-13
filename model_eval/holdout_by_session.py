#!/usr/bin/env python3
"""Leave-one-SESSION-out version of holdout_tau_tables.py.

The every-5th-run split in holdout_tau_tables.py put 6 sessions on BOTH sides of
the split. Runs from the same day share robot state (temperature, battery,
calibration), floor and operator, so a per-session offset learned from the
training runs scores on the held-out runs of that same day for free.

Here the held-out set is an entire session (default 0924) and training uses only
EARLIER sessions, so the fitted correction is judged on a day it has never seen
and cannot have seen the future of.

Known-bad runs are already dropped by deploy_constants.list_runs() (see
sim2real/robot_log_data_quality.md).

Usage:
  .venv_sim/bin/python model_eval/holdout_by_session.py               # 0924
  .venv_sim/bin/python model_eval/holdout_by_session.py --test 0922
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim2real"))
sys.path.insert(0, os.path.join(ROOT, "model_eval"))

from deploy_constants import JOINT_NAMES  # noqa: E402
from fit_friction_model import design, load_data  # noqa: E402
from holdout_tau_tables import table  # noqa: E402

MODEL = 6
LAM = 1e-3


def session_of(name):
    """'g1_run_0924/...' -> '0924'. Undated dirs get '0901' (oldest kept log)."""
    part = name.split("/")[0]
    if part.startswith("g1_run_"):
        return part[len("g1_run_"):]
    return "0901"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="0924", help="session (MMDD) to hold out")
    a = ap.parse_args()

    data = load_data(80, 5)
    for d in data:
        d["sess"] = session_of(d["name"])

    test = [d for d in data if d["sess"] == a.test]
    train = [d for d in data if d["sess"] < a.test]
    later = [d for d in data if d["sess"] > a.test]
    if not test or not train:
        raise SystemExit(f"need both sides: test={len(test)} train={len(train)}")

    print(f"held-out session : {a.test}  ({len(test)} runs, "
          f"{sum(len(d['q']) for d in test)} frames = "
          f"{sum(len(d['q']) for d in test)*0.02:.0f} s)")
    for d in test:
        print(f"      {d['name']}  ({len(d['q'])} frames)")
    print(f"train (earlier)  : {len(train)} runs, "
          f"{sum(len(d['q']) for d in train)} frames, sessions "
          f"{sorted(set(d['sess'] for d in train))}")
    if later:
        print(f"NOT used (later than {a.test}): {len(later)} runs, sessions "
              f"{sorted(set(d['sess'] for d in later))}")
    print("known-bad runs: already excluded (9 of 57, see robot_log_data_quality.md)\n")

    joints = list(range(29))
    est = {j: np.concatenate([d["tau"][:, j] for d in test]) for j in joints}
    pd = {j: np.concatenate([d["tau_pd"][:, j] for d in test]) for j in joints}
    fit, coefs = {}, {}
    for j in joints:
        X = np.vstack([design(d, j, MODEL) for d in train])
        y = np.concatenate([d["res"][:, j] for d in train])
        coef = np.linalg.solve(X.T @ X + LAM * np.eye(X.shape[1]), X.T @ y)
        coefs[j] = coef
        Xte = np.vstack([design(d, j, MODEL) for d in test])
        fit[j] = pd[j] + Xte @ coef

    t1 = table(f"TABLE 1   tau_est vs tau_pd        -- held-out session {a.test}",
               est, pd, joints)
    t2 = table(f"TABLE 2   tau_est vs tau_pd + M6   -- held-out session {a.test} "
               f"(fit on earlier sessions)", est, fit, joints)

    print("=" * 100)
    print("CHANGE FROM TABLE 1 TO TABLE 2")
    print("=" * 100)
    print(f"{'joint':22s}{'gap before':>12}{'gap after':>11}{'reduction':>11}"
          f"{'bias before':>13}{'bias after':>12}")
    print("-" * 82)
    worse = []
    for k, j in enumerate(joints):
        b, af = t1[k, 3], t2[k, 3]
        if af > b:
            worse.append(JOINT_NAMES[j])
        print(f"{JOINT_NAMES[j]:22s}{b:12.3f}{af:11.3f}{100*(b-af)/b:10.1f}%"
              f"{t1[k,4]:+13.3f}{t2[k,4]:+12.3f}")
    print("-" * 82)
    print(f"{'MEAN gap':22s}{t1[:,3].mean():12.3f}{t2[:,3].mean():11.3f}"
          f"{100*(t1[:,3].mean()-t2[:,3].mean())/t1[:,3].mean():10.1f}%")
    print(f"\njoints where the fit made the held-out gap WORSE: {len(worse)}/29"
          + (f"  -> {', '.join(worse)}" if worse else ""))

    # ---------------------------------------------------------------
    # Is the offset a property of the ROBOT (stable) or of the SESSION?
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("IS THE OFFSET STABLE ACROSS SESSIONS?   mean(tau_est - tau_pd) per session [Nm]")
    print("=" * 100)
    sess = sorted(set(d["sess"] for d in train))
    show = [0, 3, 4, 6, 9, 12, 15, 18, 22, 25]
    print(f"{'joint':20s}" + "".join(f"{s:>8}" for s in sess)
          + f"{'TEST '+a.test:>11}{'train spread':>14}{'test-train':>12}")
    print("-" * (20 + 8 * len(sess) + 37))
    for j in show:
        per = []
        for s in sess:
            r = np.concatenate([d["res"][:, j] for d in train if d["sess"] == s])
            per.append(r.mean())
        rt = np.concatenate([d["res"][:, j] for d in test]).mean()
        print(f"{JOINT_NAMES[j]:20s}" + "".join(f"{v:+8.3f}" for v in per)
              + f"{rt:+11.3f}{np.std(per):14.3f}{rt-np.mean(per):+12.3f}")
    print("\n  'train spread' = std of the per-session means. If it is small and")
    print("  the TEST value sits inside it, the offset is a stable property of the")
    print("  robot and the correction transfers. If TEST falls outside it, the")
    print("  offset is session-specific and should be measured per session.")


if __name__ == "__main__":
    main()
