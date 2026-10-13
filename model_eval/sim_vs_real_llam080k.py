#!/usr/bin/env python3
"""Sim-vs-real per-clip comparison for llam_080k.

SIM : GR00T-WholeBodyControl/logs_eval/20260921_121122-EVAL_subset_LLAM080k/metrics_eval.json
      (eval/all_metrics_dict: per-clip arrays over the 160 eval_subset clips, keyed by `motion_keys`)
REAL: sim2real/online_eval_results_20261001.json (g1_run_1001, same checkpoint, same clips)
"""
import json, os, sys
import numpy as np

SIM = "/home/grease/GR00T-WholeBodyControl/logs_eval/20260921_121122-EVAL_subset_LLAM080k/metrics_eval.json"
REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim2real", "online_eval_results_20261001.json")

d = json.load(open(SIM))
a = d["eval/all_metrics_dict"]
print("sim keys:", [k for k in a][:40])
keys = [str(k) for k in a["motion_keys"]]
print("n sim clips:", len(keys), "| example key:", keys[0])
term = np.asarray(a["terminated"]).astype(bool)
prog = np.asarray(a["progress"]) if "progress" in a else None


def sim_row(clip):
    idx = [i for i, k in enumerate(keys) if clip in k]
    if not idx:
        return None
    i = idx[0]
    g = lambda m: float(np.asarray(a[m])[i])
    return dict(key=keys[i], terminated=bool(term[i]), l=g("mpjpe_l"), pa=g("mpjpe_pa"), gl=g("mpjpe_g"),
                vel=g("vel_dist"), acc=g("accel_dist"), legs=g("mpjpe_l_legs"), foot=g("mpjpe_l_foot"),
                upper=g("mpjpe_l_other_upper_bodies"))


real = json.load(open(REAL))
rows = {}
for e in real["episodes"]:
    rows.setdefault(e["clip"], []).append(e)

print(f"\n{'clip':44s} {'':5s}{'L':>7s}{'PA':>7s}{'legs':>7s}{'upper':>7s}{'vel':>7s}{'acc':>7s}  term")
tot = []
for clip, eps in rows.items():
    s = sim_row(clip)
    if s is None:
        print(f"{clip:44s} NOT IN SIM EVAL"); continue
    print(f"{clip:44s} sim  {s['l']:7.1f}{s['pa']:7.1f}{s['legs']:7.1f}{s['upper']:7.1f}{s['vel']:7.2f}{s['acc']:7.2f}  {s['terminated']}")
    im = [e["imitation"] for e in eps]
    m = lambda k: float(np.mean([x[k] for x in im]))
    print(f"{'':44s} real {m('mpjpe_l'):7.1f}{m('mpjpe_pa'):7.1f}{m('mpjpe_l_legs'):7.1f}{m('mpjpe_l_other_upper_bodies'):7.1f}"
          f"{m('vel_dist'):7.2f}{m('accel_dist'):7.2f}  n={len(eps)} conf={sum(e['match']['confirmed'] for e in eps)}")
    print(f"{'':44s} gap  {m('mpjpe_l')-s['l']:+7.1f}{m('mpjpe_pa')-s['pa']:+7.1f}{m('mpjpe_l_legs')-s['legs']:+7.1f}"
          f"{m('mpjpe_l_other_upper_bodies')-s['upper']:+7.1f}{m('vel_dist')-s['vel']:+7.2f}{m('accel_dist')-s['acc']:+7.2f}")
    tot.append((clip, s, {k: m(k) for k in ('mpjpe_l','mpjpe_pa','mpjpe_l_legs','mpjpe_l_other_upper_bodies','vel_dist','accel_dist')}, len(eps)))

print("\nmacro-mean over clips (each clip weighted equally):")
for nm, sk, rk in (("mpjpe_l", "l", "mpjpe_l"), ("mpjpe_pa", "pa", "mpjpe_pa"), ("legs", "legs", "mpjpe_l_legs"),
                   ("upper", "upper", "mpjpe_l_other_upper_bodies"), ("vel_dist", "vel", "vel_dist"),
                   ("accel_dist", "acc", "accel_dist")):
    S = np.mean([s[sk] for _, s, _, _ in tot]); R = np.mean([r[rk] for _, _, r, _ in tot])
    print(f"  {nm:10s} sim {S:7.2f}  real {R:7.2f}  gap {R-S:+7.2f}  ratio {R/S:5.2f}")

print("\nmacro-mean over the 4 CONFIRMED locomotion clips only:")
loc = [t for t in tot if t[0] in ("walk_180_R_003__A332_M","walk_backward_start_001__A030_M","jog_ff_start_180_R_002__A192_M","walk_sideway_045_stop_005__A042_M")]
for nm, sk, rk in (("mpjpe_l", "l", "mpjpe_l"), ("mpjpe_pa", "pa", "mpjpe_pa"), ("legs", "legs", "mpjpe_l_legs"),
                   ("upper", "upper", "mpjpe_l_other_upper_bodies"), ("vel_dist", "vel", "vel_dist"), ("accel_dist", "acc", "accel_dist")):
    S = np.mean([s[sk] for _, s, _, _ in loc]); R = np.mean([r[rk] for _, _, r, _ in loc])
    print(f"  {nm:10s} sim {S:7.2f}  real {R:7.2f}  gap {R-S:+7.2f}  ratio {R/S:5.2f}")
