import glob, json, sys, os
import numpy as np

def read_metrics(pattern):
    matches = sorted(glob.glob(f"/home/grease/GR00T-WholeBodyControl/logs_eval/{pattern}/metrics_eval.json"))
    if not matches:
        return None
    d = json.load(open(matches[-1]))
    succ = d.get("eval/success/success_rate", float("nan"))
    prog = d.get("eval/success/progress_rate", float("nan"))
    l = d.get("eval/success/mpjpe_l", float("nan"))
    g = d.get("eval/success/mpjpe_g", float("nan"))
    pa = d.get("eval/success/mpjpe_pa", float("nan"))
    all_dict = d.get("eval/all_metrics_dict", {})
    term = np.asarray(all_dict.get("terminated", [])).astype(bool)
    n = len(term)
    return dict(succ=succ, prog=prog, l=l, g=g, pa=pa, n=n, n_term=int(term.sum()))

print(f"{'evaluation':46s}{'success':>10s}{'progress':>10s}{'mpjpe_l':>9s}{'mpjpe_g':>9s}{'n_clips':>8s}")
runs = [
    ("0928 (novr050k, motion-only)", "*EVAL_early_picoset_20260928_novr050k_rt"),
    ("0929 (novr050k, motion-only)", "*EVAL_early_picoset_20260929_novr050k_rt"),
    ("0930 (novr050k, motion-only)", "*EVAL_early_picoset_20260930_novr050k_rt"),
    ("1002 sit (sitmixv2_4k, chair)", "*EVAL_picoset1002sit_chair_sitmixv2_rt"),
    ("1002 sit (novr050k base, chair)", "*EVAL_picoset1002sit_chair_novr050k_rt"),
]
for title, pat in runs:
    m = read_metrics(pat)
    if m is None:
        print(f"{title:46s} NOT FOUND")
    else:
        print(f"{title:46s}{m['succ']:10.3f}{m['prog']:10.3f}{m['l']:9.1f}{m['g']:9.1f}{m['n']:8d}")
