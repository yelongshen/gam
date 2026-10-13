"""One-off: insert the picoset_20260924 cross-check section into
GR00T-WholeBodyControl/model_eval/checkpoint_comparison.md.

Kept as a file (rather than a heredoc) because the target doc lives outside the
gam workspace, so the editor tooling cannot touch it directly.
"""
import json
import glob
import os

DOC = "/home/grease/GR00T-WholeBodyControl/model_eval/checkpoint_comparison.md"
LOGS = "/home/grease/GR00T-WholeBodyControl/logs_eval"

# pico_evalset reference numbers, copied from the doc's own table.
BASE = {
    "LOW_LATENCY": (0.650, 0.814, 362.79, 30.15, 23.76),
    "novr_050k":   (0.700, 0.829, 341.78, 29.73, 23.69),
    "novrll_020k": (0.750, 0.862, 410.74, 33.25, 26.72),
    "novrll_038k": (0.700, 0.830, 310.98, 32.05, 25.28),
    "novrll_052k": (0.800, 0.907, 392.28, 31.56, 24.91),
    "novrll_062k": (0.750, 0.859, 379.93, 31.01, 24.13),
    "novrll_074k": (0.700, 0.817, 307.38, 29.05, 23.03),
    "LLAM030k":    (0.800, 0.894, 295.50, 31.37, 24.40),
    "LLAM050k":    (0.800, 0.895, 386.75, 31.87, 24.17),
    "LLAM080k":    (0.800, 0.910, 386.34, 30.00, 23.64),
}
LABEL = {
    "LOW_LATENCY": "LOW_LATENCY", "novr_050k": "REUBEN novr_050k",
    "novrll_020k": "REUBEN novrll_020k", "novrll_038k": "REUBEN novrll_038k",
    "novrll_052k": "REUBEN novrll_052k", "novrll_062k": "REUBEN novrll_062k",
    "novrll_074k": "REUBEN novrll_074k", "LLAM030k": "REUBEN novrllam_030k",
    "LLAM050k": "REUBEN novrllam_050k", "LLAM080k": "REUBEN novrllam_080k",
}


def load(tag):
    ds = sorted(glob.glob(f"{LOGS}/*EVAL_picoset20260924_{tag}"))
    d = json.load(open(os.path.join(ds[-1], "metrics_eval.json")))
    return (d["eval/success/success_rate"], d["eval/success/progress_rate"],
            d["eval/success/mpjpe_g"], d["eval/success/mpjpe_l"],
            d["eval/success/mpjpe_pa"]), os.path.basename(ds[-1])


def main():
    rows, logdirs = {}, {}
    for tag in BASE:
        rows[tag], logdirs[tag] = load(tag)

    n = [rows[t] for t in BASE]
    o = [BASE[t] for t in BASE]
    mean = lambda xs, i: sum(x[i] for x in xs) / len(xs)  # noqa: E731

    tbl = [
        "### `picoset_20260924` (93 clips) — cross-check of `pico_evalset`",
        "",
        "A larger PICO teleop eval set built from the 2026-09-24 capture",
        "(`ego_dataset/picoset_20260924`, 93 clips vs `pico_evalset`'s 20), used here to",
        "check whether the `pico_evalset` column is stable or an artifact of its small size.",
        "Ten checkpoints spanning the baselines and all three REUBEN families were re-run.",
        "",
        "| Checkpoint | Succ | Prog | MPJPE (G) | MPJPE (L) | MPJPE (PA) | | `pico_evalset` Succ | Prog | G | L | PA |",
        "|---|:---:|:---:|:---:|:---:|:---:|:-:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for t in BASE:
        a, b = rows[t], BASE[t]
        tbl.append(
            f"| {LABEL[t]} | {a[0]:.3f} | {a[1]:.3f} | {a[2]:.1f} | {a[3]:.2f} | {a[4]:.2f} "
            f"| | {b[0]:.3f} | {b[1]:.3f} | {b[2]:.1f} | {b[3]:.2f} | {b[4]:.2f} |")
    tbl.append(
        f"| **MEAN** | **{mean(n,0):.3f}** | **{mean(n,1):.3f}** | **{mean(n,2):.1f}** | "
        f"**{mean(n,3):.2f}** | **{mean(n,4):.2f}** | | {mean(o,0):.3f} | {mean(o,1):.3f} | "
        f"{mean(o,2):.1f} | {mean(o,3):.2f} | {mean(o,4):.2f} |")

    tbl += [
        "",
        "**Rank agreement between the two sets** (Spearman rho over the 10 checkpoints):",
        "",
        "| Metric | rho | p |",
        "|---|:---:|:---:|",
        "| `mpjpe_l` | **+0.92** | 0.000 |",
        "| `mpjpe_pa` | **+0.84** | 0.002 |",
        "| progress rate | +0.60 | 0.067 |",
        "| success rate | +0.57 | 0.085 |",
        "| `mpjpe_g` | **-0.20** | 0.580 |",
        "",
        "**Findings**",
        "",
        "1. **`mpjpe_l` / `mpjpe_pa` transfer.** rho = 0.92 / 0.84, and the absolute levels",
        "   agree closely (means 33.02 vs 31.00 mm L, 24.54 vs 24.37 mm PA). Checkpoint",
        "   ranking by tracking precision is reproduced on 4.6x more clips.",
        "2. **`novrllam_080k` is best on both sets**, and the LLAM 030k -> 050k -> 080k",
        "   ordering reproduces. The recommendation in the Summary section holds.",
        "3. **Success/progress are systematically lower** (-11 pts / -7 pts) but still",
        "   rank-correlated (rho ~ 0.6). The new set is genuinely harder: it is cut from",
        "   continuous teleop and includes many short/transitional clips, where",
        "   `pico_evalset` is 20 hand-picked ones.",
        "4. **Do not use `mpjpe_g` on any PICO-derived set.** rho = -0.20, i.e. no",
        "   transfer. Both sets store `transl = 0` (the PICO capture records no pelvis",
        "   world position -- only head/hand VR anchors), so global MPJPE measures root",
        "   drift against a stationary reference and behaves like noise. `pico_evalset`",
        "   shows the same pathology: its G column swings 295-411 mm with no relation to",
        "   checkpoint quality. Rank on L / PA instead.",
        "",
        "> **Caveat inherited from the data, not the eval.** Every clip in both PICO sets is",
        "> **in-place** (robot root travels ~1 cm median, vs 26 cm on `eval_subset`), so",
        "> neither set measures locomotion -- the ~60 walking clips have the legs cycling",
        "> while the base stays put. They are valid for upper-body / in-place pose tracking",
        "> only.",
        "",
        "Build script: `gam/data_process/build_picoset_20260924.py` (applies the robot-fps30",
        "vs SMPL-fps50 frame alignment from `dev_notes/fps_check_alignment`; 81 of 93 pairs",
        "needed a 1-frame trim). Eval runner: `gam/model_eval/run_picoset20260924_evals.sh`.",
        "",
    ]

    doc = open(DOC).read()
    anchor = "## Evaluating the `novrllwm` (critic world-model) checkpoint"
    assert anchor in doc and "picoset_20260924" not in doc
    doc = doc.replace(anchor, "\n".join(tbl) + "\n" + anchor, 1)

    # dataset list + source-log table
    doc = doc.replace(
        "- **pico_evalset**: 20 clips (`ego_dataset/pico_evalset`)",
        "- **pico_evalset**: 20 clips (`ego_dataset/pico_evalset`)\n"
        "- **picoset_20260924**: 93 clips (`ego_dataset/picoset_20260924`) — larger PICO\n"
        "  teleop set, used as a cross-check of `pico_evalset` (see its section below)", 1)
    logrows = "\n".join(
        f"| {LABEL[t]} | picoset_20260924 | `logs_eval/{logdirs[t]}` |" for t in BASE)
    doc = doc.replace("\n## Comparison Table", f"{logrows}\n\n## Comparison Table", 1)

    open(DOC, "w").write(doc)
    print(f"updated {DOC}")


if __name__ == "__main__":
    main()
