"""One-off: add the picoset_20260928 eval results to checkpoint_comparison.md.

The doc lives outside the agent workspace, so it is patched by script.
"""
import re

DOC = "/home/grease/GR00T-WholeBodyControl/model_eval/checkpoint_comparison.md"

with open(DOC) as f:
    txt = f.read()

# ---------------------------------------------------------------- datasets list
old_ds = """- **picoset_20260924**: 93 clips (`ego_dataset/picoset_20260924`) — larger PICO
  teleop set, used as a cross-check of `pico_evalset` (see its section below)
"""
new_ds = old_ds + """- **picoset_20260928**: 27 clips (`ego_dataset/picoset_20260928`) — first PICO set
  with **real root translation** (`--record_root_pos`), so unlike every earlier
  PICO set it actually measures locomotion (see its section below)
"""
assert txt.count(old_ds) == 1
txt = txt.replace(old_ds, new_ds)

# ------------------------------------------------------------ source log table
old_row = "| REUBEN novrllam_080k | picoset_20260924 | `logs_eval/20260927_183720-EVAL_picoset20260924_LLAM080k` |\n"
new_rows = old_row + """| LOW_LATENCY (novrll_074k) | picoset_20260928 | `logs_eval/20260928_183756-EVAL_picoset20260928_low_latency_74k` |
| REUBEN novr_050k | picoset_20260928 | `logs_eval/20260928_183836-EVAL_picoset20260928_novr_50k` |
| REUBEN novrll_062k | picoset_20260928 | `logs_eval/20260928_183916-EVAL_picoset20260928_novr_ll_62k` |
| REUBEN novrllam_080k | picoset_20260928 | `logs_eval/20260928_183957-EVAL_picoset20260928_novr_ll_ams_80k` |
| REUBEN novrllam_100k | picoset_20260928 | `logs_eval/20260928_184038-EVAL_picoset20260928_novr_ll_ams_100k` |
"""
assert txt.count(old_row) == 1
txt = txt.replace(old_row, new_rows)

# --------------------------------------------------------- new results section
anchor = """Build script: `gam/data_process/build_picoset_20260924.py` (applies the robot-fps30
vs SMPL-fps50 frame alignment from `dev_notes/fps_check_alignment`; 81 of 93 pairs
needed a 1-frame trim). Eval runner: `gam/model_eval/run_picoset20260924_evals.sh`.
"""
assert txt.count(anchor) == 1

section = anchor + """
### `picoset_20260928` (27 clips) — first PICO set with REAL root translation

Built from the 2026-09-28 capture, which is the first PICO session recorded with
`--record_root_pos`. That flag stores `body_pos_w` (the world pelvis position)
in every raw frame, so `transl` is no longer forced to zero and the clips carry
genuine locomotion:

| Set | robot root xy-path (median) | mean | max | clips with <5 cm travel |
|---|:---:|:---:|:---:|:---:|
| `eval_subset` | 0.81 m | 1.95 m | 19.27 m | 3.8% |
| `picoset_20260924` | 0.03 m | 0.06 m | 1.26 m | **68.8%** |
| **`picoset_20260928`** | **3.42 m** | **4.28 m** | **10.03 m** | **0.0%** |

**This invalidates the "never use `mpjpe_g` on PICO sets" rule for THIS set only.**
The 0924 finding (rho = -0.20 for `mpjpe_g`) was caused by `transl = 0` making
global MPJPE measure drift against a stationary reference. Here the reference
actually moves — further than `eval_subset`'s — so `mpjpe_g` is a real metric again.

| Checkpoint | Succ | Prog | MPJPE (G) | MPJPE (L) | MPJPE (PA) |
|---|:---:|:---:|:---:|:---:|:---:|
| LOW_LATENCY (novrll_074k) | 0.481 | 0.588 | 287.6 | 35.39 | 27.27 |
| REUBEN novr_050k | 0.519 | 0.597 | 274.8 | 33.66 | 24.79 |
| REUBEN novrll_062k | 0.481 | 0.611 | 280.2 | 36.52 | 28.07 |
| REUBEN novrllam_080k | 0.556 | 0.674 | 267.5 | 37.70 | 28.53 |
| REUBEN novrllam_100k | **0.593** | 0.665 | **249.5** | 35.20 | 26.24 |
| **MEAN** | **0.526** | **0.627** | **271.9** | **35.69** | **26.98** |

**Findings**

1. **`novrllam_100k` is best on the metrics that now matter.** Highest success
   (0.593) and lowest global MPJPE (249.5 mm). The LLAM 080k -> 100k step
   improves both, so the `novrllam` family keeps scaling past the 080k
   recommendation made on the older sets.
2. **`mpjpe_g` finally separates checkpoints monotonically** (287.6 -> 249.5 mm,
   spread 38 mm) and tracks success rate. On `picoset_20260924` the same column
   swung 276-392 mm with rho = -0.20 against `pico_evalset`, i.e. pure noise.
3. **Success/progress are lower than on any earlier PICO set** (0.526 / 0.627 vs
   0.633 / 0.789 on 0924). Expected: these clips contain real walking, crawling
   (`clip_005_006`) and a forward jump (`clip_018`), where the older sets were
   in-place upper-body motion only. This set is a genuine locomotion benchmark.
4. **Local/PA MPJPE got slightly WORSE** (35.69 / 26.98 vs 33.02 / 24.54 on 0924).
   Consistent with (3) — tracking a moving base is harder than tracking a pinned
   one — not a regression in the checkpoints.

**Caveat on cross-set rank transfer.** Only 4 checkpoints overlap with
`picoset_20260924`, and over those 4 the Spearman correlations are weak and all
statistically insignificant (`mpjpe_l` / `mpjpe_pa` rho = +0.60, p = 0.40;
success rho = +0.33; `mpjpe_g` rho = +0.20). With n = 4 this neither confirms nor
refutes agreement — it is simply too small a sample. Do not read the 0924 ranking
onto this set, and vice versa; re-run the checkpoints you care about here.

Build script: `gam/data_process/build_picoset_20260928.py`. The smpl (50 fps) and
robot (30 fps) branches are retargeted **separately** from the raw clips — the
robot branch goes through its own 30 fps `smpl_filtered` -> SMPL-X -> retarget
chain — then reconciled with the same `align_pair()` helper (27/27 pairs aligned,
5 already-aligned + 22 1-frame smpl trims). Source clips are chunked/QC'd by
`gam/gear_sonic/scripts/chunk_pico_session.py` and `qc_all_clips.py`.
"""

txt = txt.replace(anchor, section)

with open(DOC, "w") as f:
    f.write(txt)

print("patched OK")
print("picoset_20260928 mentions:", txt.count("picoset_20260928"))
