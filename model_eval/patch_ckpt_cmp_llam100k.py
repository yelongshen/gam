"""One-off: add REUBEN novrllam_100k eval results to checkpoint_comparison.md.

The doc lives outside the agent workspace, so it is patched by script.

Sources (all `eval/success/*` from metrics_eval.json):
  eval_subset       logs_eval/20260928_154200-EVAL_subset_amass_ll_100k
  amass_evalset     logs_eval/20260928_155845-EVAL_amass_amass_ll_100k
  pico_evalset      logs_eval/20260928_160432-EVAL_pico_amass_ll_100k
  picoset_20260924  logs_eval/20260928_160623-EVAL_picoset_amass_ll_100k
  picoset_20260928  logs_eval/20260928_184038-EVAL_picoset20260928_novr_ll_ams_100k  (already in doc)
  stair_evalset     logs_eval/20260928_154432-EVAL_stair_amass_ll_100k  (success 0.000 -- see note)
"""

DOC = "/home/grease/GR00T-WholeBodyControl/model_eval/checkpoint_comparison.md"

with open(DOC) as f:
    txt = f.read()


def sub_once(old, new):
    global txt
    assert txt.count(old) == 1, f"anchor not unique/found: {old[:70]!r}"
    txt = txt.replace(old, new)


# ------------------------------------------------------- source log directories
sub_once(
    "| REUBEN novrllam_080k     | pico_evalset  | `logs_eval/20260921_121948-EVAL_pico_LLAM080k` |\n",
    "| REUBEN novrllam_080k     | pico_evalset  | `logs_eval/20260921_121948-EVAL_pico_LLAM080k` |\n"
    "| REUBEN novrllam_100k     | eval_subset   | `logs_eval/20260928_154200-EVAL_subset_amass_ll_100k` |\n"
    "| REUBEN novrllam_100k     | amass_evalset | `logs_eval/20260928_155845-EVAL_amass_amass_ll_100k` |\n"
    "| REUBEN novrllam_100k     | pico_evalset  | `logs_eval/20260928_160432-EVAL_pico_amass_ll_100k` |\n",
)
sub_once(
    "| REUBEN novrllam_080k | picoset_20260924 | `logs_eval/20260927_183720-EVAL_picoset20260924_LLAM080k` |\n",
    "| REUBEN novrllam_080k | picoset_20260924 | `logs_eval/20260927_183720-EVAL_picoset20260924_LLAM080k` |\n"
    "| REUBEN novrllam_100k | picoset_20260924 | `logs_eval/20260928_160623-EVAL_picoset_amass_ll_100k` |\n",
)

# ------------------------------------------------------------------ eval_subset
# 100k takes global-MPJPE outright (108.30 < 116.94 novrll_074k) and ties the
# 0.988/0.993 success/progress ceiling. L/PA stay with novrll_074k.
sub_once(
    "| **REUBEN novrllam_080k**| 0.981       | **0.989**      | **122.09**| **24.45** | **17.43**  |\n",
    "| REUBEN novrllam_080k    | 0.981       | 0.989          | 122.09    | 24.45     | 17.43      |\n"
    "| **REUBEN novrllam_100k**| **0.988**   | **0.993**      | **108.30**| 24.30     | 17.09      |\n",
)

# ---------------------------------------------------------------- amass_evalset
# 100k: new best progress (0.955) and global MPJPE (180.30 < 194.60 llam_036k);
# ties 080k/070k on success. L/PA regress slightly, so drop their bold from 080k.
sub_once(
    "| **REUBEN novrllam_080k**| **0.917**   | **0.951**      | **213.21**| **32.74** | **22.30**  |\n",
    "| REUBEN novrllam_080k    | **0.917**   | 0.951          | 213.21    | **32.74** | **22.30**  |\n"
    "| **REUBEN novrllam_100k**| **0.917**   | **0.955**      | **180.30**| 32.96     | 23.16      |\n",
)

# ----------------------------------------------------------------- pico_evalset
# 100k: new best success (0.850) AND progress (0.930), beating novrll_052k/080k.
sub_once(
    "| **REUBEN novrllam_080k**| **0.800**   | **0.910**      | 386.34    | **30.00** | **23.64**  |\n",
    "| REUBEN novrllam_080k    | 0.800       | 0.910          | 386.34    | 30.00     | 23.64      |\n"
    "| **REUBEN novrllam_100k**| **0.850**   | **0.930**      | 336.43    | **28.65** | **22.51**  |\n",
)

# ------------------------------------------------------------- picoset_20260924
# Paired table: left half = picoset_20260924, right half = pico_evalset.
sub_once(
    "| REUBEN novrllam_080k | 0.688 | 0.818 | 276.3 | 32.28 | 23.78 | | 0.800 | 0.910 | 386.3 | 30.00 | 23.64 |\n"
    "| **MEAN** | **0.633** | **0.789** | **338.6** | **33.02** | **24.54** | | 0.745 | 0.862 | 357.4 | 31.00 | 24.37 |\n",
    "| REUBEN novrllam_080k | 0.688 | 0.818 | 276.3 | 32.28 | 23.78 | | 0.800 | 0.910 | 386.3 | 30.00 | 23.64 |\n"
    "| REUBEN novrllam_100k | **0.763** | **0.870** | 344.2 | 32.22 | 23.81 | | 0.850 | 0.930 | 336.4 | 28.65 | 22.51 |\n"
    "| **MEAN** (10 ckpts, excl. 100k) | **0.633** | **0.789** | **338.6** | **33.02** | **24.54** | | 0.745 | 0.862 | 357.4 | 31.00 | 24.37 |\n",
)

# 0924 finding #2 referenced 080k as best on both sets -- now superseded.
sub_once(
    """2. **`novrllam_080k` is best on both sets**, and the LLAM 030k -> 050k -> 080k
   ordering reproduces. The recommendation in the Summary section holds.""",
    """2. **`novrllam_080k` is best on both sets** among the original 10, and the LLAM
   030k -> 050k -> 080k ordering reproduces. **`novrllam_100k`, added later, beats
   it on success/progress on both** (0.763/0.870 here, 0.850/0.930 on
   `pico_evalset`) while `mpjpe_g` moves the other way (276.3 -> 344.2) --
   consistent with finding 4 below, i.e. `mpjpe_g` is not usable on these
   `transl = 0` sets.""",
)

# ------------------------------------------------------- new cross-dataset note
anchor = """**Caveat on cross-set rank transfer.** Only 4 checkpoints overlap with"""
addition = """#### `novrllam_100k` across every dataset

The 100k checkpoint was evaluated on all sets; collected here since it is the
newest point of the `novrllam` sweep and changes the Summary recommendation.

| Dataset | Succ | Prog | MPJPE (G) | MPJPE (L) | MPJPE (PA) | vs 080k |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `eval_subset` (160) | 0.988 | 0.993 | **108.30** | 24.30 | 17.09 | G -13.8, best-ever G |
| `amass_evalset` (108) | 0.917 | **0.955** | **180.30** | 32.96 | 23.16 | G -32.9, L/PA slightly worse |
| `pico_evalset` (20) | **0.850** | **0.930** | 336.43 | 28.65 | 22.51 | better on all five |
| `picoset_20260924` (93) | 0.763 | 0.870 | 344.2 | 32.22 | 23.81 | Succ +7.5pt, G worse |
| `picoset_20260928` (27) | 0.593 | 0.665 | 249.5 | 35.20 | 26.24 | Succ +3.7pt, G -18.0 |
| `stair_evalset` | **0.000** | 0.178 | n/a | n/a | n/a | **total failure** |

**New bests set by 100k:** `eval_subset` global MPJPE (108.30, previous best
116.94 at `novrll_074k`), `amass_evalset` progress (0.955) and global MPJPE
(180.30, previous 194.60 at `novrllam_036k`), and `pico_evalset` success (0.850)
and progress (0.930) — the first checkpoint in the sweep to clear 0.80 success
there. It ties the 0.988/0.993 `eval_subset` ceiling held by `novrll_062k`.

**Where it does NOT win:** `eval_subset` local/PA MPJPE still belong to
`novrll_074k` (23.44 / 16.97 vs 24.30 / 17.09), and `amass_evalset` L/PA
regressed against 080k (32.96 / 23.16 vs 32.74 / 22.30). So 100k trades a little
in-distribution pose precision for markedly better global tracking and success.

> **`stair_evalset` is a hard zero.** `success_rate = 0.000`, `progress_rate = 0.178`,
> and the MPJPE columns are `NaN` because they are computed over successful
> rollouts only and there are none. This dataset is not part of the main
> comparison above and no other checkpoint in this doc has a stair number to
> compare against, so it is reported, not ranked — but it should be checked
> before 100k is deployed anywhere stairs matter.
> Log: `logs_eval/20260928_154432-EVAL_stair_amass_ll_100k`.

"""
assert txt.count(anchor) == 1
txt = txt.replace(anchor, addition + anchor)

with open(DOC, "w") as f:
    f.write(txt)

print("patched OK")
print("novrllam_100k mentions:", txt.count("novrllam_100k"))
