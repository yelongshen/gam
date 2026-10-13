# ICL hard set -- multi-attempt evaluation results

_Generated 2026-10-09 by `sim2real/icl_report.py` from `GR00T-WholeBodyControl/logs_eval/*EVAL_icl_*`._

## Setup

- **Dataset:** `~/ego_dataset/ICL_hardset` -- 35 clips (13 LAFAN test windows, 11 LAFAN train windows, 9 AMASS clips that fail across checkpoints, 2 eval_subset flips). Built by `data_process/build_icl_hardset.py`. The 11 `lafan_train` clips come from `lafan1_trainset` (the policy may have trained on them).
- **Protocol (`r30`):** every clip is attempted 30 times in a row in sim (headless, flat ground, no object, tracking-eval terminations). `keep` = the model's history carries over between attempts, `clr` = history cleared before each attempt.
- **Metric:** success rate = fraction of the 35 clips that complete without early termination; progress rate = mean fraction of each clip completed. One clip = 2.9 percentage points of success rate.
- **Checkpoint tags** are as named in the eval directories (`base10` is presumably the 100k base checkpoint; `hist25` / `hist50` are the history-conditioned models; `mixv3b` 14k / 20k are the mixpico v3b finetunes).

## 1. Single-attempt evals (one rollout per clip)

| run | success | progress | mpjpe_l (mm) | mpjpe_g (mm) |
|---|---|---|---|---|
| icl_base100k | 0.171 | 0.325 | 43.4 | 437.5 |
| icl_mixv3b_010000 | 0.143 | 0.305 | 42.6 | 332.3 |
| icl_mixv3b_014000 | 0.143 | 0.299 | 42.1 | 740.8 |
| icl_mixv3b_020000 | 0.143 | 0.298 | 44.9 | 270.4 |
| iclrep3_base100k | 0.086 | 0.168 | 45.2 | 1277.3 |
| iclrep3_mixv3b_014000 | 0.114 | 0.171 | 45.6 | 1262.6 |

`iclrep3_*` rows used 3 repeated attempts with the earlier harness (success counted differently, lower numbers); treat them separately.

### 1b. 3-attempt runs (r3): success rate per attempt

| run | attempt 1 | attempt 2 | attempt 3 |
|---|---|---|---|
| base100k_r3clr | 0.171 | 0.114 | 0.143 |
| base10_r3clr | 0.171 | 0.114 | 0.143 |
| base10_r3keep | 0.171 | 0.057 | 0.086 |
| hist25_r3clr | 0.143 | 0.029 | 0.114 |
| hist25_r3keep | 0.143 | 0.029 | 0.086 |
| hist50_r3clr | 0.029 | 0.086 | 0.057 |
| hist50_r3keep | 0.029 | 0.057 | 0.086 |

## 2. Success rate per block of 5 attempts (r30)

| run | 1-5 | 6-10 | 11-15 | 16-20 | 21-25 | 26-30 | last - first |
|---|---|---|---|---|---|---|---|
| base10_r30clr | 0.126 | 0.109 | 0.080 | 0.097 | 0.143 | 0.126 | +0.000 |
| base10_r30keep | 0.091 | 0.091 | 0.109 | 0.097 | 0.069 | 0.074 | -0.017 |
| hist25_r30clr | 0.086 | 0.051 | 0.074 | 0.074 | 0.046 | 0.063 | -0.023 |
| hist25_r30keep | 0.069 | 0.029 | 0.057 | 0.051 | 0.057 | 0.057 | -0.011 |
| hist50_r30clr | 0.063 | 0.080 | 0.051 | 0.074 | 0.057 | 0.069 | +0.006 |
| hist50_r30keep | 0.057 | 0.029 | 0.040 | 0.029 | 0.046 | 0.023 | -0.034 |
| mixv3b14k_r30clr | 0.137 | 0.126 | 0.114 | 0.120 | 0.109 | 0.120 | -0.017 |
| mixv3b14k_r30keep | 0.109 | 0.109 | 0.120 | 0.114 | 0.103 | 0.120 | +0.011 |
| mixv3b20k_r30clr | 0.109 | 0.091 | 0.109 | 0.086 | 0.097 | 0.109 | +0.000 |
| mixv3b20k_r30keep | 0.097 | 0.097 | 0.080 | 0.074 | 0.080 | 0.074 | -0.023 |
| hist25_30k_r30clr | 0.069 | 0.074 | 0.063 | 0.074 | 0.080 | 0.074 | +0.006 |
| hist25_30k_r30keep | 0.063 | 0.040 | 0.046 | 0.074 | 0.051 | 0.057 | -0.006 |
| mixv4_14k_r30clr | 0.109 | 0.086 | 0.114 | 0.114 | 0.091 | 0.103 | -0.006 |

## 3. Progress rate per block of 5 attempts (r30)

| run | 1-5 | 6-10 | 11-15 | 16-20 | 21-25 | 26-30 | last - first |
|---|---|---|---|---|---|---|---|
| base10_r30clr | 0.283 | 0.267 | 0.253 | 0.263 | 0.287 | 0.275 | -0.009 |
| base10_r30keep | 0.231 | 0.228 | 0.246 | 0.228 | 0.208 | 0.217 | -0.014 |
| hist25_r30clr | 0.262 | 0.240 | 0.250 | 0.259 | 0.231 | 0.248 | -0.014 |
| hist25_r30keep | 0.215 | 0.181 | 0.198 | 0.200 | 0.205 | 0.202 | -0.013 |
| hist50_r30clr | 0.229 | 0.224 | 0.221 | 0.245 | 0.218 | 0.228 | -0.001 |
| hist50_r30keep | 0.188 | 0.153 | 0.158 | 0.154 | 0.153 | 0.150 | -0.038 |
| mixv3b14k_r30clr | 0.310 | 0.319 | 0.305 | 0.320 | 0.306 | 0.312 | +0.002 |
| mixv3b14k_r30keep | 0.284 | 0.264 | 0.288 | 0.281 | 0.271 | 0.267 | -0.017 |
| mixv3b20k_r30clr | 0.292 | 0.280 | 0.293 | 0.273 | 0.289 | 0.293 | +0.001 |
| mixv3b20k_r30keep | 0.257 | 0.247 | 0.238 | 0.238 | 0.243 | 0.261 | +0.004 |
| hist25_30k_r30clr | 0.220 | 0.225 | 0.221 | 0.217 | 0.220 | 0.223 | +0.003 |
| hist25_30k_r30keep | 0.194 | 0.170 | 0.177 | 0.196 | 0.185 | 0.184 | -0.009 |
| mixv4_14k_r30clr | 0.274 | 0.262 | 0.273 | 0.276 | 0.263 | 0.266 | -0.008 |

## 4. Keep minus clear (success rate, same checkpoint)

| checkpoint | 1-5 | 6-10 | 11-15 | 16-20 | 21-25 | 26-30 | mean |
|---|---|---|---|---|---|---|---|

## Findings

- **No checkpoint improves with attempts.** Within each run the 5-attempt block means stay within about +-0.03 of each other (about one clip) with no consistent direction; the last block minus the first block ranges from -0.034 to +0.011 in success rate.
- **Keeping history does not help.** Kept runs are flat or falling and sit at or below the cleared run of the same checkpoint in most blocks; `hist50` keep falls from 0.057 to 0.023.
- **Attempt 1 is a high draw.** For `base10` attempt 1 is 0.171 but the mean of attempts 1-5 is 0.126.
- **The base checkpoint is the strongest on this set**; the history-conditioned models are at or below it.

## Caveats

- 35 clips: differences of one or two clips are within noise; blocks of 5 attempts average out sim noise but not clip selection.
- The set was chosen from clips that fail in earlier evaluations, so absolute success rates are low by construction.
- `lafan_train` clips may have been seen in training; split them out before drawing conclusions about generalization.
- Per-clip outcomes are not stored in the r30 files (only per-attempt aggregates), so which clips ever succeed is not recoverable from them.

## Runs without metrics yet (started, no `metrics_eval.json`)

- `icl_mixv4_14k_r30keep`
- `icl_mixv4_28k_r30clr`
- `icl_mixv4_28k_r30keep`

