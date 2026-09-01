# Pico Teleop Evalset — 3-Model Comparison

> **Provenance:** copied from
> `GR00T-WholeBodyControl/dev_notes/eval_examples/notes_pico_evalset_3model_comparison.md`.
> Relative references below (`README.md`, `../fps_check_alignment/`,
> `dev_notes/motion_data_filter/build_pico_evalset.py`) point at paths in **that**
> repo, not this one. The source `smpl` clips are produced here by
> `data_process/pico_to_smpl_filtered.py` (see
> `PICO_SMPL_STREAMING_DATASETS_NOTE.md`).

Evaluated three checkpoints on the `pico_evalset` (20 clips, built from
teleop-captured pico VR data), across two `num_envs` settings to also
illustrate run-to-run GPU-nondeterminism sensitivity.

## Dataset

`pico_evalset` was built by `dev_notes/motion_data_filter/build_pico_evalset.py`
from:
- **smpl**: `/home/grease/gam/logs_pkl` (20 clips, pre-resampled to 50fps)
- **robot**: `/home/grease/GMR/teleop_retargeted_g1_motion_lib` (20 clips, native 30fps)

Frame-count alignment was applied via `gmr.tools.fix_evalset_frame_mismatch.align_pair()`
(same mechanism used for `amass_trainset`):

| Status | Count |
|---|---|
| `already_aligned` | 2 |
| `trimmed_smpl` | 18 |
| `trimmed_robot` | 0 |
| `padded_smpl_fallback` | 0 |

Verified with `check_fps_mismatch.py`: **0/20 mismatches** after alignment.

Output: `/home/grease/ego_dataset/pico_evalset/{robot,smpl}/` (20 files each).

## Checkpoints evaluated

| Name | Checkpoint path |
|---|---|
| RELEASED | `/home/grease/GR00T-WholeBodyControl/sonic_release/last.pt` |
| LOW_LATENCY | `/home/grease/gam/gear_sonic_deploy/policy/low_latency/last.pt` |
| PRETRAINED | `/home/grease/gam/sonic_pretrained/model_step_150000.pt` |

## Eval command template

```bash
/home/grease/miniforge3/envs/env_isaaclab/bin/python gear_sonic/eval_agent_trl.py \
  checkpoint=<checkpoint_path> \
  +headless=true \
  +num_envs=<N> \
  +manager_env.commands.motion.motion_lib_cfg.motion_file=/home/grease/ego_dataset/pico_evalset/robot \
  +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/home/grease/ego_dataset/pico_evalset/smpl \
  eval_name=<EVAL_NAME> \
  +run_once=true \
  +eval_callbacks=im_eval \
  +eval_output_dir='${eval_log_dir}' \
  ++manager_env.observations.policy.enable_corruption=False \
  ++manager_env.observations.tokenizer.enable_corruption=False \
  +manager_env/terminations=tracking/eval \
  ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=20
```

`num_envs` was varied (20 vs 64) with `max_unique_motions=20` fixed (the
dataset only has 20 unique clips), to test result stability across batch
sizes. See `../fps_check_alignment/` and the note on run-to-run
nondeterminism in `README.md` for why this matters.

## Run 1: `num_envs=20` (one env per unique clip)

| Model | Log dir | success_rate | progress_rate |
|---|---|---|---|
| RELEASED | `logs_eval/20260831_173852-EVAL_pico_20clips_RELEASED` | 0.6500 | **0.8382** |
| LOW_LATENCY | `logs_eval/20260831_174148-EVAL_pico_20clips_LOW_LATENCY` | 0.6500 | 0.8142 |
| PRETRAINED | `logs_eval/20260831_174346-EVAL_pico_20clips_PRETRAINED` | 0.6500 | 0.8169 |

All three tie on success_rate (13/20); `RELEASED` slightly ahead on progress_rate.

### Per-clip progress (num_envs=20)

| Motion | RELEASED | LOW_LATENCY | PRETRAINED |
|---|---|---|---|
| `paired_smpl_raw_clip_2` | 0.1651 | 0.0664 | 0.1328 |
| `paired_smpl_raw_clip_5` | 0.1964 | 0.1982 | 0.1922 |
| `yelong_cliptest_0` | 0.3008 | 0.2683 | **1.0000** |
| `reuben_cliptest_0` | 0.6545 | 0.6515 | **1.0000** |
| `paired_smpl_raw_clip_0` | 0.7144 | **1.0000** | 0.4438 |
| `paired_smpl_raw_clip_1` | 0.7852 | 0.5784 | 0.8953 |
| `paired_smpl_g1_deploy_clipreal_6` | 0.9484 | 0.9226 | 0.9251 |
| `paired_smpl_g1_deploy_clipreal_5` | 1.0000 | 1.0000 | 1.0000 |
| `paired_smpl_raw_clip_4` | 1.0000 | 0.5979 | 1.0000 |
| `smpl_raw_real_robot_clip_0` | **1.0000** | **1.0000** | **0.0912** |
| `paired_smpl_g1_deploy_clipreal_1` | 1.0000 | 1.0000 | 1.0000 |
| `paired_smpl_g1_deploy_clipreal_3` | 1.0000 | 1.0000 | 1.0000 |
| `reuben_cliptest_2` | 1.0000 | 1.0000 | 1.0000 |
| `reuben_cliptest_1` | 1.0000 | 1.0000 | 1.0000 |
| `yelong_cliptest_1` | 1.0000 | 1.0000 | 0.6571 |
| `paired_smpl_g1_deploy_clipreal_0` | 1.0000 | 1.0000 | 1.0000 |
| `paired_smpl_raw_clip_3` | 1.0000 | 1.0000 | 1.0000 |
| `paired_smpl_g1_deploy_clipreal_7` | 1.0000 | 1.0000 | 1.0000 |
| `paired_smpl_g1_deploy_clipreal_4` | 1.0000 | 1.0000 | 1.0000 |
| `paired_smpl_g1_deploy_clipreal_2` | 1.0000 | 1.0000 | 1.0000 |

## Run 2: `num_envs=64` (clips repeated/sampled with replacement across envs)

| Model | Log dir | success_rate | progress_rate |
|---|---|---|---|
| RELEASED | `logs_eval/20260901_114154-EVAL_pico_64envs_RELEASED` | 0.6500 | 0.8232 |
| LOW_LATENCY | `logs_eval/20260901_114331-EVAL_pico_64envs_LOW_LATENCY` | 0.6000 | 0.7847 |
| **PRETRAINED** | `logs_eval/20260901_114527-EVAL_pico_64envs_PRETRAINED` | **0.7000** | **0.8501** |

`PRETRAINED` is now the best (unlike Run 1's tie) — a direct demonstration
of GPU-physics-batch-size sensitivity noted in `README.md`.

### Per-clip progress (num_envs=64, averaged across duplicated envs per motion)

| Motion | RELEASED | LOW_LATENCY | PRETRAINED |
|---|---|---|---|
| `paired_smpl_raw_clip_2` | 0.1641 | 0.1015 | 0.1319 |
| `paired_smpl_raw_clip_5` | 0.1970 | 0.1976 | 0.1922 |
| `yelong_cliptest_0` | 0.3008 | **1.0000** | **1.0000** |
| `paired_smpl_raw_clip_0` | 0.4445 | **1.0000** | 0.1682 |
| `reuben_cliptest_0` | 0.6515 | **1.0000** | **1.0000** |
| `paired_smpl_raw_clip_1` | 0.7839 | 0.5791 | 0.9880 |
| `paired_smpl_g1_deploy_clipreal_6` | 0.9226 | 0.8673 | 0.8673 |
| `paired_smpl_g1_deploy_clipreal_5` | 1.0000 | 1.0000 | 1.0000 |
| `paired_smpl_raw_clip_4` | 1.0000 | 0.5979 | 1.0000 |
| `smpl_raw_real_robot_clip_0` | 1.0000 | 1.0000 | 1.0000 |
| `paired_smpl_g1_deploy_clipreal_1` | 1.0000 | 0.2984 | 1.0000 |
| `paired_smpl_g1_deploy_clipreal_3` | 1.0000 | 1.0000 | 1.0000 |
| `reuben_cliptest_2` | 1.0000 | 1.0000 | 1.0000 |
| `reuben_cliptest_1` | 1.0000 | 1.0000 | 1.0000 |
| `yelong_cliptest_1` | 1.0000 | 1.0000 | 0.6536 |
| `paired_smpl_g1_deploy_clipreal_0` | 1.0000 | 0.9481 | 1.0000 |
| `paired_smpl_raw_clip_3` | 1.0000 | 1.0000 | 1.0000 |
| `paired_smpl_g1_deploy_clipreal_7` | 1.0000 | 1.0000 | 1.0000 |
| `paired_smpl_g1_deploy_clipreal_4` | 1.0000 | 1.0000 | 1.0000 |
| `paired_smpl_g1_deploy_clipreal_2` | 1.0000 | 0.1035 | 1.0000 |

## Conclusions

1. **`paired_smpl_raw_clip_2` and `paired_smpl_raw_clip_5` fail consistently**
   across all 3 checkpoints and both `num_envs` settings (progress ~0.10-0.20)
   — genuinely hard/borderline clips, not a checkpoint- or batch-size-specific
   issue. Good candidates for either excluding or targeted investigation.
2. **No single checkpoint universally dominates** on this small,
   teleop-specific evalset — unlike the AMASS evalset where `RELEASED` was a
   clear winner (see `notes_amass_108clips_3model_comparison.md`). Each
   checkpoint has different clip-specific strengths (e.g. `PRETRAINED` nails
   `yelong_cliptest_0`/`reuben_cliptest_0` in both runs; `LOW_LATENCY`
   uniquely succeeds on `paired_smpl_raw_clip_0` at `num_envs=20` and on
   `yelong_cliptest_0`/`reuben_cliptest_0` at `num_envs=64`).
3. **Results are sensitive to `num_envs`** — several clips flip between
   success/failure across the two runs for the *same* checkpoint (e.g.
   `LOW_LATENCY` on `paired_smpl_g1_deploy_clipreal_1`/`clipreal_2` goes from
   1.0 at `num_envs=20` to 0.10-0.30 at `num_envs=64`). Always fix `num_envs`
   consistently when comparing checkpoints (see `README.md`).
4. **`_raw_clip` clips appear systematically harder** than
   `_deploy_clipreal_*` clips across both runs — the raw teleop-captured
   motion may be noisier/less physically feasible than the curated "deploy"
   variants.
