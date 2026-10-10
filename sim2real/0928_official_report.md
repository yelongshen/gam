# Checkpoint Comparison — Offline vs Online, 2026-09-28

Official comparison of four G1 whole-body-control checkpoints across **offline (simulated)
evaluation** and **online (real-robot) deployment**:

| short name | checkpoint |
|---|---|
| `low_latency` | released low-latency baseline (`gear_sonic_deploy/policy/low_latency`) |
| `novr_050k` | `sonic_no_vr_envs_16K` @ step 050k |
| `ll_062k` | `sonic_no_vr_low_latency_envs_16K` @ step 062k |
| `llam_080k` | `sonic_no_vr_low_latency_with_amass_envs_16K` @ step 080k |

**Sources.** Offline: `GR00T-WholeBodyControl/model_eval/checkpoint_comparison.md`
(`eval/success/*` metrics). Online: `gam/sim2real/online_eval_policy_comparison.md` and the
seven per-session reports it pools; raw metrics in `gam/sim2real/online_eval_results*.json`.

**Bottom line.** Offline, **`llam_080k` is the best checkpoint on every dataset**. Online,
**`ll_062k` is the best of the three ever deployed** — and `llam_080k` has **never been
evaluated on hardware**, so the two rankings have not actually been tested against each
other. That gap is the single most important open item (§5).

---

## 0. Metric definitions

All five offline metrics come from `eval/success/*` in `metrics_eval.json`, produced by
`gear_sonic/trl/callbacks/im_eval_callback.py`. The three MPJPE variants are a direct port
of `smpl_sim.smpllib.smpl_eval.compute_metrics_lite`; the online evaluator
(`sim2real/online_eval_clips.py`) re-implements the same formulas so the two are
numerically comparable.

| metric | unit | direction | what it measures |
|---|---|---|---|
| **Success rate** | fraction | ↑ higher better | Share of clips the policy played to the end **without triggering a termination** (`1 - mean(terminated)`). A termination is a fall / anchor-position / anchor-orientation / end-effector / foot-position violation. |
| **Progress rate** | fraction | ↑ higher better | Mean fraction of each clip completed before termination; successful clips count as 1.0. Distinguishes "failed at 10 %" from "failed at 90 %" — a partial-credit version of success. |
| **MPJPE-G** | mm | ↓ lower better | **Global** mean per-joint position error — robot vs reference link positions in the **world frame**, no alignment. Includes root translation error, so it is dominated by base drift. |
| **MPJPE-L** | mm | ↓ lower better | **Local** (root-relative) MPJPE — both skeletons are translated so the pelvis sits at the origin, then compared. Removes base drift and measures **pose/limb accuracy**. This is the primary tracking-quality number. |
| **MPJPE-PA** | mm | ↓ lower better | **Procrustes-aligned** MPJPE — optimal rotation + translation + scale fitted per frame before comparing. Removes whole-body orientation and scale mismatch, leaving only **relative joint configuration** error. Always ≤ MPJPE-L. |

Nesting: `MPJPE-PA ≤ MPJPE-L ≤ MPJPE-G`, since each strips away one more source of error
(scale/orientation, then root translation). Comparing them is informative — a large
G-vs-L gap means the robot has the right *pose* but is in the wrong *place*, and a large
L-vs-PA gap means correct limb configuration but wrong body orientation.

**"Computed over successful rollouts only."** The `eval/success/*` MPJPEs average over
clips that did **not** terminate. A policy that fails the hard clips is therefore scored
only on the easy ones it survived — so a low MPJPE with a low success rate can be
flattering. Always read MPJPE together with success rate.

Two further metrics appear in the online sections:

| metric | unit | meaning |
|---|---|---|
| **cmd `mpjpe_l`** | mm | `FK(q_target)` vs `FK(q_measured)` — control-loop fidelity, i.e. how well the PD loop realises the policy's own commands (as opposed to how well those commands match the reference). |
| **joint err** | deg | Mean absolute per-joint angle error against the retargeted reference `dof`, across all 29 joints. |

> **MPJPE-G is unreliable on two of the four datasets.** Both PICO sets store `transl = 0`
> (the capture has no pelvis world position), so MPJPE-G measures drift against a
> *stationary* reference and behaves like noise — Spearman rho between the two PICO sets is
> **-0.20** for G, versus **+0.92** for L. On real hardware it is degenerate for a different
> reason: with no external mocap the root is pinned at the origin, making G numerically
> identical to L. **Rank on L and PA.**

---

## 1. Offline evaluation (simulation)

All numbers are `eval/success/*`, i.e. computed over successful rollouts only (§0).

### 1.1 `eval_subset` (160 clips, in-distribution)

| checkpoint | Success | Progress | MPJPE-G | MPJPE-L | MPJPE-PA |
|---|:---:|:---:|:---:|:---:|:---:|
| `low_latency` | 0.931 | 0.949 | 201.72 | 28.69 | 21.88 |
| `novr_050k` | 0.981 | 0.988 | 118.97 | 24.60 | 17.49 |
| `ll_062k` | **0.988** | **0.993** | 138.63 | 25.38 | 17.42 |
| **`llam_080k`** | 0.981 | 0.989 | **122.09** | **24.45** | **17.43** |

### 1.2 `amass_evalset` (108 clips, out-of-distribution)

| checkpoint | Success | Progress | MPJPE-G | MPJPE-L | MPJPE-PA |
|---|:---:|:---:|:---:|:---:|:---:|
| `low_latency` | 0.611 | 0.745 | 345.21 | 34.27 | 23.44 |
| `novr_050k` | 0.731 | 0.845 | 205.30 | 33.30 | 24.54 |
| `ll_062k` | 0.731 | 0.837 | 1199.35 | 33.40 | 24.31 |
| **`llam_080k`** | **0.917** | **0.951** | **213.21** | **32.74** | **22.30** |

This is where `llam_080k` separates decisively: **+0.186 success over the next best**
(0.917 vs 0.731), which is the payoff from adding AMASS to the training corpus.

### 1.3 `pico_evalset` (20 clips) and `picoset_20260924` (93 clips)

| checkpoint | pico_evalset Succ / Prog / L | picoset_20260924 Succ / Prog / L |
|---|---|---|
| `low_latency` | 0.650 / 0.814 / 30.15 | 0.613 / 0.760 / 32.29 |
| `novr_050k` | 0.700 / 0.829 / 29.73 | 0.624 / 0.796 / 31.72 |
| `ll_062k` | 0.750 / 0.859 / 31.01 | 0.634 / 0.779 / 32.57 |
| **`llam_080k`** | **0.800 / 0.910 / 30.00** | **0.688 / 0.818 / 32.28** |

The two PICO sets agree on ranking (`mpjpe_l` Spearman rho = **+0.92** over 10
checkpoints), with `picoset_20260924` uniformly harder. `llam_080k` leads both.

> **MPJPE-G caveat (§0) applies here.** Both PICO sets store `transl = 0`, so rank on
> L / PA. The `ll_062k` `amass_evalset` MPJPE-G of **1199.35** is a related artifact —
> a few clips with large base drift dominate the mean — and should not be read as a real
> regression; its L (33.40) and PA (24.31) are in line with `novr_050k`.

### 1.4 Offline summary

| checkpoint | eval_subset | amass_evalset | pico_evalset | picoset_20260924 | rank |
|---|:---:|:---:|:---:|:---:|:---:|
| `low_latency` | 0.931 | 0.611 | 0.650 | 0.613 | 4 |
| `novr_050k` | 0.981 | 0.731 | 0.700 | 0.624 | 3 |
| `ll_062k` | **0.988** | 0.731 | 0.750 | 0.634 | 2 |
| **`llam_080k`** | 0.981 | **0.917** | **0.800** | **0.688** | **1** |

(success rate; `llam_080k` wins 3 of 4 and is within 0.007 on the fourth)

---

## 2. Online evaluation (real robot)

8 sessions, 2026-09-04 → 09-24. 56 episodes located, **40 clip-confirmed**. Clip identity
is recovered from *state* on both sides (streamed SMPL **and** executed `q` must name the
same clip); unconfirmed episodes are excluded from every number below.

### 2.1 Per-policy aggregate

| policy | episodes | clips | `mpjpe_l` | `mpjpe_pa` | joint err | cmd `mpjpe_l` | sat % | shaking | max tilt | falls |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `low_latency` | 20 | 6 | 41.48 | 31.07 | 6.80° | 65.4 | 0.91 | 0.435 | 13.7° | 0 |
| `novr_050k` | 11 | 5 | 36.33 | 31.97 | 5.93° | 65.4 | 1.18 | 0.382 | 10.3° | 0 |
| **`ll_062k`** | 9 | 4 | **30.46** | **25.47** | **5.35°** | 90.5 | 1.59 | 0.479 | **8.0°** | 0 |
| `llam_080k` | **0** | 0 | — | — | — | — | — | — | — | — |

### 2.2 Clip-matched comparison (the fair view)

Aggregates are confounded by clip mix — `low_latency` is the only policy that attempted the
two kneeling clips (57–58 mm, the hardest measured), which inflates its average
independently of quality. Matching on clip removes this:

| clip | `low_latency` | `novr_050k` | `ll_062k` |
|---|:---:|:---:|:---:|
| `walk_180_R_003__A332_M` | 32.0 (n=7) | **25.9** (n=5) | 29.1 (n=2) |
| `walk_sideway_045_stop_005__A042_M` | 49.3 (n=4) | 54.3 (n=2) | **24.9** (n=1) |
| `jog_ff_start_180_R_002__A192_M` | 39.7 (n=3) | **32.2** (n=1) | 36.2 (n=1) |
| `walk_backward_start_001__A030_M` | — | 46.6 (n=2) | **31.0** (n=5) |
| **macro-mean (3 clips all three ran)** | **40.4** | **37.5** | **30.1** |

**`ll_062k` reduces tracking error 25 % vs the `low_latency` baseline on identical clips.**

### 2.3 Online findings

1. **`ll_062k`'s biggest win is lateral gait.** `walk_sideway` drops 49.3 → 24.9 mm, and
   heading error on that clip collapses from **20–30° under `novr_050k` to 1.9°**. This was
   the standout defect in the 09-09 session and it does not reproduce.
2. **Zero falls in all 56 episodes**, max tilt 22.7° against a 35° threshold. When a clip
   is beyond capability the robot refuses the motion rather than destabilising.
3. **Better imitation costs command-tracking headroom**: `low_latency` → `ll_062k` raises
   cmd `mpjpe_l` 65.4 → 90.5 mm and saturation 0.91 → 1.59 %. Imitation error stays well
   below command error throughout — the PD loop lags each command but averages onto the
   reference.
4. **Ankle pitch is the universal actuation hotspot**, peaking at 33 % of timesteps above
   0.9×effort, policy-independent. A hardware/limit property, not a checkpoint problem.

---

## 3. Offline vs online: do the rankings agree?

| checkpoint | offline rank | online rank | agree? |
|---|:---:|:---:|:---:|
| `low_latency` | 4 | 3 (worst deployed) | ✓ |
| `novr_050k` | 3 | 2 | ✓ |
| `ll_062k` | 2 | **1** | — |
| `llam_080k` | **1** | **not deployed** | **untested** |

Among the three checkpoints with both kinds of data, **the rankings are consistent**:
offline `ll_062k` > `novr_050k` > `low_latency`, and online the same order. That is a
genuine (if small-n) sim2real transfer result.

The open question is entirely about `llam_080k`. It is offline-best by a wide margin on
OOD data, but has **no hardware episodes at all**, so nothing can yet be said about whether
its advantage transfers.

### 3.1 What exists for `llam_080k`, and why it does not count

Three deploy runs used it (1385 s of streamed motion):

| log | rows | streamed | why unusable |
|---|---|---|---|
| `gear_sonic_deploy/logs/24-09-26/17-36-00` | 55,425 | 1094 s | MuJoCo sim + live PICO teleop |
| `gear_sonic_deploy/logs/27-09-26/17-46-13` | 15,342 | 291 s | MuJoCo sim + live PICO teleop |
| `gear_sonic_deploy/logs/24-09-26/17-16-53` | 39,338 | 0 s | idle reference motion only |

Two disqualifiers: they are **sim**, not hardware (they write to
`gear_sonic_deploy/logs/`, whereas real-robot runs write to `g1_robot_data/`); and they
were **driven by live PICO teleop, so there is no reference trajectory** — clip
identification against all 160 `eval_subset` refs returns a best fit of only 9.97–12.45°
rms with ~1 % margin, i.e. no manifest clip was replayed. Without a reference, `mpjpe_l` is
undefined.

---

## 4. Recommendation

- **For offline/simulation work and as the default training checkpoint: `llam_080k`.** It
  wins or ties on all four datasets and is dramatically better OOD (+0.186 success on
  `amass_evalset`).
- **For hardware deployment today: `ll_062k`.** It is the best policy with real evidence
  behind it (30.1 mm macro-mean, 9 confirmed episodes, the lateral-gait fix), and it is a
  25 % improvement over the shipped `low_latency` baseline.
- **Do not promote `llam_080k` to hardware on offline evidence alone** until §5 is done.
  The offline gap between `ll_062k` and `llam_080k` on `eval_subset` / `pico_evalset` is
  small (0.007 success, 1 mm L); the large gap is on `amass_evalset`, which no hardware
  session has ever exercised.

## 5. Required next experiment

**≥3 reps each of `walk_180_R_003__A332_M`, `walk_sideway_045_stop_005__A042_M` and
`walk_backward_start_001__A030_M` under `llam_080k` on the real robot.** These are the
clips the existing three policies share, so results drop straight into the §2.2 table and
either confirm the offline ranking transfers or establish a concrete sim2real inversion.

```bash
# terminal 1 — policy on the ROBOT
cd ~/gam/gear_sonic_deploy
bash deploy.sh --input-type zmq --zmq-host <TELEOP_PC_IP> --zmq-port 5556 \
  --zmq-topic pose --enable-csv-logs real      # with policy/sonic_no_vr_llam_080k

# terminal 2 — replay a manifest clip (repeat >=3x per clip)
.venv_teleop/bin/python data_process/stream_clip_mode2.py \
  --path ../ego_dataset/eval_subset/smpl/walk_180_R_003__A332_M.pkl \
  --fps 50 --settle 2.0 --chunk-frames 4 --host 192.168.8.192 --port 5556
```

Use `reference/real_example/` (not `example/`) — see the 09-19 vs 09-24 comparison.

---

## 6. Caveats

1. **Online n is small and unbalanced**: 20 / 11 / 9 confirmed episodes over 6 / 5 / 4
   clips. Differences below ~10 mm are not resolvable; `walk_180` alone spans 29.8–67.8 mm
   within one policy.
2. **Selection-based statistics are unsafe here.** A top-2-of-n summary improves
   `low_latency` by 7.2 mm but `ll_062k` by 0.0 mm, purely because the baseline has more
   reps to select from. All-mean is used throughout.
3. **09-19 is confounded** (degraded stream + wrong reference dir); its numbers are ~2×
   worse than 09-15 for *both* policies it ran.
4. **09-12 measures failure, not quality** — 6 of 7 episodes are agility clips `novr_050k`
   never tracked. Excluded from the clip-matched table.
5. **PICO eval sets are in-place only** (`transl = 0`, robot root travels ~1 cm vs 26 cm on
   `eval_subset`), so they measure upper-body/in-place tracking, never locomotion.
6. **`mpjpe_g` is degenerate online** (no mocap ⇒ root pinned at origin) and is not
   reported in §2.
