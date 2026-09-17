# Evaluation Clip Manifest

Frozen clip list for the real-robot online deployment evaluation
(`sim2real/online_deployment_eval_plan.md` §0, §6). **Do not resample this list** — every
policy comparison must reuse exactly these clips, or results become incomparable.

Version: **v10**, frozen 2026-09-11. Source dataset: `~/ego_dataset/eval_subset/`.

---

## 1. Clips

| # | motion name | category | duration | reps on robot |
|---|---|---|---|---|
| 1 | `walk_180_R_003__A332_M` | Test-Repetition (ID) — basic locomotion | 11.20 s | 3 |
| 2 | `walk   matched none of the fourteen manifest clips. With ~160 candidates in `eval_subset`, a bulksideway_045_stop_005__A042_M` | Test-Repetition (ID) — basic locomotion | 5.54 s | 2 |
| 3 | `walk_backward_start_001__A030_M` | Test-Repetition (ID) — basic locomotion | 3.98 s | **0 — not yet replayed** |
| 4 | `walk_sideway_135_loop_003__A022` | Test-Repetition (ID) — basic locomotion | 7.40 s | **0 — not yet replayed** |
| 5 | `walk_ff_stop_225_R_002__A266_M` | Test-Repetition (ID) — basic locomotion | 4.80 s | **0 — not yet replayed** |
| 6 | `warm_up_chest_003__A359_M` | **Wrist/shoulder-yaw stress (plan §0.1)** — in-place upper body | 9.58 s | **0 — not yet replayed** |
| 7 | `jog_ff_start_180_R_002__A265` | Test-Repetition (ID) — **dynamic locomotion (jog)** | 4.70 s | **0 — not yet replayed** |
| 8 | `jog_ff_start_180_R_002__A192_M` | Test-Repetition (ID) — **same motion as clip 7, different performer** | 5.04 s | **0 — not yet replayed** |
| 9 | `reach_jump_R_001__A072_M` | **Test-Content (OOD)** — agility / jump with flight phase | 7.58 s | **0 — not yet replayed** |
| 10 | `kneeling_start_101__A063_M` | **Test-Content (OOD)** — stand → kneel descent | 5.20 s | **0 — not yet replayed** |
| 11 | `kneeling_stop_002__A051_M` | **Test-Content (OOD)** — kneel → stand recovery | 2.74 s | **0 — not yet replayed** |
| 12 | `high_jump_R_103__A389_M` | **Test-Content (OOD)** — maximal-effort vertical jump | 2.60 s | **0 — not yet replayed** |
| 13 | `dance_vouge_shake_it_babe_360_R_002__A318_M` | **Test-Content (OOD)** — vogue dance, 360° turn while travelling | 6.38 s | **0 — not yet replayed** |
| 14 | `dance_hiphop_mike_tyson_R_fast_001__A319_M` | **Test-Content (OOD)** — hip-hop dance, in-place, fast tempo | 4.84 s | **0 — not yet replayed** |

Clips 1–5, 7 and 8 are **ID / Test-Repetition** locomotion; clips 7–8 are the first **jogs**
(~2× the root speed of any walk clip) and form a **matched pair** — same motion, same take,
different subject. Clip 6 is the first **non-locomotion** entry and the first member of the
plan's **§0.1 stress subset**. Clips 9–14 are **Test-Content (OOD)**: clips 9 and 12 have a
flight phase, clips 10–11 are the only ones that reach ground contact outside
the feet — they form a **descent/recovery pair** (§3) — and clips 13–14 are **dances**, the
manifest's two fastest clips by mean joint speed and a **travelling/in-place dance pair**
(§3). Clip 12 is the shortest and by far the most violent clip in the manifest (§3).

> **Clips 3–14 have no robot data yet.** Clips 3–11 were streamed on 2026-09-08 and clips
> 12–14 on 2026-09-11, all over **loopback only** (no `--host`), so nothing reached the
> robot at `192.168.8.192`. A sliding-window search confirms none appears in any of the
> 2026-09-04 sessions. They are listed here as committed members of the eval set, but
> contribute no measurements until replayed.

---

## 2. Files and hashes

Each clip exists in two paired forms. The `robot/` file is the **primary ground truth**
for evaluation; the `smpl/` file is what gets streamed to the robot.

### `walk_180_R_003__A332_M`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/walk_180_R_003__A332_M.pkl` | 377174 B | `d3b46007cbbb72b9ec83418918ec1c33` |
| Robot (retargeted) | `eval_subset/robot/walk_180_R_003__A332_M.pkl` | 100788 B | `660f39914842bc7d0f27b004c8cf36e3` |

### `walk_sideway_045_stop_005__A042_M`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/walk_sideway_045_stop_005__A042_M.pkl` | 191732 B | `583b59a61a105e2d03cbc1b250c48039` |
| Robot (retargeted) | `eval_subset/robot/walk_sideway_045_stop_005__A042_M.pkl` | 47914 B | `8591912eceae53b6ad6280e7d132164c` |

### `walk_backward_start_001__A030_M`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/walk_backward_start_001__A030_M.pkl` | 139743 B | `77eeb451a3def90ba610795b50b0501b` |
| Robot (retargeted) | `eval_subset/robot/walk_backward_start_001__A030_M.pkl` | 33185 B | `f8481cce7c74ae6a99b0dfee1a29e338` |

### `walk_sideway_135_loop_003__A022`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/walk_sideway_135_loop_003__A022.pkl` | 259143 B | `254afc2ea8cf0ad07af2d49e9ad2ace6` |
| Robot (retargeted) | `eval_subset/robot/walk_sideway_135_loop_003__A022.pkl` | 66672 B | `5c14154734db4efbc468f341613f0bfd` |

### `walk_ff_stop_225_R_002__A266_M`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/walk_ff_stop_225_R_002__A266_M.pkl` | 167102 B | `d2d67e8b05d54e34494585df8932b240` |
| Robot (retargeted) | `eval_subset/robot/walk_ff_stop_225_R_002__A266_M.pkl` | 41476 B | `0f6d323c8e111b192a87843b367d7d3a` |

### `warm_up_chest_003__A359_M`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/warm_up_chest_003__A359_M.pkl` | 331351 B | `55021f2766ed4ea185e661076f6bea95` |
| Robot (retargeted) | `eval_subset/robot/warm_up_chest_003__A359_M.pkl` | 86039 B | `54f4cc5edd53409130fcfc9f8f42e3c2` |

### `jog_ff_start_180_R_002__A265`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/jog_ff_start_180_R_002__A265.pkl` | 160104 B | `cb7f8d5176552fde1c039e4ed857f84c` |
| Robot (retargeted) | `eval_subset/robot/jog_ff_start_180_R_002__A265.pkl` | 40601 B | `198aaf1e03c00ea83832eead5c64f097` |

### `jog_ff_start_180_R_002__A192_M`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/jog_ff_start_180_R_002__A192_M.pkl` | 177486 B | `52f90972eb1a7fdd4be42326dcde456d` |
| Robot (retargeted) | `eval_subset/robot/jog_ff_start_180_R_002__A192_M.pkl` | 44272 B | `76c9078ad66559614eccacf71ed36d42` |

### `reach_jump_R_001__A072_M`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/reach_jump_R_001__A072_M.pkl` | 244913 B | `b2f3462d284a53addd99648cb0c7f85c` |
| Robot (retargeted) | `eval_subset/robot/reach_jump_R_001__A072_M.pkl` | 65401 B | `9fd30288c6815b29672c5c5324b9d50e` |

### `kneeling_start_101__A063_M`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/kneeling_start_101__A063_M.pkl` | 170395 B | `da8700ec500d8684fb311140f90f36e9` |
| Robot (retargeted) | `eval_subset/robot/kneeling_start_101__A063_M.pkl` | 43848 B | `bd074bb052dcf9f7638ca22ad6057573` |

### `kneeling_stop_002__A051_M`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/kneeling_stop_002__A051_M.pkl` | 95665 B | `c8cdf2ff2fd471802205b86e491c6c02` |
| Robot (retargeted) | `eval_subset/robot/kneeling_stop_002__A051_M.pkl` | 21810 B | `7761441cd3e366e2ed4664e4cfe9eb21` |

### `high_jump_R_103__A389_M`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/high_jump_R_103__A389_M.pkl` | 91696 B | `eb1c2eb8cadfbe398fb9246cf05c35bd` |
| Robot (retargeted) | `eval_subset/robot/high_jump_R_103__A389_M.pkl` | 21033 B | `c2369d4da50bfb2077641059cfd15913` |

### `dance_vouge_shake_it_babe_360_R_002__A318_M`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/dance_vouge_shake_it_babe_360_R_002__A318_M.pkl` | 222140 B | `ac538167a74416acbc407a7bffeae252` |
| Robot (retargeted) | `eval_subset/robot/dance_vouge_shake_it_babe_360_R_002__A318_M.pkl` | 55846 B | `22849ba0795dbc3c3a7c7d55ede194bd` |

### `dance_hiphop_mike_tyson_R_fast_001__A319_M`

| form | path | size | md5 |
|---|---|---|---|
| SMPL (streamed) | `eval_subset/smpl/dance_hiphop_mike_tyson_R_fast_001__A319_M.pkl` | 170204 B | `692666ae462953ac6b45c3899f23fdce` |
| Robot (retargeted) | `eval_subset/robot/dance_hiphop_mike_tyson_R_fast_001__A319_M.pkl` | 41976 B | `270879e402caac79594f41a1b3228c0e` |

Verify with:

```bash
cd ~/ego_dataset/eval_subset && md5sum {smpl,robot}/{walk_180_R_003__A332_M,\
walk_sideway_045_stop_005__A042_M,walk_backward_start_001__A030_M,\
walk_sideway_135_loop_003__A022,walk_ff_stop_225_R_002__A266_M,warm_up_chest_003__A359_M,\
jog_ff_start_180_R_002__A265,jog_ff_start_180_R_002__A192_M,reach_jump_R_001__A072_M,\
kneeling_start_101__A063_M,kneeling_stop_002__A051_M,high_jump_R_103__A389_M,\
dance_vouge_shake_it_babe_360_R_002__A318_M,dance_hiphop_mike_tyson_R_fast_001__A319_M}.pkl
```

---

## 3. Clip properties

| # | clip | SMPL fr @50fps | robot `dof` @30fps | `dof` range | root disp (x,y,z) m | path | **height range** | root speed mean/peak | joint speed mean/peak | ROM arms/legs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `walk_180_R_003__A332_M` | 560 (11.20 s) | (337, 29) (11.23 s) | −35.9…+83.3° | (+0.15, −7.00, −0.00) | 7.21 m | 0.065 m | 0.64 / 1.35 m/s | 31.2 / 566 °/s | — |
| 2 | `walk_sideway_045_stop_005__A042_M` | 277 (5.54 s) | (167, 29) (5.57 s) | −52.0…+60.1° | (−1.42, +1.49, +0.01) | 2.26 m | 0.052 m | 0.41 / 0.88 | 22.1 / 510 | — |
| 3 | `walk_backward_start_001__A030_M` | 199 (3.98 s) | (120, 29) (4.00 s) | −44.6…+85.2° | (+0.09, −2.57, −0.01) | 2.62 m | 0.041 m | 0.66 / 1.24 | 36.8 / 512 | — |
| 4 | `walk_sideway_135_loop_003__A022` | 370 (7.40 s) | (223, 29) (7.43 s) | −43.4…+79.9° | (−3.06, −3.04, +0.01) | 4.40 m | 0.066 m | 0.59 / 0.98 | 40.2 / 525 | — |
| 5 | `walk_ff_stop_225_R_002__A266_M` | 240 (4.80 s) | (145, 29) (4.83 s) | −46.6…+85.3° | (+2.11, −2.10, +0.01) | 3.04 m | 0.048 m | 0.63 / 0.91 | 34.7 / 445 | — |
| 6 | `warm_up_chest_003__A359_M` | 479 (9.58 s) | (288, 29) (9.60 s) | **−171.2…+112.3°** | (−0.00, −0.01, −0.00) | **0.37 m** | 0.005 m | 0.04 / 0.15 | 39.8 / **1037** | **110.8 / 5.6°** |
| 7 | `jog_ff_start_180_R_002__A265` | 235 (4.70 s) | (142, 29) (4.73 s) | −50.1…+99.5° | (−0.00, +6.23, +0.01) | 6.34 m | 0.090 m | **1.35 / 2.32** | **69.3** / 777 | 43.3 / 50.3° |
| 8 | `jog_ff_start_180_R_002__A192_M` | 252 (5.04 s) | (152, 29) (5.07 s) | −50.9…+85.3° | (+0.03, +6.05, −0.01) | 6.26 m | 0.079 m | **1.24 / 2.14** | **64.4** / 702 | 41.5 / 41.8° |
| 9 | `reach_jump_R_001__A072_M` | 379 (7.58 s) | (228, 29) (7.60 s) | −151.4…+77.3° | (+0.01, +0.02, +0.00) | 1.06 m | **0.316 m** | 0.14 / 1.64 | 18.5 / 696 | 53.7 / 38.8° |
| 10 | `kneeling_start_101__A063_M` | 260 (5.20 s) | (157, 29) (5.23 s) | −50.0…**+155.1°** | (−0.01, −0.34, **−0.37**) | 0.76 m | **0.424 m** | 0.15 / 0.60 | 13.2 / 171 | 22.0 / 50.3° |
| 11 | `kneeling_stop_002__A051_M` | 137 (2.74 s) | (83, 29) (2.77 s) | −64.9…**+152.2°** | (−0.03, −0.03, **+0.37**) | 0.59 m | **0.377 m** | 0.21 / 0.85 | 22.8 / 339 | 16.3 / 49.1° |
| 12 | `high_jump_R_103__A389_M` | 130 (2.60 s) | (79, 29) (2.63 s) | −150.7…**+159.1°** | (−0.03, −0.12, −0.00) | 1.57 m | **0.579 m** | 0.60 / **2.27** | **100.2 / 1901** | **92.6 / 64.7°** |
| 13 | `dance_vouge_shake_it_babe_360_R_002__A318_M` | 319 (6.38 s) | (192, 29) (6.40 s) | −92.3…+77.7° | (+0.11, +3.05, −0.00) | 3.72 m | 0.152 m | 0.58 / 1.25 | **126.1** / 1190 | **84.4 / 51.3°** |
| 14 | `dance_hiphop_mike_tyson_R_fast_001__A319_M` | 242 (4.84 s) | (146, 29) (4.87 s) | −101.2…+126.4° | (−0.13, −0.16, −0.00) | 2.03 m | 0.112 m | 0.42 / 1.01 | **108.7** / 1171 | **78.2 / 62.1°** |

Pelvis mean (de-rotated), which encodes facing direction for the locomotion clips:
clip 1 (0.350, −0.018, −0.005) · clip 2 (0.333, 0.085, 0.022) · clip 3 (**−0.350**, −0.007, 0.026) ·
clip 4 (0.023, **0.342**, 0.007) · clip 5 (0.337, −0.090, 0.016) · clip 6 (0.350, 0.019, 0.012) ·
clip 7 (**−0.349**, −0.014, −0.010) · clip 8 (**−0.348**, −0.014, −0.011) · clip 9 (0.349, 0.008, 0.009) ·
clip 10 (0.342, 0.055, −0.026) · clip 11 (0.350, −0.013, 0.011) · clip 12 (0.338, 0.005, 0.068) ·
clip 13 (0.308, −0.038, −0.004) · clip 14 (0.347, 0.011, −0.023)

Regenerate any row with `sim2real/clip_stats.py <clip_name>` (root path and root speed are
**3D**, not ground-plane).

Note the **fps mismatch**: SMPL is 50 fps, the retargeted robot reference is 30 fps. The
evaluator linearly resamples the reference to the log rate; see the report's §7 caveat
about this mildly low-passing `vel_dist` / `accel_dist`.

For the locomotion clips (1–5), the **pelvis mean encodes the facing direction** relative
to travel: forward (+x, clips 1, 2, 5), backward (−x, clip 3), lateral (+y, clip 4). This
separation is why the locator never confuses them, but it is *not* uniform — clip 5 is
forward-facing like clips 1–2 and its best non-match distance is only **120 mm**, versus
437 mm (clip 4) and 675 mm (clip 3). Still ~20× the 4.6–6.7 mm true-match band, so
identification remains unambiguous, but the margin is much thinner for same-direction
clips and should not be assumed large in future sweeps.

Clip 4's diagonal displacement (−3.06, −3.04) m matches a sustained 135° sideways gait;
clip 5's (+2.11, −2.10) m is the mirrored 225° equivalent.

### Clip 6 is structurally different from the rest

`warm_up_chest_003` is **in-place upper-body motion**, not locomotion, and it is the reason
it was added — it is the first clip capable of probing the plan's §0.1 stress axes:

| | clip 6 | clips 1–5 |
|---|---|---|
| root path | **0.37 m** (essentially stationary) | 2.26–7.21 m |
| `dof` range | **−171.2° … +112.3°** | ≈ −52° … +85° |
| mean arm ROM | **110.8°** | arm-quiet |
| mean leg ROM | **5.6°** | dominant |
| `L/R_shoulder_yaw` ROM | **193.7° / 188.0°** | small |
| `L/R_wrist_roll` ROM | **43.9° / 59.9°** | small |

The top-ROM joints are `shoulder_pitch` (259°/256°), `shoulder_yaw` (194°/188°) and
`elbow` (125°/116°) — a **20× arm-to-leg ROM ratio**, the inverse of every other clip here.

This matters because the 2026-09-04 results found `left_shoulder_yaw` and `left_wrist_roll`
were the *worst* reference-tracking joints (17–20° error) even on arm-quiet walking clips.
Clip 6 exercises `shoulder_yaw` through **~190°** of travel, so it should either confirm or
refute the Phase C.2 low-inertia-twist hypothesis far more decisively than any locomotion
clip can.

> **Expect a different failure mode.** With `dof` reaching −171° and near-zero leg motion,
> the balance and ankle-saturation dynamics that dominated clips 1–2 are largely absent
> here. Do not compare clip 6's aggregate `mpjpe_l` against the locomotion clips' as if
> they measured the same thing — report it as its own stratum (plan §5).

### Clip 7 is the dynamic-locomotion case

`jog_ff_start_180_R_002` is the first non-walking gait in the manifest and the fastest
clip by a wide margin:

| | clip 7 (jog) | clips 1–5 (walk) |
|---|---|---|
| root speed mean | **1.35 m/s** | 0.41–0.66 m/s |
| root speed peak | **2.32 m/s** | 0.88–1.35 m/s |
| joint speed mean | **69.3 °/s** | 22.1–40.2 °/s |
| joint speed peak | 777 °/s (`left_knee`) | 445–566 °/s |

It runs at **~2.1× the mean root speed** and **~1.9× the mean joint speed** of the fastest
walk clip. Its ROM is otherwise unremarkable (arms 43°, legs 50°) — what distinguishes it
is *rate*, not amplitude.

This is the natural probe for the 2026-09-04 finding that `walk_180` had 5× the shaking of
the sideway clip despite near-identical positional error: if that pattern is rate-driven,
clip 7 should amplify it further. It also stresses ankle-pitch saturation (the session's
worst joint, peaking at 21.7%) under genuinely higher ground-reaction loads.

> **Expect the tightest actuator margins here.** A 777 °/s knee transient at 1.35 m/s is
> the most demanding combination in the manifest, so this is the clip most likely to
> expose torque saturation or a fall. Treat a non-fall on clip 7 as a stronger result than
> a non-fall on any walk clip, and a fall here as less alarming than one on clip 2.

### Clips 7 and 8 are a matched pair (different performer, same motion)

`jog_ff_start_180_R_002__A265` and `jog_ff_start_180_R_002__A192_M` are the **same motion,
same take number, same 180° right turn**, performed by **different subjects** (A265 vs
A192) — and only the latter is mirrored. This is precisely the plan §0 definition of
Test-Repetition: *"new performances of known motion types."*

| | clip 7 (A265) | clip 8 (A192_M) | Δ |
|---|---|---|---|
| duration | 4.70 s | 5.04 s | +7% |
| root path | 6.34 m | 6.26 m | −1% |
| root speed mean / peak | 1.35 / 2.32 m/s | 1.24 / 2.14 m/s | −8% / −8% |
| joint speed mean / peak | 69.3 / 777 °/s | 64.4 / 702 °/s | −7% / −10% |
| leg ROM | 50.3° | 41.8° | **−17%** |
| arm ROM | 43.3° | 41.5° | −4% |

Same trajectory (paths agree to 1%), executed **~8% slower with 17% less leg ROM** — A192
jogs the same distance more conservatively. This is a **useful, deliberate confound**: it
isolates *performer variation* from *motion-type variation*, which no other pair in the
manifest does. Reps of the same clip measure robot/initial-condition noise; clips 7 vs 8
measure sensitivity to how a human performed the motion.

> **Do not average clips 7 and 8 into one "jog" number** without also reporting them
> separately. If the policy is robust to performer style they should agree closely, and a
> gap between them is itself the result — one that pooling would destroy. Given the 09-04
> finding that rep-to-rep variance alone spans 2.3× on `walk_180`, expect to need ≥3 reps
> of *each* before any clip-7-vs-8 difference is distinguishable from noise.

### Clip 9 is the first OOD entry — and the only one with a flight phase

`reach_jump_R_001` is a **reach-and-jump**: nearly stationary in the ground plane
(1.06 m path, net displacement ~0) but with large vertical travel.

| | clip 9 | clips 1–8 |
|---|---|---|
| **root height range** | **0.316 m** | 0.005–0.090 m |
| vertical velocity | −1.37 … +1.53 m/s | negligible |
| root speed mean / peak | 0.14 / **1.64** m/s | see §3 table |
| `dof` range | −151.4° … +77.3° | ≈ ±50–100° (except clip 6) |
| ROM arms / legs | 53.7° / 38.8° | — |

The **0.316 m root-height range is 3.5× the next-highest clip** (jog, 0.090 m) and 63× the
warm-up. Combined with ±1.4–1.5 m/s vertical velocity, this is unambiguous ballistic
motion — the manifest's first clip containing a genuine **flight phase and landing
impact**, and therefore its first true agility / Test-Content entry.

Note the low *mean* root speed (0.14 m/s) alongside a high *peak* (1.64 m/s): the motion is
mostly stationary reaching, punctuated by one explosive event. Mean-based summaries will
make this clip look mild; it is not.

> **This clip breaks assumptions the other eight share.** During flight there is no ground
> contact, so ankle-torque tracking, foot-position error and the tilt-based non-fall proxy
> all mean something different — the robot is *supposed* to leave the ground. The plan's
> §3 fall definition (pelvis height below threshold) is particularly fragile here, since
> the landing crouch may dip below a threshold tuned on walking clips. **Re-tune or
> special-case the fall detector before scoring clip 9**, and expect the landing impact —
> not the takeoff — to dominate torque saturation.

### Clips 10 and 11 are the descent/recovery pair — and the only viable ground-contact test

`kneeling_start_101` (stand → kneel) and `kneeling_stop_002` (kneel → stand) are the only
clips in all 160 that transition between standing and ground contact:

| | clip 10 (descent) | clip 11 (recovery) |
|---|---|---|
| pelvis height | 0.774 → **0.350 m** | 0.404 → **0.781 m** |
| height range | **0.424 m** (largest in manifest) | 0.377 m |
| net vertical displacement | **−0.37 m** | **+0.37 m** |
| `dof` max | **+155.1°** (deep knee flexion) | +152.2° |
| duration | 5.20 s | 2.74 s (2× faster ascent) |
| joint speed mean / peak | 13.2 / 171 °/s (slowest in manifest) | 22.8 / 339 °/s |

They are **near-exact inverses**: −0.37 m vs +0.37 m net vertical travel. Streaming 10 then
11 gives a complete stand → kneel → stand cycle.

**Why this pair matters for deployment.** The deploy binary's `InitControl()` ramps the
robot to `default_angles` (standing) and there is no mechanism to start from any other
pose. Clip 10 is the only clip that *begins* at standing height (0.774 m) and descends, so
it is the only way to reach ground contact from the robot's mandatory initial state:

- G1 at `default_angles` sits with pelvis **0.722 m** above the ankle links; clip 10 starts
  at **0.774 m** — close.
- Joint-space distance from `default_angles` to clip 10 frame 0: **13.4° mean**, worst joint
  `right_shoulder_yaw` 54.5° (an arm, not load-bearing).

So clip 10 splices onto the robot's init pose without a discontinuity. Chaining
10 → 11 is also clean: joint-space distance from clip 10's end to clip 11's start is
**7.7° mean / 32.8° max** (`left_wrist_roll`) — by far the best splice among all candidate
pairs (the next best, into a crouch clip, is 17.4° mean / 77.7° max).

> **These clips do *not* enable crawl.** True crawl clips (`turn_crawl_360_003`,
> `change_idle_crawl_to_idle_crawl_right_003`) start at **0.082–0.087 m** pelvis height —
> prone, torso down. Clip 10 ends at 0.405 m — an upright kneel. Splicing kneel → crawl
> costs **121° of knee/shoulder discontinuity in one frame**, which is not streamable. No
> clip in the 160 bridges kneel → prone, so crawl replay needs new retargeted data or a
> code change to `InitControl()` (see §5.7).

> **Safety before the first hardware attempt.** These are the only manifest clips with
> intended knee/shin ground contact — a load path the G1 is not designed for and the policy
> was likely not trained on. The fall detector is invalid here for the same reason as
> clip 9 (a legitimate kneel trips a height threshold tuned on walking). Run 10 → 11 in
> sim first; if the policy cannot hold a kneel in sim, hardware will not go better.

### Clip 12 is the manifest's extreme point on every dynamic axis

`high_jump_R_103` is a **maximal-effort vertical jump** — 2.60 s, the shortest clip here,
and the most violent by a wide margin. It supersedes clip 9 as the manifest's agility
ceiling:

| | clip 12 (high jump) | clip 9 (reach jump) | rest of manifest |
|---|---|---|---|
| duration | **2.60 s** (shortest) | 7.58 s | 2.74–11.20 s |
| **root height range** | **0.579 m** | 0.316 m | 0.005–0.424 m |
| root speed mean / peak | 0.60 / **2.27 m/s** | 0.14 / 1.64 | ≤1.35 / ≤2.32 |
| joint speed **mean** | **100.2 °/s** | 18.5 | 13.2–69.3 |
| joint speed **peak** | **1901 °/s** | 696 | 171–1037 |
| `dof` range | −150.7 … **+159.1°** | −151.4 … +77.3 | — |
| ROM arms / legs | **92.6° / 64.7°** | 53.7 / 38.8 | — |

Three records in one clip: **largest root-height range** (0.579 m — 1.8× clip 9, 37%
more than the kneel descent), **highest mean joint speed** (100 °/s — 1.4× the jog), and
**highest peak joint speed** (1901 °/s — 1.8× clip 6's previous record, on
`left_shoulder_pitch`). It is also the only clip that is simultaneously top-tier in *both*
ROM (arms 93°, legs 65°) and *rate*; clip 6 had the ROM without the speed, clip 7 the speed
without the ROM.

The top-ROM joints are `shoulder_pitch` (188°/185°), `left_shoulder_yaw` (170°) and
`knee` (154°/151°) — a full-body arm-swing-plus-leg-extension pattern, unlike clip 9's
arm-dominated reach.

> **This is the most likely clip in the manifest to produce a fall or a torque abort.** A
> 1901 °/s transient combined with a 0.58 m ballistic excursion and 154° of knee flexion
> exceeds every actuator envelope previously exercised. Two consequences: (a) the fall
> detector is invalid here for the same reason as clip 9 but more so — the flight phase is
> ~2× longer in vertical extent; (b) **run it in sim before hardware**, and treat a
> hardware attempt as a deliberate stress test, not a routine eval rep.

> **Also beware the short duration.** At 2.60 s, the 2 s `--settle` lead-in is nearly as
> long as the clip itself, and only ~130 streamed frames survive after trimming. Per-clip
> metrics on clip 12 will be far noisier than on the 5–11 s clips, so it needs more reps,
> not fewer, despite being the quickest to run.

### Clip 13 is the first clip that is fast in the arms *and* the legs at once

`dance_vouge_shake_it_babe_360_R_002` is the manifest's first **dance** entry: a vogue
routine with a 360° right turn performed **while travelling 3 m**. It holds the manifest's
**highest mean joint speed — 126.1 °/s** — ahead of the high jump (100.2) and nearly 2×
the jog (69.3).

| | clip 13 (vogue) | clip 6 (warm-up) | clip 7 (jog) | clip 12 (high jump) |
|---|---|---|---|---|
| joint speed **mean** | **126.1 °/s** | 39.8 | 69.3 | 100.2 |
| joint speed peak | 1190 °/s | 1037 | 777 | 1901 |
| arm ROM | **84.4°** | 110.8 | 43.3 | 92.6 |
| leg ROM | **51.3°** | 5.6 | 50.3 | 64.7 |
| root path | **3.72 m** | 0.37 | 6.34 | 1.57 |
| root speed mean | 0.58 m/s | 0.04 | 1.35 | 0.60 |
| height range | 0.152 m | 0.005 | 0.090 | 0.579 |

Every other high-speed clip is specialised: clip 6 has arm ROM with a stationary base,
clip 7 has locomotion with quiet arms, clip 12 has a ballistic burst over 2.6 s. Clip 13 is
the only one that sustains **84° arm ROM and 51° leg ROM simultaneously while covering
3.7 m for 6.4 s** — the closest thing in the manifest to a full-body coordination test.

Its `dof` range (−92 … +78°) is the *mildest* of the OOD clips, and its height range
(0.152 m) is modest — so the difficulty here is **coordination and rate**, not extreme
posture. The top-ROM joints (`shoulder_pitch` 131°, `right_shoulder_yaw` 130°, `elbow`
127°/127°) plus `right_wrist_roll` at **86°** — the largest forearm-twist travel of any
manifest clip — make it a genuine §0.1 stress entry as well, and unlike clip 6 it applies
that twist while the legs are loaded.

Note also the 360° turn. This is the clip most likely to expose the unresolved `root_rot`
heading issue flagged at the end of this section: the root translation shows a clean +3.05 m
lateral path, so if the heading channel again fails to accumulate ~360°, that confirms the
problem is in `root_rot` rather than in any one clip.

> **Expect the yaw/waist chain to be the bottleneck.** A 360° turn under 126 °/s mean joint
> rate loads `waist_yaw` and `hip_yaw` in a way no walk or jog clip does, and the 2026-09-04
> results already showed the waist joints sitting near zero tracking correlation. If those
> joints are the failure mode, this is the clip that will show it.

### Clips 13 and 14 are a travelling/in-place dance pair

`dance_hiphop_mike_tyson_R_fast_001` is the second dance entry and the manifest's
**second-fastest clip by mean joint speed (108.7 °/s)**, behind only the vogue clip. Its
value is that it isolates the *dance* difficulty from the *locomotion* difficulty:

| | clip 13 (vogue) | clip 14 (hip-hop) |
|---|---|---|
| duration | 6.38 s | 4.84 s |
| root path | **3.72 m** (travels) | **2.03 m** (returns to start) |
| net displacement | +3.05 m lateral | (−0.13, −0.16) m ≈ **0** |
| root speed mean / peak | 0.58 / 1.25 m/s | 0.42 / 1.01 m/s |
| joint speed mean / peak | 126.1 / 1190 °/s | 108.7 / 1171 °/s |
| arm ROM | 84.4° | 78.2° |
| leg ROM | 51.3° | **62.1°** |
| `dof` range | −92.3 … +77.7° | −101.2 … **+126.4°** |
| height range | 0.152 m | 0.112 m |

Both sustain ~110–126 °/s mean joint rate with ~80° arm ROM, but **clip 13 covers 3.7 m
while clip 14 ends where it started**. Any difference between them is therefore
attributable to base translation rather than to limb dynamics — the same logical trick as
the clip 7/8 performer pair, applied to a different axis.

Clip 14 is also the more **leg-aggressive** of the two: 62.1° leg ROM and `right_knee`
reaching **+126.4°** (the clip's `dof` maximum) against the vogue clip's 51.3°/+77.7°. The
bobbing, weight-shifting hip-hop stance flexes the knees far harder than the vogue
routine's travelling steps, so despite being the *slower* clip it puts more demand on the
knee actuators.

The top-ROM joints (`shoulder_pitch` 142°/141°, `right_shoulder_yaw` 141°, `elbow`
141°/138°) show an upper body working through roughly the same envelope as clip 13, which
is what makes the leg-side contrast interpretable.

> **Run 13 and 14 back to back and report them separately.** Pooling them into one "dance"
> number discards the in-place/travelling contrast that is the only reason to carry both.
> Note clip 14's near-zero net displacement also makes it one of the few OOD clips that is
> **safe to run in a confined space** — worth scheduling first if floor area is limited.

### Naming convention

`<motion type>_<parameter>_<take>__<subject><_M mirrored>`

- `walk_180_R_003__A332_M` — walking with a **180° right turn**, take 003, subject A332,
  **M** = mirrored.
- `walk_sideway_045_stop_005__A042_M` — **sideways walk at 45°** ending in a **stop**,
  take 005, subject A042, mirrored.
- `walk_backward_start_001__A030_M` — **backward walk**, **start** phase (from standstill),
  take 001, subject A030, mirrored.
- `walk_sideway_135_loop_003__A022` — **sideways walk at 135°**, **loop** phase (steady
  state, no start/stop), take 003, subject A022, **not** mirrored.
- `walk_ff_stop_225_R_002__A266_M` — **forward walk** (`ff`) turning **225° right** and
  ending in a **stop**, take 002, subject A266, mirrored.
- `warm_up_chest_003__A359_M` — **chest warm-up** exercise (in-place arm/torso motion,
  no locomotion), take 003, subject A359, mirrored.
- `jog_ff_start_180_R_002__A265` — **jogging forward** (`ff`), **start** phase, turning
  **180° right**, take 002, subject A265, **not** mirrored.
- `jog_ff_start_180_R_002__A192_M` — identical motion spec to the above, subject **A192**,
  **mirrored**. Same name up to the subject suffix ⇒ a controlled performer-variation pair.
- `reach_jump_R_001__A072_M` — **reach and jump** (`R` = right-handed reach), take 001,
  subject A072, mirrored. In-place ballistic motion with a flight phase.
- `kneeling_start_101__A063_M` — **kneeling**, **start** phase (stand → kneel descent),
  take 101, subject A063, mirrored.
- `kneeling_stop_002__A051_M` — **kneeling**, **stop** phase (kneel → stand recovery),
  take 002, subject A051, mirrored. Note the different take/subject from clip 10, so the
  10 → 11 chain is not a single continuous performance (see §3 for the splice quality).
- `high_jump_R_103__A389_M` — **high jump** (`R` = right-side lead), take **103** (the
  1xx take numbers mark a separate recording block, as in clip 10), subject A389,
  mirrored. Maximal-effort vertical ballistic motion.
- `dance_vouge_shake_it_babe_360_R_002__A318_M` — **vogue dance**, choreography name
  *"shake it babe"*, with a **360° right turn**, take 002, subject A318, mirrored. The
  choreography name occupies the parameter slot that carries gait parameters elsewhere,
  so dance clips do not parse the same way as the locomotion names above.
- `dance_hiphop_mike_tyson_R_fast_001__A319_M` — **hip-hop dance**, routine name
  *"mike tyson"*, `R` lead, **`fast`** tempo qualifier, take 001, subject A319, mirrored.
  The `fast` token is a tempo marker, not a gait phase; other takes of the same routine at
  other tempos exist in `eval_subset` if a speed-matched pair is ever wanted.

The phase suffixes (`start` / `loop` / `stop`) matter for evaluation: `loop` clips are
steady-state and should be the *easiest* to track, while `start`/`stop` contain the
acceleration transients where the 2026-09-04 results showed ankle-pitch saturation
peaking. The manifest now covers all three phases.

Source recordings live under `~/ego_dataset/smpl_bones_seed/230418/` and
`~/ego_dataset/smpl_bones_seed/220728/` respectively (BVH originals under
`work_bearlu/data/bones-studio-seed/soma_uniform/bvh/`).

> **Open question**: `walk_180_R_003` should contain a 180° turn, but the heading derived
> from `root_rot` totals only ~9° over the clip, while root translation shows a clean 7 m
> straight-ish path. Either `root_rot` is not global heading, or the turn is encoded
> elsewhere. This is the same unresolved issue as the report's §7 heading caveat — resolve
> before using `root_rot` for anything.

---

## 4. Where these clips were replayed on the robot

Session 2026-09-04 (`~/g1_robot_data/`). Episode windows located by
`sim2real/locate_clip_in_session.py`; full derivation in
`sim2real/online_eval_20260904_report.md` §2.

| # | clip | stream session @offset | policy run | run-relative window |
|---|---|---|---|---|
| 1 | `walk_sideway_045_stop_005__A042_M` | `streamed_092422` @5547 | `g1_deploy_run_09042026_run5` | 127.56–133.10 s |
| 2 | `walk_sideway_045_stop_005__A042_M` | `streamed_092026` @104 | `g1_deploy_run_09042026_run5` | 38.52–44.06 s |
| 3 | `walk_180_R_003__A332_M` | `streamed_091340` @105 | `g1_deploy_run_09042026_run3` | 18.02–29.22 s |
| 4 | `walk_180_R_003__A332_M` | `streamed_091541` @104 | `g1_deploy_run_09042026_run4` | 37.20–48.40 s |
| 5 | `walk_180_R_003__A332_M` | `streamed_091118` @105 | `g1_deploy_run_09042026_run2` | 20.50–31.70 s |

Streaming command used (mode 2, ZMQ to the robot):

```bash
.venv_teleop/bin/python ./data_process/stream_clip_mode2.py \
    --path ../ego_dataset/eval_subset/smpl/<clip>.pkl \
    --fps 50 --settle 2.0 --chunk-frames 10 --visualize \
    --host 192.168.8.192 --port 5556
```

`--settle 2.0` holds the first frame for 2 s; the resulting ~104-frame lead-in is excluded
from all metrics.

---

## 5. Known gaps in this manifest

1. **OOD content is started but thin.** Clips 9–14 are the Test-Content entries — two jumps
   with flight phases, a kneel descent/recovery pair, and two dances. Clips 1–5, 7, 8 are
   basic locomotion and clip 6 is an in-place warm-up, all plausibly ID. Six OOD clips is a
   start, not a category: `~/ego_dataset/eval_subset/` holds **~160 paired clips** with
   plenty more Test-Content material (`flip_360_004`, other `dance_hiphop_*`,
   `turn_crawl_360_003`, `jump_ff_*`, `crawl_ff_stop_225_R_002`,
   injured-gait variants). This remains a *selection* gap, not a data gap.
2. **The §0.1 stress subset is started but thin.** Clip 6 (`warm_up_chest_003`) is the
   first clip that actually exercises the low-inertia twist axes — `shoulder_yaw` through
   ~190°, `wrist_roll` 44–60°. Clip 12 adds `wrist_roll` 68–76° and clip 13 the manifest
   maximum at **86°** (`right_wrist_roll`), the latter under leg load — but in both cases
   as a by-product of the motion rather than as an isolated probe (clip 14 is milder at
   38–59°). These are **four clips with no robot data yet**, and none isolates forearm
   twist. A dedicated twist clip is still worth adding; candidates already in the dataset:
   `mid_small_valve_ccw_002`, `nailing_floor_R_004`, `brush_of_dust_002`, `itching_*`.
3. **Clips 3–14 have no robot episodes yet** (§1) — all must be replayed with
   `--host 192.168.8.192` before they contribute anything. **Twelve fourteenths of the
   manifest is currently unmeasured**, and the gap widens with each addition: the measured
   portion is still only the two clips from 2026-09-04.
4. **Only 2 clips with data, 2–3 reps each.** The plan calls for 5–10 clips per category.
   Observed rep-to-rep variance is large (`walk_180` spans 29.8–67.8 mm `mpjpe_l`), so
   both more clips and more reps are needed before per-policy differences resolve. Note
   the clip 7/8 pair and the clip 13/14 pair each need ≥3 reps *per member* to be
   interpretable (§3), and clip 12's 2.60 s duration makes it the noisiest per-rep entry
   in the manifest.
5. **The fall detector is not valid for clips 9, 10, 11 or 12.** The plan's §3 pelvis-height
   threshold and the evaluator's 35° tilt proxy were both tuned on walking. A jump's flight
   phase (0.58 m of vertical excursion on clip 12), and a kneel's legitimate 0.35 m pelvis
   height, will both trip them spuriously. Fix before scoring these clips (§3).
6. **Crawl is still unreachable.** Clips 10/11 reach a kneel (0.35–0.40 m) but true crawl
   clips start prone at 0.082–0.087 m, and the kneel → prone splice costs 121° of
   single-frame discontinuity. Options: retarget a bridging clip, or modify
   `InitControl()` in `g1_deploy_onnx_ref.cpp` to ramp toward the reference motion's first
   frame instead of `default_angles` (with the robot physically pre-placed).
7. **A sixth streamed session (`streamed_090954`, 27.9 s) is still unidentified** — it
   matched none of the twelve manifest clips. With ~160 candidates in `eval_subset`, a bulk
   sweep of `locate_clip_in_session.py` over the whole directory would identify it — but
   **the safety margin is narrower than it first appeared**. Observed best-non-match
   distances have fallen steadily as clips were added: 675 → 437 → 142 → 120 → **62 mm**
   (clip 9), against a 4.6–6.7 mm true-match band. That is still ~10×, but a bulk sweep
   must report the **top-k matches and their separation**, not the argmin alone, and should
   treat any match above ~20 mm as unconfirmed.

---

## 6. Change log

| version | date | change |
|---|---|---|
| v1 | 2026-09-08 | Initial freeze: 2 clips, 5 robot episodes from the 2026-09-04 session. |
| v2 | 2026-09-08 | Added `walk_backward_start_001__A030_M` (no robot episodes yet). Corrected §5.1: `eval_subset` does contain OOD content (~160 clips); the gap is selection, not data. |
| v3 | 2026-09-08 | Added `walk_sideway_135_loop_003__A022` (no robot episodes yet). Manifest now covers all three gait phases (start / loop / stop) and four facing directions. |
| v4 | 2026-09-08 | Added `walk_ff_stop_225_R_002__A266_M` (no robot episodes yet). Fixed the §3 property-table header, which was one column short since v3. |
| v5 | 2026-09-08 | Added `warm_up_chest_003__A359_M` (no robot episodes yet) — first non-locomotion clip and first member of the §0.1 stress subset (shoulder_yaw ROM ~190°, arm/leg ROM ratio 20×). |
| v6 | 2026-09-08 | Added `jog_ff_start_180_R_002__A265` (no robot episodes yet) — first non-walking gait; 2.1x the root speed and 1.9x the joint speed of the fastest walk clip. Added root/joint speed rows to the §3 table. |
| v7 | 2026-09-08 | Added `jog_ff_start_180_R_002__A192_M` (no robot episodes yet) — same motion/take as clip 7 by a different performer, giving the manifest its first controlled performer-variation pair. |
| v8 | 2026-09-08 | Added `reach_jump_R_001__A072_M` (no robot episodes yet) — first Test-Content (OOD) clip and the only one with a flight phase (root height range 0.316 m, 3.5x the next clip). Added a root-height row to the §3 table; flagged that the fall detector is invalid for this clip. |
| v9 | 2026-09-08 | Added `kneeling_start_101__A063_M` and `kneeling_stop_002__A051_M` (no robot episodes yet) — the only stand<->ground transitions in the dataset and the only route to ground contact from the robot's mandatory standing init pose. Transposed the §3 property table to clip-per-row. |
| v10 | 2026-09-11 | Added `high_jump_R_103__A389_M` (clip 12), `dance_vouge_shake_it_babe_360_R_002__A318_M` (clip 13) and `dance_hiphop_mike_tyson_R_fast_001__A319_M` (clip 14), none replayed. Clip 12 sets manifest records for root-height range (0.579 m) and peak joint speed (1901 °/s); clips 13/14 are the two fastest by *mean* joint speed (126.1 / 108.7 °/s) and form a travelling-vs-in-place dance pair. Added `sim2real/clip_stats.py`, which regenerates the §3 table rows. |

When adding clips, bump the version, append here, and **re-run all policies** on the new
list — never compare numbers computed against different manifest versions.
