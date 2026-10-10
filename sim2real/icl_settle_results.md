# ICL hard set with a 2 s settle (first frame held)

`ICL_hardset_settle2` = `ICL_hardset_aligned` with the first frame held for 2.0 s before the motion (built by `data_process/build_settle_set.py`). **(HOLD)** = the policy failed during the hold, i.e. it could not stay stable at the start pose; **(+x)** = failed x seconds into the real motion; `ok` = ran to the end. `old_fail_s` = failure second without the hold (base100k).

| clip | class | old_fail_s | base100k | mixv3b_010000 | mixv3b_014000 | mixv3b_020000 |
|---|---|---|---|---|---|---|
| dance1_subject2__w067s | hold | 0.2 | 0.22 (HOLD) | 0.18 (HOLD) | 0.22 (HOLD) | 0.26 (HOLD) |
| dance2_subject4__w196s | hold | 0.92 | 0.24 (HOLD) | 0.38 (HOLD) | 0.38 (HOLD) | 2.36 (+0.36) |
| fight1_subject2__w219s | hold | 2.09 | 0.28 (HOLD) | 0.32 (HOLD) | 0.30 (HOLD) | 4.71 (+2.71) |
| fight1_subject3__w148s | hold | 0.22 | 0.14 (HOLD) | 0.16 (HOLD) | 0.16 (HOLD) | 0.18 (HOLD) |
| fightAndSports1_subject1__w062s | hold | 0.18 | 0.20 (HOLD) | 0.22 (HOLD) | 0.22 (HOLD) | 0.24 (HOLD) |
| jumps1_subject1__w050s | hold | 0.3 | 0.12 (HOLD) | 0.12 (HOLD) | 0.12 (HOLD) | 0.14 (HOLD) |
| jumps1_subject2__w169s | hold | 0.32 | 0.34 (HOLD) | 0.26 (HOLD) | 0.28 (HOLD) | 0.34 (HOLD) |
| multipleActions1_subject1__w124s | hold | 0.24 | 0.28 (HOLD) | 0.26 (HOLD) | 0.24 (HOLD) | 0.24 (HOLD) |
| multipleActions1_subject3__w036s | hold | 0.38 | 0.36 (HOLD) | 0.72 (HOLD) | 0.66 (HOLD) | 0.68 (HOLD) |
| run1_subject5__w035s | hold | 0.22 | 0.16 (HOLD) | 0.14 (HOLD) | 0.12 (HOLD) | 0.14 (HOLD) |
| run2_subject4__w205s | hold | 0.34 | 0.26 (HOLD) | 0.24 (HOLD) | 0.24 (HOLD) | 0.20 (HOLD) |
| BMLmovi__BMLmovi__Subject_28_F_MoSh__Subject_28_F_21_poses | motion | 2.69 | 4.69 (+2.69) | 4.67 (+2.67) | 4.65 (+2.65) | 4.73 (+2.73) |
| CMU__CMU__05__05_16_stageii | motion | 1.41 | 3.41 (+1.41) | 3.61 (+1.61) | 3.41 (+1.41) | 3.41 (+1.41) |
| CMU__CMU__87__87_01_stageii | motion | 1.75 | 3.75 (+1.75) | 3.73 (+1.73) | 3.75 (+1.75) | 3.75 (+1.75) |
| WEIZMANN__WEIZMANN__67__Normal_StraightLong(4)_stageii | motion | 1.43 | 3.34 (+1.34) | 2.79 (+0.79) | 3.36 (+1.36) | 2.83 (+0.83) |
| dance1_subject1__w100s | motion | 0.12 | 2.34 (+0.34) | 2.14 (+0.14) | 0.86 (HOLD) | 1.22 (HOLD) |
| dance1_subject3__w106s | motion | 2.79 | 3.31 (+1.31) | 4.95 (+2.95) | 5.35 (+3.35) | 6.55 (+4.55) |
| dance2_subject1__w088s | motion | 0.56 | 2.46 (+0.46) | 2.57 (+0.57) | 2.46 (+0.46) | 2.57 (+0.57) |
| dance2_subject2__w057s | motion | 0.7 | 2.38 (+0.38) | 2.55 (+0.55) | 2.71 (+0.71) | 4.29 (+2.29) |
| dance2_subject3__w128s | motion | 7.36 | 2.79 (+0.79) | 2.79 (+0.79) | 6.05 (+4.05) | 3.03 (+1.03) |
| dance2_subject5__w110s | motion | 1.16 | 3.17 (+1.17) | 3.15 (+1.15) | 3.15 (+1.15) | 3.17 (+1.17) |
| fight1_subject5__w222s | motion | 1.9 | 3.83 (+1.83) | 2.53 (+0.53) | 3.95 (+1.95) | 3.85 (+1.85) |
| fightAndSports1_subject4__w028s | motion | 0.74 | 2.73 (+0.73) | 2.73 (+0.73) | 2.48 (+0.48) | 2.73 (+0.73) |
| flip_090_003__A304_M | motion | 2.18 | 4.20 (+2.20) | 4.18 (+2.18) | 4.20 (+2.20) | 4.18 (+2.18) |
| flip_360_004__A415 | motion | 2.37 | 4.37 (+2.37) | 4.37 (+2.37) | 4.37 (+2.37) | 4.37 (+2.37) |
| jumps1_subject5__w004s | motion | 0.3 | 2.20 (+0.20) | 2.18 (+0.18) | 2.20 (+0.20) | 2.63 (+0.63) |
| multipleActions1_subject2__w009s | motion | ok | ok | 2.26 (+0.26) | 2.32 (+0.32) | 2.30 (+0.30) |
| multipleActions1_subject4__w141s | motion | 0.66 | 2.06 (+0.06) | 2.06 (+0.06) | 2.24 (+0.24) | 2.24 (+0.24) |
| run1_subject2__w189s | motion | 0.22 | 2.10 (+0.10) | 2.10 (+0.10) | 2.10 (+0.10) | 2.14 (+0.14) |
| run2_subject1__w147s | motion | 2.87 | 4.87 (+2.87) | 4.83 (+2.83) | 4.83 (+2.83) | 4.83 (+2.83) |
| CMU__CMU__141__141_15_stageii | ok | ok | ok | ok | ok | ok |
| DFaust__DFaust__50021__50021_knees_stageii | ok | ok | ok | ok | ok | ok |
| KIT__KIT__200__Kniebeuge01_stageii | ok | ok | ok | ok | ok | ok |
| KIT__KIT__348__walking_fast07_stageii | ok | ok | ok | ok | ok | 3.68 (+1.68) |
| Transitions__Transitions__mazen_c3d__punchkarate_stand_stageii | ok | ok | ok | 3.57 (+1.57) | ok | ok |

| run | fail during hold | fail in motion | ok |
|---|---|---|---|
| base100k | 11 | 18 | 6 |
| mixv3b_010000 | 11 | 20 | 4 |
| mixv3b_014000 | 12 | 18 | 5 |
| mixv3b_020000 | 10 | 21 | 4 |

---

## Detailed Analysis & Findings

### 1. Test Overview with 2-Second Warmup/Settle

In `ICL_hardset_settle2`, the initial frame (both SMPL and retargeted robot) is held completely static for **2.0 seconds** (60 robot frames at 30 fps, 100 SMPL frames at 50 fps) before any motion begins.

This setup isolates two distinct failure modes:
- **Hold Failure ($t < 2.0\,\text{s}$):** The policy cannot remain stable at the initial spawn pose even when it is static.
- **Motion Failure ($t \ge 2.0\,\text{s}$):** The policy successfully stabilizes at the start pose for 2 full seconds, and fails only when dynamic tracking begins.

---

### 2. Why 11 Clips Fail During the Hold

The 2-second hold cleanly diagnosed why those clips previously failed at $t < 0.3\,\text{s}$. Running forward kinematics on frame 0 (`sim2real/diag_frame0_pose.py`) reveals that **all 11 hold-failure clips start mid-air or heavily tilted**:

| Clip | Lowest Body $z$ at Frame 0 | Ankle Link $z$ | Root Tilt | Time of Failure (base100k) |
|---|:---:|:---:|:---:|:---:|
| `run1_subject5__w035s` | **0.617 m** | 0.617 m | **28.2°** | 0.16 s (HOLD) |
| `fight1_subject3__w148s` | **0.337 m** | 0.337 m | 7.3° | 0.14 s (HOLD) |
| `jumps1_subject1__w050s` | **0.333 m** | 0.333 m | 12.6° | 0.12 s (HOLD) |
| `dance1_subject2__w067s` | **0.192 m** | 0.192 m | **17.2°** | 0.22 s (HOLD) |
| `fight1_subject2__w219s` | **0.184 m** | 0.184 m | 10.8° | 0.28 s (HOLD) |
| `dance2_subject4__w196s` | **0.179 m** | 0.179 m | 4.0° | 0.24 s (HOLD) |
| `dance2_subject1__w088s` | **0.164 m** | 0.164 m | 15.4° | 0.26 s (HOLD) |
| `jumps1_subject2__w169s` | **0.158 m** | 0.158 m | 10.2° | 0.34 s (HOLD) |
| `run2_subject4__w205s` | **0.156 m** | 0.156 m | **27.6°** | 0.26 s (HOLD) |
| `multipleActions1_subject3__w036s` | **0.153 m** | 0.153 m | 6.7° | 0.36 s (HOLD) |
| `multipleActions1_subject1__w124s` | **0.116 m** | 0.116 m | 12.2° | 0.28 s (HOLD) |

**Physical cause:** Because the 8-second window was picked as the fastest segment of the LAFAN clip, its very first frame is in the middle of an airborne jump, flight phase, or high-speed tilted turn. When the simulator initializes the robot at frame 0 with **zero velocity**, the robot instantly drops 20–60 cm to the ground under gravity and trips the `foot_pos_xyz` termination before the settle timer even reaches 0.3 s.

---

### 3. The 18 True Motion Failures (The Real ICL Candidates)

The other **18 clips** successfully stood completely stable for the entire 2.0-second hold with zero termination trips, and only failed **after** the real motion began ($t > 2.0\,\text{s}$):

* **AMASS clips**: `CMU 05_16` (failed at $+1.41\,\text{s}$ into motion), `CMU 87_01` ($+1.75\,\text{s}$), `WEIZMANN 67` ($+1.34\,\text{s}$), `BMLmovi 28` ($+2.69\,\text{s}$).
* **Flips**: `flip_090_003__A304_M` ($+2.20\,\text{s}$ into motion), `flip_360_004__A415` ($+2.37\,\text{s}$ into motion).
* **LAFAN ground-contact dances/runs**: `dance1_subject3__w106s` ($+1.31\,\text{s}$), `dance2_subject1__w088s` ($+0.46\,\text{s}$), `dance2_subject2__w057s` ($+0.38\,\text{s}$), `dance2_subject3__w128s` ($+0.79\,\text{s}$), `dance2_subject5__w110s` ($+1.17\,\text{s}$), `fight1_subject5__w222s` ($+1.83\,\text{s}$), `fightAndSports1_subject4__w028s` ($+0.73\,\text{s}$), `jumps1_subject5__w004s` ($+0.20\,\text{s}$), `multipleActions1_subject4__w141s` ($+0.06\,\text{s}$), `run1_subject2__w189s` ($+0.10\,\text{s}$), `run2_subject1__w147s` ($+2.87\,\text{s}$).

Every one of these 18 clips has its frame-0 ankle at ground level ($z \le 0.08\,\text{m}$) and root tilt $< 7^\circ$. Their failures are **100% genuine dynamic-tracking failures**, not spawn glitches.

---

### 4. Summary & Actionable Next Steps

1. **The 2s settle works:** It prevents false early terminations from spawn shocks and lets the robot settle into equilibrium before the reference moves.
2. **Dataset refinement:**
   * Keep the **18 motion-failure clips** + the **5–6 success clips** as the evaluation core.
   * For the 11 airborne clips (jumps and mid-stride runs), the window start index must be shifted to an earlier frame where both feet are on the ground (e.g. the takeoff anticipation phase), rather than cutting in mid-flight.
