# ICL hard set with settle (first frame held) - ICL_hardset_settle

`ICL_hardset_settle` = `ICL_hardset_aligned` with the first frame held before the motion (`motion_start_s` column; built by `data_process/build_settle_set.py` / `sim2real/build_grounded_settle_set.py`). **(HOLD)** = the policy failed during the hold, i.e. it could not stay stable at the start pose; **(+x)** = failed x seconds into the real motion; `ok` = ran to the end. `old_fail_s` = failure second without the hold (base100k).

| clip | motion_start_s | class | old_fail_s | base100k | mixv3b_010000 | mixv3b_014000 | mixv3b_020000 |
|---|---|---|---|---|---|---|---|
| fightAndSports1_subject1__w062s | 2.0 | hold | 0.18 | 0.20 (HOLD) | 0.22 (HOLD) | 0.24 (HOLD) | 0.20 (HOLD) |
| multipleActions1_subject1__w124s | 2.0 | hold | 0.24 | 0.26 (HOLD) | 0.24 (HOLD) | 0.22 (HOLD) | 0.22 (HOLD) |
| run1_subject5__w035s | 0.2 | hold | 0.22 | 0.24 (+0.04) | 0.14 (HOLD) | 0.16 (HOLD) | 0.18 (HOLD) |
| BMLmovi__BMLmovi__Subject_28_F_MoSh__Subject_28_F_21_poses | 2.0 | motion | 2.69 | 4.69 (+2.69) | 4.67 (+2.67) | 4.65 (+2.65) | 4.73 (+2.73) |
| CMU__CMU__05__05_16_stageii | 2.0 | motion | 1.41 | 3.41 (+1.41) | 3.61 (+1.61) | 3.41 (+1.41) | 3.41 (+1.41) |
| CMU__CMU__87__87_01_stageii | 2.0 | motion | 1.75 | 3.75 (+1.75) | 3.73 (+1.73) | 3.75 (+1.75) | 3.75 (+1.75) |
| WEIZMANN__WEIZMANN__67__Normal_StraightLong(4)_stageii | 2.0 | motion | 1.43 | 3.34 (+1.34) | 2.79 (+0.79) | 3.36 (+1.36) | 2.83 (+0.83) |
| dance1_subject1__w100s | 2.0 | motion | 0.12 | 2.34 (+0.34) | 2.16 (+0.16) | 0.86 (HOLD) | 1.22 (HOLD) |
| dance1_subject2__w067s | 2.0 | motion | 0.2 | 2.59 (+0.59) | 2.12 (+0.12) | 2.14 (+0.14) | 0.32 (HOLD) |
| dance1_subject3__w106s | 2.0 | motion | 2.79 | 3.31 (+1.31) | 4.95 (+2.95) | 5.35 (+3.35) | 6.55 (+4.55) |
| dance2_subject1__w088s | 2.0 | motion | 0.56 | 2.46 (+0.46) | 2.57 (+0.57) | 2.46 (+0.46) | 2.57 (+0.57) |
| dance2_subject2__w057s | 2.0 | motion | 0.7 | 2.38 (+0.38) | 2.55 (+0.55) | 2.71 (+0.71) | 4.29 (+2.29) |
| dance2_subject3__w128s | 2.0 | motion | 7.36 | 2.79 (+0.79) | 2.79 (+0.79) | 6.03 (+4.03) | 3.03 (+1.03) |
| dance2_subject4__w196s | 2.0 | motion | 0.92 | 2.89 (+0.89) | 2.93 (+0.93) | 2.89 (+0.89) | 2.89 (+0.89) |
| dance2_subject5__w110s | 2.0 | motion | 1.16 | 3.17 (+1.17) | 3.15 (+1.15) | 3.15 (+1.15) | 3.17 (+1.17) |
| fight1_subject2__w219s | 2.0 | motion | 2.09 | 4.11 (+2.11) | 4.65 (+2.65) | 4.67 (+2.67) | 4.17 (+2.17) |
| fight1_subject3__w148s | 2.0 | motion | 0.22 | 2.22 (+0.22) | 2.24 (+0.24) | 2.24 (+0.24) | 2.22 (+0.22) |
| fight1_subject5__w222s | 2.0 | motion | 1.9 | 3.83 (+1.83) | 2.53 (+0.53) | 3.95 (+1.95) | 3.85 (+1.85) |
| fightAndSports1_subject4__w028s | 2.0 | motion | 0.74 | 2.73 (+0.73) | 2.73 (+0.73) | 2.48 (+0.48) | 2.73 (+0.73) |
| flip_090_003__A304_M | 2.0 | motion | 2.18 | 4.20 (+2.20) | 4.18 (+2.18) | 4.20 (+2.20) | 4.18 (+2.18) |
| flip_360_004__A415 | 2.0 | motion | 2.37 | 4.37 (+2.37) | 4.37 (+2.37) | 4.37 (+2.37) | 4.37 (+2.37) |
| jumps1_subject1__w050s | 0.2 | motion | 0.3 | 0.52 (+0.32) | 0.52 (+0.32) | 0.52 (+0.32) | 0.46 (+0.26) |
| jumps1_subject2__w169s | 0.2 | motion | 0.32 | 0.60 (+0.40) | 0.54 (+0.34) | 0.62 (+0.42) | 0.60 (+0.40) |
| jumps1_subject5__w004s | 2.0 | motion | 0.3 | 2.20 (+0.20) | 2.18 (+0.18) | 2.20 (+0.20) | 2.63 (+0.63) |
| multipleActions1_subject2__w009s | 2.0 | motion | ok | ok | 2.26 (+0.26) | 2.32 (+0.32) | 2.30 (+0.30) |
| multipleActions1_subject3__w036s | 2.0 | motion | 0.38 | 2.10 (+0.10) | 2.26 (+0.26) | 2.26 (+0.26) | 2.24 (+0.24) |
| multipleActions1_subject4__w141s | 2.0 | motion | 0.66 | 2.06 (+0.06) | 2.06 (+0.06) | 2.24 (+0.24) | 2.24 (+0.24) |
| run1_subject2__w189s | 2.0 | motion | 0.22 | 2.10 (+0.10) | 2.10 (+0.10) | 2.10 (+0.10) | 2.14 (+0.14) |
| run2_subject1__w147s | 2.0 | motion | 2.87 | 4.87 (+2.87) | 4.83 (+2.83) | 4.83 (+2.83) | 4.83 (+2.83) |
| run2_subject4__w205s | 0.2 | motion | 0.34 | 0.42 (+0.22) | 0.28 (+0.08) | 0.24 (+0.04) | 0.24 (+0.04) |
| CMU__CMU__141__141_15_stageii | 2.0 | ok | ok | ok | ok | ok | ok |
| DFaust__DFaust__50021__50021_knees_stageii | 2.0 | ok | ok | ok | ok | ok | ok |
| KIT__KIT__200__Kniebeuge01_stageii | 2.0 | ok | ok | ok | ok | ok | ok |
| KIT__KIT__348__walking_fast07_stageii | 2.0 | ok | ok | ok | ok | ok | 3.68 (+1.68) |
| Transitions__Transitions__mazen_c3d__punchkarate_stand_stageii | 2.0 | ok | ok | ok | 3.57 (+1.57) | ok | ok |

| run | fail during hold | fail in motion | ok |
|---|---|---|---|
| base100k | 2 | 27 | 6 |
| mixv3b_010000 | 3 | 28 | 4 |
| mixv3b_014000 | 4 | 26 | 5 |
| mixv3b_020000 | 5 | 26 | 4 |
