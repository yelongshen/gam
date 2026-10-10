# Early failures after the retargeting fix (base checkpoint novr_050k, motion-only, no chair)

t_fail in seconds of the clip (30 steps/s). `*` = failed within 1 s. Reasons: foot = foot_pos_xyz, ee = ee_body_pos, ori = anchor_ori_full.


## picoset_20260928: early failures AFTER the fix: <1 s: 4   <3 s: 11   (of 27 clips)

| clip | t_fail (s) | <1 s | reason | robot lean / human lean (deg) | frame-0 foot z (m) | first-frame speed (rad/s) | re-retargeted | t_fail before (s) |
|---|---:|:---:|---|---:|---:|---:|:---:|---:|
| pico0928_clip_018 | 0.30 | * | foot | 3.5 / 3.2 | 0.486 | 1.4 | no | 0.30 |
| pico0928_clip_021 | 0.50 | * | foot | 2.8 / 2.5 | 0.166 | 2.9 | no | 0.50 |
| pico0928_clip_003 | 0.87 | * | foot | 2.8 / 4.4 | 0.153 | 2.3 | no | 0.87 |
| pico0928_clip_001 | 0.90 | * | foot | 3.3 / 5.3 | 0.021 | 3.3 | no | 0.90 |
| pico0928_clip_017 | 1.03 |  | foot | 5.5 / 5.8 | 0.055 | 1.4 | no | 1.03 |
| pico0928_clip_005_006 | 1.27 |  | foot | 2.3 / 4.1 | 0.200 | 0.9 | no | 1.30 |
| pico0928_clip_007 | 1.47 |  | foot | 3.3 / 3.8 | 0.058 | 0.7 | no | 1.47 |
| pico0928_clip_013 | 1.53 |  | foot | 13.6 / 34.6 | 0.151 | 0.8 | no | 1.53 |
| pico0928_clip_010 | 1.77 |  | foot | 9.2 / 10.6 | 0.090 | 8.8 | no | 1.77 |
| pico0928_clip_015 | 1.93 |  | foot | 4.8 / 9.0 | 0.027 | 0.9 | no | 1.93 |
| pico0928_clip_023 | 2.17 |  | foot | 3.0 / 3.1 | 0.031 | 3.8 | no | 2.17 |

## picoset_20260929: early failures AFTER the fix: <1 s: 11   <3 s: 45   (of 163 clips)

| clip | t_fail (s) | <1 s | reason | robot lean / human lean (deg) | frame-0 foot z (m) | first-frame speed (rad/s) | re-retargeted | t_fail before (s) |
|---|---:|:---:|---|---:|---:|---:|:---:|---:|
| pico0929_clip_077 | 0.23 | * | foot | 13.4 / 7.2 | 0.099 | 2.7 | no | 0.23 |
| pico0929_clip_048 | 0.30 | * | foot | 21.5 / 19.6 | 0.093 | 0.4 | yes | 0.07 |
| pico0929_clip_101 | 0.33 | * | ee | 6.9 / 8.7 | 0.037 | 3.9 | no | 0.33 |
| pico0929_clip_032 | 0.40 | * | foot | 8.8 / 4.7 | 0.042 | 1.5 | no | 0.40 |
| pico0929_clip_054 | 0.50 | * | foot | 22.7 / 31.2 | 0.124 | 7.3 | no | 0.50 |
| pico0929_clip_085 | 0.77 | * | foot | 6.5 / 6.2 | 0.081 | 2.4 | no | 0.77 |
| pico0929_clip_098 | 0.80 | * | foot | 6.6 / 2.7 | 0.047 | 0.0 | yes | 0.30 |
| pico0929_clip_011 | 0.83 | * | foot | 89.1 / 108.8 | 0.234 | 2.1 | yes | 0.10 |
| pico0929_clip_162 | 0.83 | * | foot | 6.8 / 2.8 | 0.045 | 0.2 | yes | 0.90 |
| pico0929_clip_067 | 0.87 | * | foot | 9.6 / 8.6 | 0.040 | 3.2 | no | 0.87 |
| pico0929_clip_010 | 0.97 | * | foot | 9.7 / 15.4 | 0.113 | 0.8 | no | 0.97 |
| pico0929_clip_053 | 1.00 |  | foot | 22.7 / 24.4 | 0.123 | 7.9 | no | 1.00 |
| pico0929_clip_119 | 1.10 |  | foot | 14.4 / 12.7 | 0.052 | 0.1 | yes | 0.07 |
| pico0929_clip_035 | 1.23 |  | foot | 6.1 / 5.1 | 0.087 | 0.2 | yes | 1.23 |
| pico0929_clip_052 | 1.30 |  | foot | 9.2 / 5.7 | 0.037 | 7.0 | no | survived |
| pico0929_clip_065 | 1.30 |  | foot | 12.7 / 12.6 | 0.067 | 2.2 | yes | 3.17 |
| pico0929_clip_084 | 1.30 |  | foot | 4.5 / 4.7 | 0.064 | 3.1 | no | 1.30 |
| pico0929_clip_149 | 1.33 |  | foot | 6.0 / 5.5 | 0.057 | 9.5 | no | 1.33 |
| pico0929_clip_088 | 1.43 |  | foot | 25.8 / 29.2 | 0.084 | 0.3 | yes | 0.10 |
| pico0929_clip_038 | 1.67 |  | foot | 5.3 / 6.1 | 0.097 | 1.0 | no | 1.67 |
| pico0929_clip_103 | 1.70 |  | foot | 11.4 / 7.8 | 0.054 | 5.0 | no | 1.70 |
| pico0929_clip_148 | 1.73 |  | foot | 6.3 / 4.9 | 0.063 | 8.3 | no | 1.73 |
| pico0929_clip_157 | 1.73 |  | foot | 13.2 / 7.6 | 0.039 | 2.2 | no | 1.73 |
| pico0929_clip_152 | 1.77 |  | foot | 13.3 / 12.4 | 0.057 | 0.3 | yes | 2.50 |
| pico0929_clip_156 | 1.80 |  | foot | 4.2 / 5.9 | 0.074 | 7.0 | no | 1.80 |
| pico0929_clip_134 | 1.83 |  | foot | 6.0 / 4.6 | 0.065 | 6.1 | no | 1.83 |
| pico0929_clip_037 | 1.87 |  | foot | 11.7 / 8.5 | 0.039 | 2.2 | no | 1.87 |
| pico0929_clip_111 | 1.87 |  | foot | 8.0 / 3.1 | 0.055 | 0.1 | yes | 0.07 |
| pico0929_clip_070 | 1.93 |  | foot | 12.1 / 9.9 | 0.076 | 2.1 | no | 1.93 |
| pico0929_clip_127 | 1.97 |  | foot | 4.2 / 5.2 | 0.068 | 3.0 | no | 1.97 |
| pico0929_clip_153 | 1.97 |  | foot | 10.1 / 4.5 | 0.079 | 0.2 | yes | 0.07 |
| pico0929_clip_087 | 2.07 |  | foot | 5.9 / 3.5 | 0.056 | 2.4 | no | 2.07 |
| pico0929_clip_123 | 2.10 |  | foot | 12.5 / 9.9 | 0.043 | 1.8 | no | 2.10 |
| pico0929_clip_060 | 2.13 |  | foot | 5.3 / 6.5 | 0.059 | 0.2 | yes | 2.63 |
| pico0929_clip_021 | 2.17 |  | foot | 6.0 / 5.0 | 0.045 | 2.5 | no | 2.17 |
| pico0929_clip_039 | 2.17 |  | foot | 7.4 / 4.7 | 0.053 | 1.4 | no | 2.17 |
| pico0929_clip_090 | 2.20 |  | foot | 8.6 / 2.3 | 0.082 | 1.5 | yes | 0.27 |
| pico0929_clip_019 | 2.30 |  | foot | 9.5 / 7.3 | 0.039 | 2.1 | yes | 2.30 |
| pico0929_clip_155 | 2.50 |  | foot | 6.2 / 4.9 | 0.074 | 1.4 | no | 2.50 |
| pico0929_clip_043 | 2.63 |  | foot | 7.0 / 8.6 | 0.055 | 0.2 | yes | 6.17 |
| pico0929_clip_069 | 2.77 |  | foot | 10.3 / 8.0 | 0.062 | 4.9 | no | 2.77 |
| pico0929_clip_028 | 2.90 |  | foot | 15.4 / 16.2 | 0.023 | 3.6 | no | 2.90 |
| pico0929_clip_062 | 2.90 |  | foot | 5.1 / 6.5 | 0.037 | 4.0 | no | 2.90 |
| pico0929_clip_097 | 2.90 |  | foot | 1.8 / 3.8 | 0.051 | 2.5 | no | 2.90 |
| pico0929_clip_046 | 2.97 |  | foot | 22.8 / 39.0 | 0.150 | 0.3 | yes | 0.30 |

## picoset_20260930: early failures AFTER the fix: <1 s: 4   <3 s: 20   (of 106 clips)

| clip | t_fail (s) | <1 s | reason | robot lean / human lean (deg) | frame-0 foot z (m) | first-frame speed (rad/s) | re-retargeted | t_fail before (s) |
|---|---:|:---:|---|---:|---:|---:|:---:|---:|
| pico0930_145524_clip_022 | 0.30 | * | ee | 32.6 / 35.9 | 0.260 | 3.5 | yes | 0.20 |
| pico0930_145524_clip_034 | 0.30 | * | foot | 17.4 / 22.4 | 0.300 | 6.2 | no | 0.30 |
| pico0930_145524_clip_021 | 0.33 | * | foot | 25.5 / 28.0 | 0.241 | 8.9 | no | 0.33 |
| pico0930_162422_clip_059 | 0.90 | * | foot | 20.3 / 25.9 | 0.063 | 7.1 | no | 0.90 |
| pico0930_162422_clip_044 | 1.10 |  | foot | 17.1 / 12.4 | 0.051 | 0.9 | no | 1.10 |
| pico0930_162422_clip_065 | 1.27 |  | foot | 11.4 / 22.3 | 0.080 | 2.6 | no | 1.27 |
| pico0930_162422_clip_035 | 1.30 |  | foot | 7.7 / 13.8 | 0.081 | 1.9 | no | 1.30 |
| pico0930_162422_clip_007 | 1.33 |  | foot | 18.1 / 11.7 | 0.080 | 3.3 | no | 1.33 |
| pico0930_162422_clip_016 | 1.33 |  | foot | 67.6 / 96.1 | 0.133 | 2.5 | no | 1.33 |
| pico0930_162422_clip_045 | 1.77 |  | foot | 17.0 / 17.2 | 0.082 | 1.7 | no | 1.77 |
| pico0930_162422_clip_048 | 1.87 |  | foot | 23.6 / 20.0 | 0.046 | 2.0 | no | 1.87 |
| pico0930_162422_clip_046 | 2.00 |  | foot | 21.2 / 14.7 | 0.051 | 1.6 | no | 2.00 |
| pico0930_162422_clip_040 | 2.07 |  | foot | 19.0 / 15.9 | 0.084 | 2.3 | no | 2.07 |
| pico0930_162422_clip_057 | 2.30 |  | foot | 17.5 / 16.3 | 0.047 | 1.2 | no | 2.30 |
| pico0930_145524_clip_012 | 2.47 |  | foot | 9.9 / 8.7 | 0.159 | 0.9 | no | 2.47 |
| pico0930_145524_clip_018 | 2.47 |  | foot | 8.4 / 6.4 | 0.042 | 0.8 | no | 2.47 |
| pico0930_162422_clip_053 | 2.57 |  | foot | 27.6 / 27.7 | 0.061 | 2.7 | yes | survived |
| pico0930_162422_clip_069 | 2.73 |  | foot | 14.7 / 10.9 | 0.047 | 0.7 | no | 2.73 |
| pico0930_162422_clip_058 | 2.93 |  | foot | 17.2 / 23.7 | 0.082 | 0.2 | yes | 2.93 |
| pico0930_162422_clip_062 | 2.93 |  | foot | 16.7 / 22.1 | 0.065 | 0.8 | yes | 2.50 |
