# PICO Raw Recording Session — 2026-09-24

Inventory and quality assessment of the live PICO VR SMPL recordings made on
**2026-09-24**, plus the one policy run logged alongside them.

- **Raw PICO data**: `/home/grease/g1_robot_data/pico_raw/<YYYYMMDD_HHMMSS>/`
- **Policy logs**: `gear_sonic_deploy/logs/24-09-26/<HH-MM-SS>/`
- **Producer**: `gear_sonic/scripts/pico_manager_thread_server.py --manager --target_fps 50
  --num_frames_to_send 4 --auto_pose --record_dir <dir>`
- **Policy under test**: `policy/sonic_no_vr_llam_080k` in **MuJoCo sim** (`run_sim_loop.py`,
  DDS on `lo`), `--input-type zmq` → encoder mode 2 (SMPL)
- **Totals**: 4 sessions, **73,673 chunks**, **580 MB**, ~25 min of wall-clock recording

See `gear_sonic_deploy/DEPLOYMENT_COMMANDS.md` §2b/§3 for how the stack is brought up, and
`model_eval/PICO_SMPL_STREAMING_DATASETS_NOTE.md` for the older PICO datasets.
Next session: `model_eval/PICO_RAW_SESSION_20260925.md` (sitting / chair teleop).

---

## 0. Where everything lives

| what | path | size | contents |
|---|---|---|---|
| **Raw PICO captures** | `/home/grease/g1_robot_data/pico_raw/` | **579 M** | 4 session dirs, `pose_%06d.npz` chunks (outside the repo) |
| ├ stalled session | `…/pico_raw/20260924_173425/` | 32 M | 3,969 chunks |
| ├ **primary take** | `…/pico_raw/20260924_173612/` | 427 M | 54,347 chunks |
| ├ secondary | `…/pico_raw/20260924_175748/` | 85 M | 10,809 chunks |
| └ rate-degraded | `…/pico_raw/20260924_181122/` | 36 M | 4,548 chunks |
| **Policy CSV logs** | `gear_sonic_deploy/logs/24-09-26/` | **494 M** | 4 run dirs; only `17-36-00/` is usable |
| **Session overview videos** | `data_visualization/pico_20260924/` | 4.0 M | `overview_<session>.mp4` ×4 |
| **Clip-detection reports** | `data_analysis/pico_20260924/` | 28 K | `clips_<session>.txt` ×4 |
| **This document** | `model_eval/PICO_RAW_SESSION_20260924.md` | — | — |

The raw captures and the policy logs are **outside the git repo** (they total ~1.1 GB); only
the videos, the detection reports and this doc are tracked.

---

## 1. Sessions

| session | chunks | wall clock | duration | PICO fps | size | head path | head xyz range | usable |
|---|---|---|---|---|---|---|---|---|
| `20260924_173425` | 3,969 | 17:34:31–17:35:52 | 76 s | **7.3** | 32 M | 6.9 m | 0.30 / 0.60 / 0.55 m | ❌ stalled stream |
| `20260924_173612` | 54,347 | 17:36:18–17:54:28 | **1090 s** | 89.6 | **427 M** | **62.2 m** | 1.04 / 1.00 / 1.17 m | ✅ **primary take** |
| `20260924_175748` | 10,809 | 17:57:54–18:01:46 | 232 s | 91.0 | 85 M | 14.0 m | 0.67 / 0.90 / 0.61 m | ✅ secondary |
| `20260924_181122` | 4,548 | 18:11:28–18:12:59 | 91 s | 71.9 | 36 M | 8.1 m | 0.36 / 0.89 / 0.47 m | ⚠️ degraded rate |

All four are **structurally complete**: contiguous `pose_000000.npz …`, `frame_index`
advancing 0 → N with no gaps, 21 keys per chunk.

`head path` / `head xyz range` are derived from `vr_position[:3]` (the head anchor) sampled
at ~300 points per session — a proxy for how much the operator actually moved.

### Rate is the discriminator

`pico_fps` is the **headset's source rate**, independent of the `--target_fps 50` chunking
(the recorder always writes ~50 chunks/s of 4 frames each; `pico_to_smpl_filtered.py`
resamples ~90 → 50 Hz on conversion).

- **89.6 / 91.0 Hz** (`173612`, `175748`) — nominal, the headset was tracking properly.
- **71.9 Hz** (`181122`) — ~20% frames dropped; usable but treat as lower quality.
- **7.3 Hz** (`173425`) — the body-tracking stream had effectively stalled. Its low motion
  content (14 mm mean inter-sample joint displacement, vs 224 mm in `173612`) is a
  consequence of that stall, not of the operator standing still. **Do not use.**

---

## 2. Chunk schema

Each `pose_%06d.npz` is exactly the packet that went out on ZMQ (written at
`pico_manager_thread_server.py:1730`, after `pack_pose_message`), so the recording is
byte-identical to what the policy consumed. 21 keys, `num_frames_to_send = 4` per chunk:

| key | shape | notes |
|---|---|---|
| `smpl_pose` | (4, 21, 3) | body-joint axis-angle, **no root** |
| `smpl_joints` | (4, 24, 3) | already root-local, Z-up |
| `body_quat_w` | (4, 4) | global root orientation, `[w,x,y,z]` |
| `body_pos_w`, `body_lin_vel`, `body_ang_vel` | (4, …) | root state |
| `joint_pos` | (4, 29) | **retargeted G1 dofs** (python-side retarget) |
| `joint_vel` | (4, 29) | zeros — not populated by the producer |
| `vr_position` / `vr_orientation` | (9,) / (12,) | head + L/R hand anchors |
| `left_hand_joints`, `right_hand_joints` | flattened | hand tracking |
| `left/right_trigger`, `left/right_grip` | (1,) | controller inputs |
| `toggle_data_collection`, `toggle_data_abort` | (1,) bool | operator markers |
| `frame_index` | (4,) int64 | monotonic, gap-free across a session |
| `pico_dt`, `pico_fps` | (1,) | source-rate telemetry |
| `timestamp_realtime`, `timestamp_monotonic` | (1,) | wall / monotonic clocks |
| `heading_increment` | (1,) | accumulated yaw change |

---

## 3. Paired policy logs

Only **one** PICO session has a matching policy log. `gear_sonic_deploy/logs/24-09-26/`:

| run | rows | duration | streamed span | encoder mode | pairs with |
|---|---|---|---|---|---|
| `17-16-53` | 39,338 | 786.7 s | none | 0 only | — (idle reference motion, no SMPL) |
| `17-35-00` | **0** | — | — | — | aborted, empty |
| `17-35-29` | 7 | 0.1 s | none | 0 only | aborted |
| `17-36-00` | 55,425 | **1108.5 s** | `[724:55425]` = **1094.0 s** | **2** for 1094.0 s | ✅ `20260924_173612` |

`17-36-00` is a clean pairing: it starts at 17:36:00, the PICO session's first chunk lands
at 17:36:18 (≈ the 14.5 s of `motion_name != "streamed"` lead-in), and both end at 17:54:28.
Encoder mode is **2 (SMPL) for the entire streamed span** — i.e. the policy really was
consuming the PICO stream, not falling back to mode 0.

The other three PICO sessions (`173425`, `175748`, `181122`) have **no policy log** —
`--enable-csv-logs` was not passed on those runs, so they are human-motion-only captures.

Logged CSVs in `17-36-00`: `q`, `dq`, `action`, `motor_torque`, `base_quat`, `base_accel`,
`base_ang_vel`, `torso_*`, `motor_temperature`, `motor_error`, `encoder_mode`,
`motion_name`, `motion_playing`, `token_state`, `left/right_hand_*`, `metadata.json`.

---

## 4. Visualization and clip detection

Both tools read the **raw `pose_*.npz` directory directly** — there is no need to convert to
`smpl_filtered` first, and converting first would *lose* information, because
`pico_to_smpl_filtered.py` defaults to `--transl_mode zero` (drops root translation).
Convert only when you need a `.pkl` clip for training/streaming.

### 4.1 Session overviews (whole recording, strided)

`data_visualization/pico_20260924/overview_<session>.mp4` — left panel: 24-joint SMPL
skeleton; right panel: VR 3-point (head + wrists) with trigger/grip state. Each video spans
the **entire** session.

```bash
n=$(ls /home/grease/g1_robot_data/pico_raw/<session> | wc -l); stride=$(( (n+599)/600 ))
.venv_sim/bin/python data_visual_script/visualize_pico.py \
    --dir /home/grease/g1_robot_data/pico_raw/<session> \
    --out data_visualization/pico_20260924/overview_<session>.mp4 \
    --stride $stride --fps 50 --max_frames 600
```

| session | chunk range | stride | rendered | playback | size |
|---|---|---|---|---|---|
| `20260924_173425` | `[0:3969)` | 7 | 567 | 7 fps | 707 K |
| `20260924_173612` | `[0:54347)` | 91 | 598 | 5 fps | 1.5 M |
| `20260924_175748` | `[0:10809)` | 19 | 569 | 5 fps | 1023 K |
| `20260924_181122` | `[0:4548)` | 8 | 569 | 6 fps | 844 K |

> **These are activity maps, not motion.** At stride 91 the 18-minute session is sampled
> every ~1.8 s, so the overview shows *where* activity happens, not what the motion looks
> like. For actual motion inspection render a range at stride 1 (§4.2).

### 4.2 Inspecting a single clip in real time

```bash
.venv_sim/bin/python data_visual_script/visualize_pico.py \
    --dir /home/grease/g1_robot_data/pico_raw/20260924_173612 \
    --out /tmp/clip.mp4 --start 33700 --end 34100 --stride 1 --fps 50
```

Good candidates from §4.3 (all in `20260924_173612` unless noted): `[33700,34100)` jumping
(37.5% airborne, the most ballistic segment of the day), `[7300,7600)` jumping,
`[6600,7100)` walking (root_v 0.63, fastest locomotion), `[4800,5200)` jumping in
`20260924_175748`.

### 4.3 Clip detection

Full reports in `data_analysis/pico_20260924/clips_<session>.txt`.

```bash
.venv_sim/bin/python data_process/detect_action_clips.py \
    --dir /home/grease/g1_robot_data/pico_raw/<session> --fps 50
```

| session | candidates | selected | selected duration | limb_energy p60 |
|---|---|---|---|---|
| `20260924_173425` | 7 | 5 | 32.0 s | 1.44 |
| `20260924_173612` | **129** | 12 | 88.0 s | **8.17** |
| `20260924_175748` | 28 | 10 | 54.0 s | 4.09 |
| `20260924_181122` | 9 | 9 | 53.0 s | 1.84 |
| **TOTAL** | **173** | **36** | **227.0 s** | — |

Selected clips by label: **walking 11, waving 7, gesture 7, jumping 6, agile 5**.

The `limb_energy` threshold tracks the session quality ranking from §1 exactly: **8.17** for
the primary take vs **1.44** for the stalled one — a 5.7× difference in how much the
operator's limbs actually moved per frame.

Only `20260924_173612` and `20260924_175748` contain genuine **jumping** with a measurable
airborne fraction (9.5–37.5%). The stalled session `173425` yields only `waving`/`walking`
labels with `root_v ≈ 0.01–0.04` — i.e. no real locomotion, consistent with §1.

> **`--fps 50` is mandatory.** Both tools previously derived their rate from `pico_fps`, the
> headset *source* rate (~90 Hz), while the sequence they build has **one sample per chunk**
> (~50/s). Without the override every duration is wrong by 1.8× — and by 6.8× on the 7.3 Hz
> stalled session. Both scripts now take `--fps` and warn when falling back. The same fix
> added `--start` / `--end` to `visualize_pico.py`, and corrected `--max_frames` so it caps
> *rendered* frames rather than truncating the source range (previously a large stride only
> ever showed the first `max_frames` chunks).

---

## 5. How to use these

### Convert raw → `smpl_filtered` clip

```bash
cd /home/grease/gam
.venv_teleop/bin/python data_process/pico_to_smpl_filtered.py \
    /home/grease/g1_robot_data/pico_raw/20260924_173612 ...
```

Produces `pose_aa (T,72)`, `transl`, `smpl_joints (T,24,3)`, `fps=50`, `original_pose_aa`,
`original_fps≈90` — see `SMPL_FILTERED_DATA_FORMAT.md`.

### Replay a session bit-for-bit

```bash
.venv_teleop/bin/python gear_sonic/scripts/pico_replay_server.py \
    --replay_dir /home/grease/g1_robot_data/pico_raw/20260924_173612 --fps 50
```

`pico_replay_server.py` re-packs with the same `pack_pose_message`, so a recorded session
reproduces the exact input stream — this is the way to A/B two policies on **identical**
human motion, which live teleop can never give you.

### Split the 18-minute take

`20260924_173612` is a single 1090 s recording containing many separate attempts. §4.3 has
already segmented it — 129 candidates, 12 selected. Re-run with different thresholds via
`--energy_pct` / `--min_dur_s` / `--top_n_per_label` if you want a different cut.

`toggle_data_collection` / `toggle_data_abort` in the chunks may also carry operator
markers — check whether they were actually pressed before relying on automatic segmentation.

---

## 6. Caveats

1. **`20260924_173425` is unusable** (7.3 Hz). Keep it only as a record of the failure mode.
2. **`20260924_181122` is rate-degraded** (71.9 Hz, ~20% dropped). Acceptable for
   qualitative work; do not mix it with the 90 Hz sessions in a quantitative comparison.
3. **Three of four sessions have no robot data.** They capture *human* motion only. Any
   tracking metric needs the paired run, i.e. only `173612` ↔ `17-36-00`.
4. **This is sim, not hardware.** `17-36-00` ran against MuJoCo via `run_sim_loop.py`, so
   its torque/saturation numbers are simulator outputs, not measured motor telemetry — not
   comparable to the real-robot sessions in `sim2real/online_eval_*_report.md`.
5. **The clip-identification path in `sim2real/eval_online_runs.py` does not apply here.**
   It matches against the frozen `eval_subset` library; live PICO motion is not in it. The
   robustness / control-loop metrics (saturation, shaking, tilt, cmd-vs-`q`) are still valid,
   but `mpjpe_l`-vs-reference is not defined for these runs.
6. **The recorder writes synchronously** inside the send loop (`np.savez_compressed` per
   chunk, ~50 files/s). Keep `--record_dir` on local disk; a network mount adds latency
   directly to the pose stream.
7. **Session stability was the limiting factor.** The headset client
   (`com.xrobotoolkit.client`) died repeatedly between takes whenever the headset slept —
   that is why there are four short sessions instead of one continuous one, and why two of
   them are rate-degraded. See `DEPLOYMENT_COMMANDS.md` §11 for the adb relaunch and the
   `svc power stayon usb` mitigation.
