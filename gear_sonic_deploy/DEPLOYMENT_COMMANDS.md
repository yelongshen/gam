# G1 Policy Deployment Commands

Reference for launching the G1 whole-body-control (WBC) pipeline in **MuJoCo simulation**
with either the **official release** policy or the **low-latency** policy.

---

## 1. Pipeline Overview

Three processes must run together (in this order):

| # | Component | Purpose | Port / Topic |
|---|-----------|---------|--------------|
| 1 | `run_sim_loop.py` | MuJoCo physics simulation (robot + DDS on `lo`) | DDS on `lo` |
| 2 | `pico_manager_thread_server.py` | PICO VR → SMPL pose → ZMQ publisher | PUB `5556` / topic `pose` |
| 3 | `g1_deploy_onnx_ref` | Policy inference (encoder + decoder) → motor commands | SUB `5556`, PUB `5557` / `g1_debug` |

---

## 2. Start the Simulator

```bash
cd /home/grease/gam
.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py
```

---

## 3. Start the PICO Manager (pose streaming)

Visualization (VR 3-point + SMPL body) is **enabled by default**.
`--auto_pose` starts streaming immediately — **no VR controller button presses required**.

```bash
cd /home/grease/gam
.venv_teleop/bin/python -u gear_sonic/scripts/pico_manager_thread_server.py \
    --manager \
    --target_fps 100 \
    --num_frames_to_send 4 \
    --auto_pose
```

> **Calibration:** `--auto_pose` calibrates from your pose at startup.
> Stand in the zero-reference pose when launching. Restart to re-calibrate.

### Useful variants

```bash
# Disable ALL visualization (max streaming performance)
... --manager --auto_pose --no_vis

# VR 3-point only, no SMPL body mesh
... --manager --auto_pose --no_vis_smpl

# Enable G1 waist tracking from VR head orientation
... --manager --auto_pose --waist_tracking

# Manual mode (requires controller: A+B+X+Y → PLANNER, then A+X → POSE)
... --manager
```

---

## 4. Start the Policy (choose ONE)

Run from `/home/grease/gam/gear_sonic_deploy`.

### 4a. Low-Latency Model  ⚡ (more responsive)

```bash
cd /home/grease/gam/gear_sonic_deploy
./target/release/g1_deploy_onnx_ref lo \
  policy/low_latency/model_decoder.onnx \
  reference/example/ \
  --obs-config policy/low_latency/observation_config.yaml \
  --encoder-file policy/low_latency/model_encoder.onnx \
  --input-type zmq \
  --zmq-host localhost \
  --zmq-port 5556 \
  --zmq-topic pose \
  --zmq-conflate \
  --disable-crc-check
```

### 4b. Official Release Model (baseline)

```bash
cd /home/grease/gam/gear_sonic_deploy
./target/release/g1_deploy_onnx_ref lo \
  policy/release/model_decoder.onnx \
  reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --input-type zmq \
  --zmq-host localhost \
  --zmq-port 5556 \
  --zmq-topic pose \
  --zmq-conflate \
  --disable-crc-check
```

> Only the three `policy/<name>/...` paths differ between 4a and 4b.

---

## 5. Model Comparison

| | Low-Latency | Official Release |
|---|---|---|
| Source | HF `nvidia/GEAR-SONIC` → `low_latency/` | HF `nvidia/GEAR-SONIC` (repo root) |
| Local path | `policy/low_latency/` | `policy/release/` |
| Encoder input dim | **1247** | **1762** |
| Decoder / policy dim | 994 | 994 |
| Encoder observations | 12 | 14 |
| SMPL lookahead | **4 frames** (~80 ms @ 50 Hz) | 10 frames (~200 ms @ 50 Hz) |
| `model_decoder.onnx` | 142.8 MB | 39.0 MB |
| `model_encoder.onnx` | 43.8 MB | 47.8 MB |

### SMPL encoder observation differences

| Purpose | Low-Latency | Official Release |
|---|---|---|
| SMPL joints | `smpl_joints_4frame_step1` (288) | `smpl_joints_10frame_step1` (720) |
| SMPL root orientation | `smpl_anchor_orientation_4frame_step1` (24) | `smpl_anchor_orientation_10frame_step1` (60) |
| Wrist joint positions | `motion_joint_positions_wrists_4frame_step1` (24) | `motion_joint_positions_wrists_10frame_step1` (60) |

> **Note:** "low latency" = shorter *temporal lookahead*, **not** a smaller/faster network.
> The low-latency decoder is ~3.7× larger, so per-inference GPU cost may be higher.
> Watch the `Policy:` field in the loop-timing log (budget is 20 000 µs at 50 Hz).

---

## 6. Downloading Models from Hugging Face

```bash
cd /home/grease/gam

# Low-latency model
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='nvidia/GEAR-SONIC',
                  allow_patterns=['low_latency/*'],
                  local_dir='gear_sonic_deploy/policy/low_latency_dl')
"
mkdir -p gear_sonic_deploy/policy/low_latency
mv gear_sonic_deploy/policy/low_latency_dl/low_latency/* gear_sonic_deploy/policy/low_latency/
rm -rf gear_sonic_deploy/policy/low_latency_dl
```

Required files per policy directory: `model_encoder.onnx`, `model_decoder.onnx`,
`observation_config.yaml`.

---

## 7. Verifying a Model Loads (dimension check)

Loads the model, validates observation dimensions, then exits. Useful after
changing `observation_config.yaml` or the C++ observation registry.

```bash
cd /home/grease/gam/gear_sonic_deploy
./target/release/g1_deploy_onnx_ref lo \
  policy/low_latency/model_decoder.onnx reference/example/ \
  --obs-config policy/low_latency/observation_config.yaml \
  --encoder-file policy/low_latency/model_encoder.onnx \
  --input-type zmq --disable-crc-check > /tmp/model_test.log 2>&1 &

sleep 30
grep -iE "Dimension match|Summary|Total dimension|Model dimension|mismatch" /tmp/model_test.log
pkill -f g1_deploy_onnx_ref
```

Expected (low-latency):
```
Policy Summary:  Total dimension: 994  / Model dimension: 994   ✓
Encoder Summary: Total dimension: 1247 / Model dimension: 1247  ✓
```

> **Always redirect to a log file.** Piping to `grep` buffers output, so if the
> process is killed the output is lost and it looks like nothing happened.

---

## 8. Useful Flags

| Flag | Effect |
|------|--------|
| `--disable-crc-check` | **Required for MuJoCo sim** (skips hardware CRC validation) |
| `--zmq-conflate` | Keep only the newest ZMQ message (drops stale queued poses) |
| `--policy-precision 16` | Run the policy in FP16 — use if `Policy:` timing is too high |
| `--planner-precision 16` | Run the planner in FP16 |
| `--planner-file <path>` | Enable the kinematic planner |
| `--enable-csv-logs` | Write CSV state logs |
| `--enable-motion-recording` | Record streamed / planner motion |
| `--logs-dir <path>` | Custom log output directory |
| `--zmq-verbose` | Verbose ZMQ subscriber logging |

---

## 9. Monitoring & Diagnostics

### Control-loop timing (printed ~1 Hz by the deploy binary)

```
Loop timing - LowState age: <N>ms, Streaming data mean delay: <N>ms,
              Obs: <N>us, Policy: <N>us, Obs 2 Motor Command: <N>us
```

- `Policy:` — policy inference time. Must stay well under **20 000 µs** (50 Hz budget).
- `Streaming data mean delay:` — end-to-end pose stream latency.

### Streaming health (`[StreamedMotionMerger]` / `[ZMQEndpointInterface]` lines)

- `did_catchup=0` → healthy, no forced resync.
- `streamed_motion_->timesteps` should stay **roughly flat**. Continuous growth
  (`4 → 8 → 16 → 27 …`) means the playback cursor is falling behind the live stream.
- `GetEncodeMode()=2` → SMPL-only mode (expected for PICO body tracking).

### End-to-end latency measurement

```bash
cd /home/grease/gam
.venv_teleop/bin/python gear_sonic/scripts/measure_latency.py --seconds 15
# Wave a hand distinctly during recording
```

### Verify the pose stream independently

```bash
.venv_teleop/bin/python gear_sonic/scripts/verify_pose_stream.py
```

### PICO body-tracking diagnostics

```bash
.venv_teleop/bin/python -u gear_sonic/scripts/pico_manager_thread_server.py \
    --xrt_diag --xrt_diag_seconds 12 --no_xrt_service
```

Healthy output shows `body_available=True` and non-empty `serials=[...]`.
`serials=[]` → PICO Motion Trackers are not paired/powered on.

---

## 10. Rebuilding the Deploy Binary

Required after editing `g1_deploy_onnx_ref.cpp` (e.g. adding observation-registry entries):

```bash
cd /home/grease/gam/gear_sonic_deploy/build
cmake --build . --target g1_deploy_onnx_ref -j"$(nproc)"
```

---

## 11. Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| First launch takes minutes | TensorRT compiles ONNX → `.trt` engine (one-time). Cached next to the ONNX. Low-latency decoder is 143 MB, so its first build is slow. |
| `Unknown observation function '<name>'` | The observation is missing from `GetObservationRegistry()` in `g1_deploy_onnx_ref.cpp`. Add the entry and rebuild (§10). |
| Observation dimension mismatch | `observation_config.yaml` doesn't match the ONNX. Verify with §7. |
| PICO stuck at `waiting for body data...` | Motion Trackers not paired/on. Run `--xrt_diag` (§9) and check `serials`. |
| Visualization window frozen | Manager is in `OFF` mode — use `--auto_pose`, or press A+B+X+Y then A+X on the controllers. |
| Robot lags further behind over time | Producer FPS > 50 Hz consumption. Adaptive catch-up handles this; confirm `timesteps` stays flat (§9). |
| `selected interface "lo" is not multicast-capable` | Harmless warning in loopback/sim mode. |

---

## 12. Quick Copy-Paste: Full Pipeline

Three separate terminals:

```bash
# Terminal 1 — simulator
cd /home/grease/gam && .venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py
```

```bash
# Terminal 2 — PICO pose streaming (visualization on by default)
cd /home/grease/gam && .venv_teleop/bin/python -u \
  gear_sonic/scripts/pico_manager_thread_server.py \
  --manager --target_fps 100 --num_frames_to_send 4 --auto_pose
```

```bash
# Terminal 3 — policy (swap low_latency ↔ release as needed)
cd /home/grease/gam/gear_sonic_deploy && ./target/release/g1_deploy_onnx_ref lo \
  policy/low_latency/model_decoder.onnx reference/example/ \
  --obs-config policy/low_latency/observation_config.yaml \
  --encoder-file policy/low_latency/model_encoder.onnx \
  --input-type zmq --zmq-host localhost --zmq-port 5556 --zmq-topic pose \
  --zmq-conflate --disable-crc-check
```
