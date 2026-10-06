# G1 Policy Deployment Commands

Reference for launching the G1 whole-body-control (WBC) pipeline in **MuJoCo simulation**
with either the **official release** policy or the **low-latency** policy.

---

## 1. Pipeline Overview

Three processes must run together (in this order):

| # | Component | Purpose | Port / Topic |
|---|-----------|---------|--------------|
| 1 | `run_sim_loop.py` | MuJoCo physics simulation (robot + DDS on `lo`) | DDS on `lo` |
| 2 | `RoboticsServiceProcess` + PICO client | XRoboToolkit PC service ⇄ headset app (body tracking source) | TCP `63901` (headset), `127.0.0.1:60061` (SDK) |
| 3 | `pico_manager_thread_server.py` | PICO VR → SMPL pose → ZMQ publisher | PUB `5556` / topic `pose` |
| 4 | `g1_deploy_onnx_ref` | Policy inference (encoder + decoder) → motor commands | SUB `5556`, PUB `5557` / `g1_debug` |

---

## 2. Start the Simulator

```bash
cd /home/grease/gam
.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py
```

---

## 2b. Start the XRoboToolkit Stack (PC service + PICO client via adb)

`pico_manager_thread_server.py` (§3) is only a *consumer* — the body-tracking data comes
from the **XRoboToolkit PC service** on the workstation, paired over **Wi-Fi** with the
**client app running on the headset**. Both must be up first, or §3 hangs at
`waiting for body data...`.

### 2b.1 PC service (workstation)

```bash
cd /home/grease/XRoboToolkit-PC-Service/RoboticsService/bin
nohup ./RoboticsServiceProcess > /tmp/xrobo_pc_service.log 2>&1 &
```

Verify — expect the process plus two listeners (`63901` faces the headset, `60061` is the
local SDK endpoint that `pico_manager_thread_server.py` connects to):

```bash
pgrep -af RoboticsServiceProcess
ss -lntp | grep -i robotics
```

> **Do not run it with `sudo`.** `/opt/apps/roboticsservice/runService.sh` is only a
> wrapper that `cd`s into the path above; under `sudo` it resolves the wrong `$HOME` and
> exits 1. Run the binary directly as your own user, as shown.

### 2b.2 PICO client (headset, launched over adb)

The headset app is `com.xrobotoolkit.client` (Unity). With the PICO attached over USB:

```bash
adb devices -l                       # note the serial, e.g. PA921HMGL4180126G
adb -s <SERIAL> shell am start -n \
    com.xrobotoolkit.client/com.unity3d.player.UnityPlayerActivity

# confirm it is up (prints a pid)
adb -s <SERIAL> shell pidof com.xrobotoolkit.client
```

> **Always pass `-s <SERIAL>`.** A stale/offline transport makes bare `adb shell` fail with
> `error: more than one device/emulator` even when `adb devices` lists only one headset.

Useful adb helpers:

```bash
adb -s <SERIAL> shell ip route | grep -oE "src [0-9.]+"   # headset IP
adb -s <SERIAL> logcat -s Unity:V                          # client-side logs
adb -s <SERIAL> shell am force-stop com.xrobotoolkit.client
```

### 2b.3 Pair them (inside the headset)

1. In the client, set **PC Service** to the **workstation's** IP (`hostname -I`, e.g.
   `192.168.8.192`) → *Enter*, or *Reconnect* if it is already filled in.
2. Wait for **`Status: WORKING`**.
3. Tick **Head** and **Controller** under *Tracking*.
4. Set *Data/Control* to **Send**.
5. Set *Pico Motion Tracker* to **Full body**.

Both machines must be on the **same Wi-Fi network** — USB/adb is only used to *launch*
and debug the app, the pose data itself travels over Wi-Fi.

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

### 4c. No-VR / AMASS-augmented low-latency model (`sonic_no_vr_llam_080k`)

Same invocation, different policy directory. This family (`sonic_no_vr_050k`,
`sonic_no_vr_ll_062k`, `sonic_no_vr_llam_080k`) was trained **without** the teleop/VR
encoder branch, so it has only two encoder modes (`g1`=0, `smpl`=2).

```bash
cd /home/grease/gam/gear_sonic_deploy
./target/release/g1_deploy_onnx_ref lo \
  policy/sonic_no_vr_llam_080k/model_decoder.onnx \
  reference/example/ \
  --obs-config policy/sonic_no_vr_llam_080k/observation_config.yaml \
  --encoder-file policy/sonic_no_vr_llam_080k/model_encoder.onnx \
  --input-type zmq \
  --zmq-host localhost \
  --zmq-port 5556 \
  --zmq-topic pose \
  --zmq-conflate \
  --disable-crc-check
```

| | value |
|---|---|
| Encoder / decoder dim | **979 / 994** (not 1247 — that is `low_latency`) |
| SMPL lookahead | **4 frames** ⇒ pair with `--num_frames_to_send 4` (§3) |
| Expected mode | `GetEncodeMode()=2`; mode 0 means the SMPL stream is not being consumed and the robot will stand still |

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

## 10b. Real Robot Deployment (two machines)

On the real G1 the pipeline is **split across two machines**:

| Program | Runs on | Why |
|---------|---------|-----|
| `g1_deploy_onnx_ref` (`deploy.sh ... real`) | **Robot** (onboard Jetson) | Needs DDS on the robot's `192.168.123.x` network to drive motors at 500 Hz + CUDA/TensorRT |
| `pico_manager_thread_server.py` | **Teleop PC** | Needs XRoboToolkit PC Service (`127.0.0.1:60061`) paired with the PICO headset, plus a display for visualization |

The ZMQ link is no longer loopback — the robot subscribes across the network.

**Terminal 1 — on the ROBOT:**
```bash
cd ~/gam/gear_sonic_deploy
bash deploy.sh --input-type zmq \
  --zmq-host <TELEOP_PC_IP> \
  --zmq-port 5556 \
  --zmq-topic pose \
  real
```

**Terminal 2 — on the TELEOP PC:**
```bash
cd ~/gam
.venv_teleop/bin/python -u gear_sonic/scripts/pico_manager_thread_server.py \
    --manager --auto_pose
```

> `deploy.sh real` auto-detects the `192.168.123.x` interface and does **not**
> pass `--disable-crc-check` (that flag is simulation-only).

> ⚠️ **Start conservatively on hardware.** Use `--target_fps 50 --num_frames_to_send 2`
> for the first runs: that matches the 50 Hz control rate exactly so the adaptive
> catch-up path (§11b) never activates.

---

## 11. Troubleshooting


| Symptom | Cause / Fix |
|---------|-------------|
| First launch takes minutes | TensorRT compiles ONNX → `.trt` engine (one-time). Cached next to the ONNX. Low-latency decoder is 143 MB, so its first build is slow. |
| `Binding to port: 5557` → `zmq::error_t: Address already in use`, exit 134 | Something else holds the debug PUB port. Find it with `ss -lntp \| grep 5557`. **VS Code auto-forwards 5557/5558** (process `code`, `--utility-sub-type=node.mojom.NodeService`) after it sees them in use once — in that case either *Stop Forwarding Port* in the **Ports** panel (`"remote.autoForwardPorts": false` prevents a repeat), kill that helper PID, or just move the port with `--zmq-out-port 5559`. Note 5558 collides with `perception_heightmap/view_lidar_client.py` for the same reason. |
| `Unknown observation function '<name>'` | The observation is missing from `GetObservationRegistry()` in `g1_deploy_onnx_ref.cpp`. Add the entry and rebuild (§10). |
| Observation dimension mismatch | `observation_config.yaml` doesn't match the ONNX. Verify with §7. |
| PICO stuck at `waiting for body data...` | Check in order: PC service running (§2b.1), client app running on the headset (§2b.2), `Status: WORKING` + *Send* + *Full body* ticked (§2b.3), then Motion Trackers paired/on — `--xrt_diag` (§9), check `serials`. |
| `adb: error: more than one device/emulator` (only one headset attached) | Stale transport. Always target the serial explicitly: `adb -s <SERIAL> shell ...` (§2b.2). |
| `sudo /opt/apps/roboticsservice/runService.sh` exits 1 | The wrapper `cd`s into `$HOME/XRoboToolkit-PC-Service/...`; under `sudo` that path doesn't resolve. Run `./RoboticsServiceProcess` directly as your user (§2b.1). |
| Visualization window frozen | Manager is in `OFF` mode — use `--auto_pose`, or press A+B+X+Y then A+X on the controllers. |
| Robot lags further behind over time | Producer FPS > 50 Hz consumption. Adaptive catch-up handles this; confirm `timesteps` stays flat (§9). |
| `selected interface "lo" is not multicast-capable` | Harmless warning in loopback/sim mode. |

---

## 11b. Debugging the ZMQ Link (robot ⇄ teleop PC)

The stream is **PUB (teleop PC, binds `0.0.0.0:5556`) → SUB (robot, connects out)**.
Debug it in layers — stop at the first one that fails.

### Layer 1 — Is the publisher actually publishing?

**On the TELEOP PC:**
```bash
# Is the process alive and is 5556 bound (0.0.0.0, not 127.0.0.1)?
pgrep -fl pico_manager_thread_server
ss -ltnp | grep 5556

# Is real data flowing on loopback?
.venv_teleop/bin/python gear_sonic/scripts/verify_pose_stream.py --seconds 10
```

Expect `✅ Receiving pose data at ~N msg/s` and an advancing `frame_index`.

- No data → the manager is idling in `OFF` mode. Use `--auto_pose`.
- `frame_index` not advancing → body tracking is stale; see §9 (`--xrt_diag`).

### Layer 2 — Is the network path open?

**On the ROBOT:**
```bash
ping -c 3 <TELEOP_PC_IP>
nc -vz <TELEOP_PC_IP> 5556        # must report "succeeded"/"open"
```

If ping works but the port doesn't, it's a firewall on the **publisher**:
```bash
# On the TELEOP PC
sudo ufw allow 5556/tcp
```

Confirm both machines are on the robot subnet:
```bash
ip -4 addr show | grep 192.168.123
```

### Layer 3 — Does ZMQ data cross the network?

**On the ROBOT** (independent of the deploy binary — this is the decisive test):
```bash
python gear_sonic/scripts/verify_pose_stream.py \
    --host <TELEOP_PC_IP> --seconds 15
```

| Result | Meaning |
|--------|---------|
| `✅ Receiving pose data` | Link is healthy — problem is inside the deploy binary (Layer 4) |
| `❌ NO DATA RECEIVED` | Network/firewall issue — go back to Layer 2 |
| `⚠️ messages ARRIVE but NONE parsed` | Version mismatch between publisher and subscriber code |
| Large `p95`/`max` intervals | Wi-Fi jitter — prefer wired, and see the catch-up note below |

### Layer 4 — Is the deploy binary consuming it?

In the robot's deploy terminal, look for:

```
[ZMQEndpointInterface] Received ZMQ message - topic: 'pose', protocol_version: 3
[ZMQEndpointInterface] Decode interval: ~10 ms, decode time: ~1-2 ms
[ZMQEndpointInterface] ... did_catchup=0
[ZMQEndpointInterface] result.motion->GetEncodeMode()=2
```

| Symptom | Cause |
|---------|-------|
| No `Received ZMQ message` lines at all | Wrong `--zmq-host` / `--zmq-port` / `--zmq-topic` (topic must be exactly `pose`) |
| Messages received, robot doesn't move | Policy not started — check the operator start signal and `Streaming data mean delay` |
| `did_catchup=1` repeatedly | Stream stalling badly; check Layer 3 jitter |
| `timesteps` growing without bound | Producer FPS > consumption rate (see §11b note below) |

Also watch the ~1 Hz timing line:
```
Loop timing - ... Streaming data mean delay: <N>ms, Policy: <N>us
```
`Streaming data mean delay` is your end-to-end network+pipeline latency.

### Quick one-liner triage

```bash
# ON ROBOT — is the publisher reachable and streaming?
nc -vz <TELEOP_PC_IP> 5556 && \
python gear_sonic/scripts/verify_pose_stream.py --host <TELEOP_PC_IP> --seconds 10
```

### Note on the adaptive catch-up

The playback cursor advances up to `kMaxCatchupStep` (4) frames per 50 Hz tick when
buffered lookahead exceeds `kCatchupCushion` (3) — see `CurrentFrameAdvancement()`
in `g1_deploy_onnx_ref.cpp`.

- At `--target_fps 100`, equilibrium is `step=2` = exactly 1.0× real-time (correct).
- Draining a backlog can transiently reach `step=4` = **2× real-time playback**.

On a jittery wireless link this transient fast-forward happens more often. For real
hardware, either rate-match the producer (`--target_fps 50 --num_frames_to_send 2`,
which keeps `step=1`) or lower `kMaxCatchupStep` to 3 (caps overspeed at 1.5×).

---

## 12. Quick Copy-Paste: Full Pipeline


One-time per session, then three terminals:

```bash
# Terminal 0 — XRoboToolkit stack (PC service + headset client), see §2b
cd /home/grease/XRoboToolkit-PC-Service/RoboticsService/bin && \
  nohup ./RoboticsServiceProcess > /tmp/xrobo_pc_service.log 2>&1 &
adb -s <SERIAL> shell am start -n \
  com.xrobotoolkit.client/com.unity3d.player.UnityPlayerActivity
# then in the headset: PC Service IP = `hostname -I`, Status: WORKING,
# Head+Controller ticked, Data/Control = Send, Motion Tracker = Full body
```

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
# Terminal 3 — policy (swap low_latency ↔ release ↔ sonic_no_vr_llam_080k as needed)
cd /home/grease/gam/gear_sonic_deploy && ./target/release/g1_deploy_onnx_ref lo \
  policy/low_latency/model_decoder.onnx reference/example/ \
  --obs-config policy/low_latency/observation_config.yaml \
  --encoder-file policy/low_latency/model_encoder.onnx \
  --input-type zmq --zmq-host localhost --zmq-port 5556 --zmq-topic pose \
  --zmq-conflate --disable-crc-check
```
