# G1 Robot Teleoperation Replay Guide

This document captures the end-to-end workflow for bypassing the live VR streaming (`pico_manager`) and feeding recorded human SMPL mocap data directly into the C++ G1 mapping policy securely using a custom replay server.

## Overview

The `g1_deploy_onnx_ref` C++ binary relies on heavily customized, real-time message structures (ZMQ header arrays) originally published by the Pico VR headsets. To test the robot simulation blindly with offline data, we run a custom Python script that loads raw `.npz` recordings and explicitly broadcasts them exactly mimicking the original setup.

## Workflow

### 1. Identify Target Human Motion Sequence
The streaming `.npz` teleoperation files are typically stored in:
- `logs/smpl_raw` 
- or `logs/raw_smpl`

*(Ensure you isolate the exact frames you want to test into a smaller target directory, such as `logs/yelong_cliptest_1`, so the system can seamlessly cache them.)*

### 2. Start MuJoCo Simulation Environment
The C++ policy controls the simulated robot via CycloneDDS messages targeting the `run_sim_loop.py` script. The simulator must be instantiated before the policy evaluates.
```bash
cd ~/gam
env -u CYCLONEDDS_HOME .venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py
```

### 3. Launch the ZMQ PICO Replay Node
The `pico_replay_server.py` node sits exactly where `pico_manager_thread_server.py` would conceptually sit. It loads your target sequence, creates the custom 1024-byte padded JSON structural boundaries utilizing `pack_pose_message()`, and transmits blindly at exactly ~50 FPS.
```bash
cd ~/gam
.venv_teleop/bin/python gear_sonic/scripts/pico_replay_server.py --replay_dir logs/yelong_cliptest_1
```
*(Leave this running in its own terminal or background it.)*

### 4. Deploy C++ Neural Network Core
Finally, attach the core C++ deployment loop to the configured `localhost:5556` node hosted by our script. We set `--disable-crc-check` to support simulation bounds.
```bash
cd ~/gam/gear_sonic_deploy
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
    --disable-crc-check \
    --enable-csv-logs \
    --logs-dir logs/run_replay_test
```

### Process Verification
The C++ logging will immediately trace standard continuous timing evaluations, looping at ~50 Hz to keep the control structure secure. As the `pico_replay_server.py` streams exact frame indices looping across ZMQ, the robot in `Mujoco` will automatically bind and physically map the movements!

Outputs of the internal execution mapping during this sequence are securely dropped inside:
`~/gam/gear_sonic_deploy/logs/run_replay_test/` *(For plotting kinematics verification later).*
