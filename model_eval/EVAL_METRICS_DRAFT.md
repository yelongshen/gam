# Comprehensive Evaluation Metrics for Humanoid Motion Tracking

Based on the analysis of SONIC, Humanoid-GPT, and state-of-the-art teleoperation/motion reproduction methods (e.g., OmniH2O, H2O, ReL2), an effective systemic evaluation framework splits into **Simulation Metrics** (large-scale policy validation) and **Real-World Metrics** (hardware tracking fidelity).


## 1. Simulation Evaluation (Policy & Tracking Quality)
In simulation, physics are rigid and exact, making it the ideal sandbox for calculating precise geometric error tracking and overall reinforcement learning robustness. To accurately benchmark against prior works like Humanoid-GPT and OmniH2O, we must evaluate the policy across a standardized, categorical dataset rather than a single trajectory.

### A. Proposed Evaluation Set (The Motion Dataset splits)
A robust evaluation requires testing both **In-Distribution (ID)** (motions similar to training) and **Out-of-Distribution (OOD)** (novel or extreme motions). *Note: The dataset previously processed in `smpl_filtered` (e.g. standard AMASS / CMU) was utilized extensively during Model Training. To prevent data contamination during Evaluation, we strictly extract testing clips organically from independent sub-datasets (e.g. ACCAD) that the model has formally never trained against.*

According to the **SONIC paper methodology**, tracking must be validated across designated splits rather than arbitrary categories. The paper explicitly specifies evaluation testing splits:
1.  **Test-Content (OOD / Out-of-Distribution):** Tests generalization to *novel motion content*, consisting of sub-categories entirely absent from training (e.g., hip-hop dance, stage bow, sword lunge, roundhouse kick).
2.  **Test-Repetition (ID / In-Distribution):** Tests robustness to *new performances/repetitions* of known motion types (e.g., different takes and actor performances of motions structurally similar to training data).
3.  **PHUMA Benchmark:** An external standardized benchmark referenced by SONIC for baseline tracking metrics against GMT, Any2Track, and BeyondMimic.

For our local integration tests derived natively from AMASS (e.g., sampling from the downloaded `ACCAD` dataset in `~/egodata/downloads/amass/extracted/ACCAD/`), we mapped our test clips to match the **Test-Content** categories directly referenced in the SONIC paper's success tracking supplements (Fig S1):
*   **Agility & Acrobatics:** Cartwheels (`D6- CartWheel`), Kicks (`Male2MartialArtsKicks`). Represents the *roundhouse kick / sword lunge* OOD task.
*   **Complex Gestures / Dance:** Urban/Conversational Gestures (`Female1Gestures`), Swaying / Swinging (`A2`, `A3`). Represents the *hip-hop dance / stage bow* OOD task.
*   **Unstructured Motion:** Crawling backwards/forwards (`A11`, `A12`). Directly tests the theoretical failure boundaries specifically noted in SONIC (e.g., *zombie crawl / cross-legged sit failures*).
*   **Basic Locomotion (Baseline Repetition):** Walking, running, and crouching variations across actors for the *Test-Repetition* benchmark equivalent.

*(Note on Hand Teleoperation: The core AMASS motion sequences target the 21 main human body joints. Because exact articulation of fingers drops heavily during automated MoCap solving, our tracking framework (`pico_manager`) handles hand manipulation specifically using discrete physical triggers directly piped to a 7-DOF hand solver (`G1GripperInverseKinematicsSolver`). For simulation evaluation of upper-body arm movement, we focus on End-Effector wrist placement rather than precise finger limits).*

### B. Core Metrics (Simulation Sandbox)

#### 1. Tracking Accuracy (Is it doing what the human does?)
*   **Pose Error / Joint Angle MAE ($\Delta Q$):** 
    *   *Calculation:* Mean Absolute Error between the target retargeted joint angles and the robot's actual simulated joint angles (in radians).
*   **End-Effector Spatial Error (EE-MPJPE):** 
    *   *Calculation:* The L2 (Euclidean) distance in 3D Cartesian space between the target wrist/ankle positions and the simulated wrist/ankle positions (measured in cm). 
    *   *Significance:* Crucial for upper-body tasks where joint-angle errors compound into large hand positioning misses.
*   **Root Velocity Tracking Error ($\Delta V_{root}$):**
    *   *Calculation:* The difference between the target human global velocity and the robot's simulated pelvic velocity.
    *   *Significance:* Ensures the robot is actually traveling through the world, not just swinging its legs in place (foot-sliding).

#### 2. Physical Viability & Robustness (Is it surviving the physics engine?)
*   **Task Success Rate / Non-Fall Rate (%):**
    *   *Calculation:* The percentage of episodes where the robot completes the motion task without entering a failure state (defined by the robot's base/pelvis dropping below a minimum height threshold or pitch/roll exceeding stable bounds).
*   **Action Smoothness / Normalized Jerk:**
    *   *Calculation:* The variance of the action output's third derivative $\dddot{q}$. 
    *   *Significance:* Directly penalized in Humanoid-GPT and OmniH2O. Lower jerk proves the policy learned biological smoothness, avoiding snapping behaviors that would destroy physical hardware.
*   **Mean Torque / Energy Efficiency:**
    *   *Calculation:* Average absolute joint torque ($N\cdot m$) exerted across the trajectory. Good policies find the "path of least resistance" in the physics engine.

---

## 2. Real-World Evaluation (Hardware Deployment)
Simulation metrics evaluate the reinforcement learning reward. Real-world metrics evaluate the policy's real-time Sim2Real transfer capability overcoming gravity, friction, and network latency. Because ground-truth spatial data (like precise AMASS sequences) isn't inherently available on the live physical robot out-of-the-box without motion capture cameras, evaluating Real-World deployment typically focuses on internal sensor consistency and control latency derived from on-board logging (e.g., extracting from `action.csv` and `q.csv`).

### A. Main Metric (Hardware Teleoperation Fidelity)
*   **Hardware Tracking Error / Pose Deviation ($\Delta Q_{hw}$):**
    *   *Calculation:* Mean Absolute Error (MAE) between the Policy's generated Target Pose inside the neural net's output ($Q_{des}$ / actions) and the actual hardware read-state returned by the joint encoders ($Q_{curr}$) across a trajectory.
    *   *Significance:* Confirms that the PID gains and policy actions aren't fighting physical limits. If the policy outputs a joint goal that the physical motor cannot reach due to gravity or friction, this gap widens drastically. This is identical to the diagnostic analysis derived historically from checking `action.csv` against `q.csv`.
*   **Dynamic Response Latency (Glass-to-Motor equivalent):**
    *   *Calculation:* The measured pipeline time (in milliseconds) from when the human moves (e.g., VR headset timestamps the SMPL frame) to when the physical G1 motor encoder registers $>90\%$ of the target rotation trajectory.
    *   *Significance:* For egocentric teleoperation, high latency destroys immersion. Target boundaries for responsive human remote control are typically strictly $<50ms$.

### B. Systemic Hardware Health (Safety Metrics)
*   **Peak Motor Torque Analysis ($Max(\tau)$):**
    *   *Evaluation:* Plotting maximum effort limits via the physical `motor_torque.csv`. For example, if a hip roll joint repeatedly hits critical Nm thresholds ($>100Nm$), the policy highlights a heavy Sim2Real gap failing to account for true physical mass, generating aggressive target states that trigger self-collisions or over-exertion.
*   **Action Smoothness Error (Jerk Rate):**
    *   *Calculation:* Measuring the rapid spikes in target joint velocities ($dq$) generated by the neural network before smoothing.
    *   *Significance:* Directly penalized in Humanoid-GPT. A high jerk magnitude means the network is jittering in response to noisy VR tracking, resulting in dangerous physical "snapping" behaviors that can break mechanical transmissions over prolonged deployments.
*   **Thermal Accumulation Curve:**
    *   *Evaluation:* Mean and Peak changes inside `motor_temperature.csv`. Extremely rigid or unstable policies waste huge blocks of current fighting balance dynamics, rapidly climbing above 50°C and threatening thermal shutdown blocks.

---

## 3. Systemic & Architectural Evaluations (The "Why it works")
If comparing SONIC vs. Humanoid-GPT or OmniH2O:
*   **Zero-Shot Modality Transfer:**
    *   *Test:* Train using $g_r$ (Robot MoCap only), deploy using $g_h$ (Live Headset SMPL). Evaluate the Drop in Success Rate %. SONIC's architecture is explicitly designed to minimize this gap using unified cross-modal representation tokens ($z$).
*   **Context Horizon Window Robustness:**
    *   *Test:* Compare latency/accuracy tracking at a 4-frame window vs 1-frame vs 10-frames. Humanoid-GPT utilizes large temporal context windows for autoregressive sequence stability, whereas SONIC utilizes low-latency 4-frame buffer queues specifically to sacrifice historical smoothing in exchange for sub-$50ms$ predictive reaction speeds.
*   **Latent Space Clustering (Tokenization Variance):**
    *   *Test:* Visualizing a t-SNE reduction of the tokens encoded when acting across basic locomotion vs. upper-body reach tasks. Ensures the separate modality branches correctly folded multi-format vectors (like 72-DoF AMASS arrays versus Pico VR constraints) into identical operational latent distributions before hitting the decoder.

### C. Recommended Test Set (Filtered from Physical & AMASS formats)
To execute the tests properly, we pull direct simulated targets referencing the `gear_sonic_deploy/reference/example` format datasets in our system, ensuring they align directly to the C++ tracking headers.
*   **Basic Locomotion / Baseline Balance:** `squat_001`, `tired_forward_lunge_R_001`
*   **Agility / High-Dynamic:** `tired_one_leg_jumping_R_001`, `neutral_kick_R_001`
*   **Upper-Body Manipulation:** `dance_in_da_party_001`, `macarena_001` 
*   **Out-of-Distribution (OOD):** Introduce artificially delayed ZMQ streams generated by the Python proxy to test stability over poor inference loops. *(Note: Our specific real robot test `g1_deploy_run002` highlighted physical latency snaps on high-velocity requests; we can purposefully degrade the ZMQ proxy FPS during simulation tests as our main OOD latency injection test!)*

### D. AMASS Dataset Processing Pipeline (Mode 2 Neural Tracker)
*Note: The dataset previously processed in `smpl_filtered` (e.g. standard AMASS / CMU) was utilized extensively during Model Training. To prevent data contamination during Evaluation, we strictly extract testing clips organically from independent sub-datasets (e.g. ACCAD) that the model has formally never trained against.*

The downloaded open-source human motion datasets (AMASS `ACCAD.tar.bz2`) exist natively as raw human `.npz` parameter layouts. Rather than running a slow mathematical Inverse Kinematics (IK) solver to retarget these joints manually into 29-DOF arrays (which is explicitly what PHUMA relies on internally for Mode 0), our policy is specifically designed to interpret raw human geometry directly through structural latent tokens using **Mode 2**.

To inject these raw external files into the `g1_deploy_onnx_ref` C++ simulation validation architecture properly, you bypass physical formatting via ZMQ stream proxying:
1.  **Decompression & AMASS Mapping:** Expanding the AMASS `.npz` parameter payloads (which contain 156-dimensional parameterized bones). 
2.  **Network Payload Restructuring:** We isolate the 72-dimension Axis-Angle poses (`poses[:, :72]`), and restructure them dynamically in Python into a fake 4-frame lookahead queue shaped `smpl_joints [4, 24, 3]`.
3.  **ZMQ Injection:** Pack this array matrix into the strict C++ `pack_pose_message` JSON header structure and publish it at 50Hz via local UDP/TCP on Port 5556.
4.  **C++ Encoder Modality 2 Evaluation:** The `g1_deploy_onnx_ref` receives this pure AMASS payload over the socket. Since it natively operates in Mode 2 (`name: "smpl", mode_id: 2`), it pipes the data smoothly into `model_encoder.onnx`, turning the 72-dim AMASS human into a 32-dim unified latent representation without resolving IK boundaries natively.

*(This perfectly recreates what the live Pico headset executes physically during teleoperation!).*


### E. Note on Reference Action Generation (Motion Targets)
The network relies strictly on its internal token-encoding infrastructure. **The `pico_manager_thread_server.py` does not perform Whole-Body Inverse Kinematics (IK) retargeting natively.**

Instead, the workflow operates conceptually differently from traditional IK architectures:
1.  The `pico_manager` packages and transmits pure, unconverted human SMPL joints/geometries directly into the ZMQ stream.
2.  The target deployment execution (e.g. `g1_deploy_onnx_ref`) loads the designated **encoder policy** (`model_encoder.onnx`), which accepts these pure human representations as input.
3.  The model mathematically embeds the human SMPL geometries directly into a continuous, 32-dimensional (or 64-dimensional) discrete latent representation token sequence without relying on manual rigid-body matrix inversions.
4.  The action-decoder policy dynamically maps this compressed, latent intent token into output space control metrics tailored globally for the G1 robot in real-time.

**This is why PHUMA acts as our primary evaluation backend:** 
Evaluating standard humanoid RL setups usually requires explicitly retargeted sequences to grade against. PHUMA precisely delivers idealized tracking limits natively mapping to G1 robotic targets, allowing us to compute precise MPJPE against optimal baselines over 29-DOF rather than measuring human-scale discrepancies inherently distorted by height/reach mappings internally inside SMPL parameters!

---

## 4. Applied Example: 77-Motion LAFAN1 Sweep, Checkpoint Comparison (2026-08-24)

As a concrete application of the "Task Success Rate / Non-Fall Rate" and category-split framework
described in Sections 1-3 above, we ran a full sweep of all 77 LAFAN1 motion clips
(`/tmp/lafan1_all/motion_lib_individual/g1_csv`) through `eval_agent_trl.py` in a single parallel
batch (`num_envs=77`), comparing 3 checkpoints: `sonic_release/last.pt`, a low-latency variant,
and `sonic_pretrained/model_step_150000.pt`.

**Caveat:** the `ee_body_pos`/`foot_pos_xyz`/`anchor_pos` termination thresholds were relaxed to
`10000.0` to work around a known RSI floor-penetration data bug (see
`LAFAN1_eval_investigation.md`), so these are diagnostic/relaxed-bar numbers, not the standard
production eval metric.

| Checkpoint | success_rate | progress_rate |
|---|---|---|
| `last.pt` | **11.5%** | **35.3%** |
| low_latency | 5.1% | 23.9% |
| pretrained (150k) | 5.1% | 23.7% |

Category split (using LAFAN1's native action prefixes, analogous to this doc's Section 1.A
Basic Locomotion / Agility / Gestures / Unstructured taxonomy but mapped to LAFAN1's own naming
convention — walk/run/sprint, jumps/fight/dance, push*/fallAndGetUp/ground,
aiming/multipleActions/obstacles respectively) showed **Basic Motion is the only category with
non-trivial success** (21-32% depending on checkpoint) across all 3 models — every other category
(Agile, Interaction/Perturbation, Misc/Object-Obstacle) sits at or near 0% success rate for 2 of
the 3 checkpoints, and only marginally recovers for `last.pt` (6-8%).

This matches the expected pattern from Section 1.A's ID/OOD framing: **Basic Locomotion acts as
the in-distribution baseline and is comparatively easy**, while the other 3 categories behave like
out-of-distribution "harder" splits — here compounded by the still-unresolved LAFAN1 leg-scale/
floor-penetration data bug, which likely suppresses all 4 categories' true achievable success
rate until fixed. See `checkpoint_comparison.md` Section 11 and `LAFAN1_eval_investigation.md`
Section 7 for full per-category tables and root-cause discussion.
