# 🚀 GEAR-SONIC Training Pipeline

Complete end-to-end training infrastructure for action prediction models on egocentric teleoperation data.

## 📌 Quick Links

| Document | Purpose |
|----------|---------|
| **[TRAINING_PIPELINE_SUMMARY.md](TRAINING_PIPELINE_SUMMARY.md)** | Overview of pipeline, current status, next steps |
| **[TRAINING_COMMANDS.sh](TRAINING_COMMANDS.sh)** | Copy-paste commands for common tasks |
| **[gear_sonic/training/README.md](gear_sonic/training/README.md)** | Full technical documentation |
| **[gear_sonic/training/GETTING_STARTED.md](gear_sonic/training/GETTING_STARTED.md)** | Step-by-step guide |

## ⚡ 30-Second Start

```bash
cd /home/grease/gam
source .venv_sim/bin/activate

# Check test training (currently running)
tail -f outputs/sonic_training_test.log

# Or start new training
python gear_sonic/training/train.py --config gear_sonic/training/config.yaml
```

## 📊 Status

✅ **Training pipeline deployed**
- Test run active: `outputs/sonic_training_test/`
- Full config ready: `gear_sonic/training/config.yaml`
- Dataset: 200 episodes, 496GB, 286K samples
- Models: Transformer + MLP baseline
- Features: Checkpointing, TensorBoard, GPU acceleration

## �� What's New

### Code Added (1,500+ lines)
```
gear_sonic/training/
├── data_loader.py   - Egocentric dataset loader
├── model.py         - Action prediction models
├── trainer.py       - Training loop & utilities  
├── train.py         - CLI entry point
├── config.yaml      - Default configuration
└── config_test.yaml - Quick test configuration
```

### Key Features
- ✅ Parquet data loader with normalization
- ✅ Configurable model architectures
- ✅ Full training loop with validation
- ✅ Checkpoint saving (best + periodic)
- ✅ TensorBoard integration
- ✅ GPU acceleration support
- ✅ CLI argument overrides
- ✅ Comprehensive logging

## 🚀 Next Steps

### Immediate (1 min)
```bash
# Watch test training
tail -f outputs/sonic_training_test.log
```

### Short-term (1 hour)
```bash
# Start full training when ready
python gear_sonic/training/train.py --config gear_sonic/training/config.yaml
```

### Medium-term (1-3 days)
```bash
# Monitor progress via TensorBoard
tensorboard --logdir outputs/sonic_training/logs

# Use best model for inference
torch.load("outputs/sonic_training/best_model.pt")
```

## 📚 Documentation

**For quick reference**: See [TRAINING_COMMANDS.sh](TRAINING_COMMANDS.sh)

**For detailed info**: See [TRAINING_PIPELINE_SUMMARY.md](TRAINING_PIPELINE_SUMMARY.md)

**For architecture details**: See [gear_sonic/training/README.md](gear_sonic/training/README.md)

## 💡 Key Commands

```bash
# Test training (50 ep, MLP, fast)
python gear_sonic/training/train.py --config gear_sonic/training/config_test.yaml

# Full training (all data, transformer)
python gear_sonic/training/train.py --config gear_sonic/training/config.yaml

# With custom hyperparameters
python gear_sonic/training/train.py --config gear_sonic/training/config.yaml \
  --batch-size 64 --learning-rate 5e-4 --num-epochs 300

# Monitor with TensorBoard
tensorboard --logdir outputs/sonic_training/logs

# Check logs
tail -f outputs/sonic_training/sonic_training.log
```

## 📊 Data Overview

| Property | Value |
|----------|-------|
| **Location** | `/home/grease/ego_dataset/work_bearlu/data/` |
| **Format** | Parquet (egocentric teleoperation) |
| **Episodes** | 200 |
| **Total size** | 496GB |
| **Trajectory samples** | 286,400 |
| **Obs dimension** | 57 (joints, grippers, EE pose) |
| **Action dimension** | 35 (joint commands) |

## 🏗️ Architecture

**Models**:
- `SonicActionPredictor`: Transformer (temporal modeling)
- `SonicMLP`: Feedforward baseline (fast)

**Loss**: MSE between predicted and target actions

**Training**: SGD with cosine annealing LR, gradient clipping

**Validation**: Every 5 epochs, saves best checkpoint

## 📈 Expected Timeline

| Epochs | Time | Loss Trend |
|--------|------|-----------|
| 0-10 | 2.5h | Rapid drop |
| 10-50 | 10h | Steady improvement |
| 50-100 | 12.5h | Continued improvement |
| 100-200 | 25h | Convergence |

Total: ~50 hours for 200 epochs on full dataset

## 🔧 Configuration

See [gear_sonic/training/config.yaml](gear_sonic/training/config.yaml) for all options:

**Key parameters**:
- `context_length`: 4 (observation history)
- `action_horizon`: 8 (actions to predict)
- `batch_size`: 32 (memory tradeoff)
- `learning_rate`: 1e-3 (start here)
- `num_epochs`: 100 (or more for better results)

## ✅ Validation

```bash
# Check training started
grep "Found" outputs/sonic_training_test.log

# Check training progress
grep "Epoch" outputs/sonic_training_test.log | tail -5

# Check validation results
grep "Validation Metrics" outputs/sonic_training_test.log

# Check best model found
grep "Saved best model" outputs/sonic_training_test.log
```

## 🎓 Using Trained Models

```python
import torch
from gear_sonic.training import SonicMLP

# Load
model = SonicMLP(obs_dim=57, action_dim=35)
ckpt = torch.load("outputs/sonic_training/best_model.pt")
model.load_state_dict(ckpt["model_state"])
model.eval()

# Predict
with torch.no_grad():
    obs = torch.randn(1, 4, 57)  # (batch, context, obs_dim)
    actions = model(obs)          # (batch, horizon, action_dim)
```

## 📁 Output Structure

```
outputs/
├── sonic_training_test/      (Test run)
│   ├── best_model.pt         (Best checkpoint)
│   ├── checkpoint_epoch_*.pt  (Periodic saves)
│   ├── config.json           (Config used)
│   └── logs/                 (TensorBoard)
└── sonic_training/           (Full training)
    ├── best_model.pt
    ├── checkpoint_epoch_*.pt
    ├── config.json
    └── logs/
```

## 🐛 Troubleshooting

**Training slow?** Reduce `context_length` or `batch_size`

**Out of memory?** Use MLP instead of transformer, reduce `hidden_dim`

**Loss not decreasing?** Try lower `learning_rate` (5e-4 or 1e-4)

See [TRAINING_PIPELINE_SUMMARY.md#-troubleshooting](TRAINING_PIPELINE_SUMMARY.md#-troubleshooting) for more

## 📖 References

- Model architectures: `gear_sonic/training/model.py`
- Training loop: `gear_sonic/training/trainer.py`  
- Data loading: `gear_sonic/training/data_loader.py`
- Full docs: `gear_sonic/training/README.md`

## 🎯 Success Criteria

✅ Training pipeline deployed
✅ Test run active (currently running)
✅ Dataset integrated (200 episodes loaded)
✅ Models ready (Transformer + MLP)
✅ Logging configured (TensorBoard)
✅ Documentation complete

---

## 📐 SONIC Full Training Flow (from paper / design doc)

This section documents the intended SONIC encoder–decoder training pipeline.
The current code in `gear_sonic/training/` implements an earlier, simpler
baseline.  See **Gap Analysis** below for a precise comparison.

### Step 1 — Data Preparation

1. Collect human motion videos and extract SMPL motion sequences.  
   Each frame is represented as **24 × 3** joint positions.

2. Build synchronized pairs of human and robot motion:
   - **g_h** — original SMPL 24-joint 3D positions (human representation).
   - **g_r** — retargeted G1 robot joint representation, converted offline using
     **GMR** (Araujo et al., 2025) and **PyRoki**.
   - **g_m** — mixed representation: upper-body 3-point VR/PICO tracking
     (head + two wrists) combined with lower-body robot joint state.

   All three representations are prepared offline before training begins.
   The correspondence does not need to be learned during training.

### Step 2 — Per-Episode Forward Pass (each training step)

1. **Sample** a motion segment from the motion-capture dataset.

2. **Encode** each representation separately into a latent token:

   | Encoder | Input | Token |
   |---------|-------|-------|
   | E_r (robot encoder) | g_r | z_r |
   | E_h (human encoder) | g_h | z_h |
   | E_m (mixed encoder) | g_m | z_m |

3. **Decode** (for reconstruction / policy use):
   - Motion decoder **D_r** takes any token and reconstructs **g_r**.
   - RL policy decoder takes the token (z_r, z_h, or z_m) plus robot
     proprioception history and outputs a 29-DoF motor action.

4. **Execute** the motor action in the physics simulation (MuJoCo / Isaac).

### Step 3 — Loss Computation (all losses computed simultaneously)

Four losses are computed and back-propagated together:

```
L_total = L_PPO + λ_recon * L_recon + λ_token * L_token + λ_cycle * L_cycle
```

#### L_PPO — Reinforcement Learning Loss
The standard PPO surrogate objective. Reward signal comes from executing the
decoded motor actions in the physics simulation (motion tracking reward:
joint-position tracking, root-pose tracking, balance, smoothness, etc.).

#### L_recon — Reconstruction Loss
All three tokens must be able to reconstruct **g_r** via the shared decoder D_r:

```
L_recon = ||D_r(z_r) - g_r||² + ||D_r(z_h) - g_r||² + ||D_r(z_m) - g_r||²
```

This acts as **implicit retargeting**: when `||D_r(z_h) - g_r||²` is
back-propagated, gradients flow through both D_r and E_h, forcing the human
encoder to learn the human-to-robot mapping without an explicit retargeting
supervision signal.

#### L_token — Token Alignment Loss
Direct cross-modal supervision that pulls human and robot tokens together in
latent space:

```
L_token = ||z_r - z_h||²
```

Unlike VAE-style approaches that align representations only indirectly through
reconstruction, this loss provides direct supervision in the token space.

#### L_cycle — Cycle-Consistency Loss
Ensures that encoding a z_h-decoded robot motion with the robot encoder
recovers the original robot token:

```
L_cycle = ||E_r(D_r(z_h)) - z_r||²
```

#### Back-propagation
A single `loss.backward()` updates **all** network parameters (E_r, E_h, E_m,
D_r, policy decoder) simultaneously.

### Step 4 — Deployment

At inference / deployment time only **one** encoder is active, selected by
`encode_mode` in `observation_config.yaml`:

| Mode | Encoder used | Input source |
|------|-------------|--------------|
| 0 (`g1`) | E_r | Retargeted robot reference motion |
| 1 (`teleop`) | E_m | Live VR 3-point + lower-body robot state |
| 2 (`smpl`) | E_h | Live SMPL pose stream (PICO) |

The encoder output `z` becomes `token_state` in the observation vector fed to
the decoder policy (`model_decoder.onnx`).

---

## 🔍 Gap Analysis: Current Code vs. SONIC Training Flow

| Aspect | Current `gear_sonic/training/` | SONIC Full Flow (paper) |
|--------|-------------------------------|-------------------------|
| **Task** | Supervised action prediction on egocentric parquet data | PPO motion-tracking in physics simulation |
| **Data representation** | Single flat observation vector (57-dim: joints + EE) | Three synchronized representations: g_r, g_h, g_m |
| **Encoders** | None — raw obs fed directly to model | Three separate encoders: E_r, E_h, E_m |
| **Latent token** | Not present | z_r, z_h, z_m; used as policy condition |
| **Model** | `SonicActionPredictor` (Transformer) or `SonicMLP` | Encoder + Decoder policy (ONNX: `model_encoder.onnx` + `model_decoder.onnx`) |
| **Loss** | MSE only: `loss = MSE(predicted_action, target_action)` | L_PPO + L_recon + L_token + L_cycle |
| **RL** | ❌ No physics simulation or RL | ✅ PPO with MuJoCo / Isaac simulation |
| **Reconstruction** | ❌ No reconstruction objective | ✅ D_r(z) must reconstruct g_r |
| **Cross-modal alignment** | ❌ Single modality only | ✅ L_token = `||z_r - z_h||²` |
| **Cycle consistency** | ❌ Not implemented | ✅ `||E_r(D_r(z_h)) - z_r||²` |
| **Retargeting** | ❌ Not handled; assumes pre-processed parquet | ✅ Implicit via L_recon gradient flow through E_h |
| **Deployment target** | Not directly deployable to real robot | `model_encoder.onnx` + `model_decoder.onnx` → `g1_deploy_onnx_ref` |

### What needs to be added to reach the full SONIC training flow

1. **Three-encoder architecture** — Implement E_r, E_h, E_m that each compress a
   different motion representation into a fixed-size token z.

2. **Shared motion decoder D_r** — A decoder that reconstructs robot joint
   trajectory from any token, enabling the reconstruction and cycle losses.

3. **PPO training loop with physics simulation** — Replace the supervised
   `SonicTrainer` MSE loop with a PPO loop that rolls out the decoder policy
   in MuJoCo/Isaac, collects rewards, and computes advantages.

4. **Multi-representation data pipeline** — Extend `data_loader.py` to load and
   serve g_r, g_h, g_m simultaneously per episode instead of a single flat
   observation vector.

5. **Combined loss** — Replace `self.criterion = nn.MSELoss()` in `trainer.py`
   with a combined loss function:
   ```python
   loss = L_ppo + lambda_recon * L_recon + lambda_token * L_token + lambda_cycle * L_cycle
   ```

6. **Encode-mode routing at inference** — The trained encoder must be exported
   as `model_encoder.onnx` with the correct `encode_mode` field so the
   `g1_deploy_onnx_ref` binary can select the right modality at runtime.

---

## 🔨 Implementation Roadmap: Full SONIC Training Pipeline

Concrete checklist to evolve `gear_sonic/training/` from the current supervised
baseline into the full SONIC encoder–decoder PPO training system.

---

### Item 1 — Three-encoder architecture

- [ ] Create `gear_sonic/training/encoders.py` with three encoder classes:
  - `RobotEncoder(E_r)` — input: `g_r` (retargeted 29-DoF joint trajectory window); output: `z_r` (token, e.g. 64-dim)
  - `HumanEncoder(E_h)` — input: `g_h` (SMPL 24 × 3 joint positions window); output: `z_h`
  - `MixedEncoder(E_m)` — input: `g_m` (upper-body 3-point pose + lower-body robot joints); output: `z_m`
- [ ] Each encoder is a small Transformer or MLP that maps a short temporal window
  (e.g. 10 frames × feature_dim) → fixed-size token vector.
- [ ] All three share the same output dimension so the decoder is agnostic to which
  encoder was used at deployment time.
- [ ] Register all three in `gear_sonic/training/model.py` alongside the existing
  `SonicActionPredictor`.

---

### Item 2 — Shared motion decoder D_r

- [ ] Add `MotionDecoder(D_r)` in `gear_sonic/training/encoders.py` (or a new
  `gear_sonic/training/decoders.py`).
- [ ] Input: token `z` (64-dim); output: reconstructed `g_r` sequence
  (T × 29 joint positions, e.g. T = 10).
- [ ] This decoder is **shared** — it must be called with z_r, z_h, and z_m
  separately during training to compute `L_recon`.
- [ ] Distinguish from the **RL policy decoder** (`SonicActionPredictor`) which takes
  `z + robot_proprioception_history → motor_action`. They can share weights or be
  separate depending on ablation results.

---

### Item 3 — PPO training loop with physics simulation

- [ ] Create `gear_sonic/training/ppo_trainer.py` replacing the supervised
  `SonicTrainer` in `trainer.py`.
- [ ] The PPO loop must:
  1. Roll out the policy in MuJoCo (via `gear_sonic/utils/mujoco_sim/`) or Isaac.
  2. Collect `(state, action, reward, next_state, done)` tuples.
  3. Compute GAE advantages and PPO clipped surrogate loss (`L_PPO`).
  4. Back-propagate combined loss (see Item 5) through encoders, decoders, and policy.
- [ ] Reward function components to implement in `gear_sonic/training/rewards.py`:
  - `r_joint_tracking` — `exp(-||q_robot - q_ref||² / σ_q)`
  - `r_root_pose` — root height, orientation tracking
  - `r_balance` — penalise large base angular velocity
  - `r_smooth` — penalise action rate `||a_t - a_{t-1}||²`
  - `r_torque` — penalise large motor torque
- [ ] Use existing simulation bridge: `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py`
  and `gear_sonic/utils/mujoco_sim/base_sim.py`.

---

### Item 4 — Multi-representation data pipeline

- [ ] Extend `gear_sonic/training/data_loader.py`:
  - Current: loads a single flat 57-dim observation vector from Parquet.
  - Required: load and return a dict `{"g_r": ..., "g_h": ..., "g_m": ...}` per step.
- [ ] Add a new dataset class `SonicMotionDataset` that reads the three CSV arrays
  from the `reference/` folder structure (same format as `squat_001__A359`):
  - `joint_pos.csv` / `joint_vel.csv` → g_r
  - `smpl_joint.csv` → g_h
  - Mixed: combine `body_pos.csv` upper-body rows + robot `joint_pos.csv` lower-body → g_m
- [ ] Ensure the data loader yields aligned triplets `(g_r[t:t+W], g_h[t:t+W], g_m[t:t+W])`
  for a window of W frames per training step.
- [ ] Keep the existing `EgocentricDataset` intact for the supervised baseline.

---

### Item 5 — Combined loss function

- [ ] Create `gear_sonic/training/losses.py` with:

  ```python
  def sonic_combined_loss(z_r, z_h, z_m, g_r,
                          D_r, E_r,
                          ppo_loss,
                          lambda_recon=1.0,
                          lambda_token=0.5,
                          lambda_cycle=0.1):
      # L_recon: all tokens must reconstruct g_r
      L_recon = (mse(D_r(z_r), g_r)
               + mse(D_r(z_h), g_r)
               + mse(D_r(z_m), g_r))

      # L_token: pull robot and human tokens together
      L_token = mse(z_r, z_h)

      # L_cycle: E_r(D_r(z_h)) must recover z_r
      L_cycle = mse(E_r(D_r(z_h)), z_r.detach())

      return ppo_loss + lambda_recon * L_recon + lambda_token * L_token + lambda_cycle * L_cycle
  ```

- [ ] Replace `self.criterion = nn.MSELoss()` in `trainer.py` with a call to
  `sonic_combined_loss` once the PPO loop (Item 3) is in place.
- [ ] Add λ hyper-parameters to `gear_sonic/training/config.yaml`:
  ```yaml
  lambda_recon: 1.0
  lambda_token: 0.5
  lambda_cycle: 0.1
  ```

---

### Item 6 — Encode-mode routing and ONNX export

- [ ] Add `gear_sonic/training/export.py` with an `export_encoder` function that:
  1. Loads the trained `RobotEncoder`, `HumanEncoder`, or `MixedEncoder`.
  2. Wraps it in a single ONNX-compatible module that accepts an `encode_mode`
     integer input (0 = g1/robot, 1 = teleop/mixed, 2 = smpl/human).
  3. Exports to `policy/release/model_encoder.onnx` matching the shape expected
     by `g1_deploy_onnx_ref` (`observation_config.yaml`, `encoder.dimension`).
- [ ] Add `export_decoder` that exports the RL policy decoder to
  `policy/release/model_decoder.onnx`.
- [ ] Validate exported models against `policy/release/observation_config.yaml`
  encoder dimension (currently 64) before deployment.
- [ ] Test round-trip: Python inference → ONNX export → TensorRT compile on Unitree
  Orin → deploy with `./target/release/g1_deploy_onnx_ref`.

---

### Summary Checklist

| # | File(s) to create / modify | Status |
|---|---------------------------|--------|
| 1 | `gear_sonic/training/encoders.py` — E_r, E_h, E_m | ⬜ TODO |
| 2 | `gear_sonic/training/encoders.py` or `decoders.py` — D_r | ⬜ TODO |
| 3 | `gear_sonic/training/ppo_trainer.py`, `rewards.py` | ⬜ TODO |
| 4 | `gear_sonic/training/data_loader.py` — `SonicMotionDataset` | ⬜ TODO |
| 5 | `gear_sonic/training/losses.py`, update `config.yaml` | ⬜ TODO |
| 6 | `gear_sonic/training/export.py` — ONNX encoder/decoder export | ⬜ TODO |

---

**Status**: Ready for training ✅  
**Data**: 496GB egocentric teleoperation  
**Compute**: GPU accelerated (CUDA available)  
**Timeline**: 50 hours for 200 epoch run  

See [TRAINING_PIPELINE_SUMMARY.md](TRAINING_PIPELINE_SUMMARY.md) for full details.
