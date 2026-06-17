# 🔍 SONIC Training Flow: Current vs. Full Implementation

## Executive Summary

The current `gear_sonic/training/` implementation is a **supervised action prediction baseline** designed for quick iteration on egocentric teleoperation data. The full SONIC training flow from the paper/design doc is a **multi-encoder PPO system** with cross-modal alignment and physics-based reward learning. This document details the gap and provides a roadmap to bridge it.

---

## 📊 Detailed Comparison Table

| Dimension | Current Implementation | Full SONIC Flow |
|-----------|----------------------|-----------------|
| **Purpose** | Supervised baseline for quick iteration | Production encoder-decoder PPO system |
| **Task Type** | Behavior cloning (action prediction) | Reinforcement learning (motion tracking) |
| **Data Input** | Single egocentric parquet file | Three synchronized representations per motion |
| **Data Representations** | Flat 57-dim vector (joints+EE) | g_r, g_h, g_m (robot, human, mixed) |
| **Architecture** | Single model (Transformer or MLP) | Three encoders + Two decoders + Policy |
| **Model Count** | 1 | 5 (E_r, E_h, E_m, D_r, Policy) |
| **Latent Space** | None | Token z ∈ ℝ^64 (or other dim) |
| **Training Objective** | MSE(predicted, target) | L_PPO + λ_recon·L_recon + λ_token·L_token + λ_cycle·L_cycle |
| **Physics Loop** | ❌ No simulation | ✅ MuJoCo / Isaac Sim |
| **Reward Signal** | ❌ Ground truth actions | ✅ Motion tracking (joint pos, root pose, balance, smoothness) |
| **RL Algorithm** | ❌ None (supervised) | ✅ PPO with GAE advantages |
| **Cross-modal Learning** | ❌ Single modality | ✅ Three modalities learned jointly |
| **Human-Robot Mapping** | Pre-computed (Parquet) | Learned implicitly via L_recon |
| **Inference** | Direct: obs → action | Encoder: obs → z; Decoder: (z + proprioception) → action |
| **Deployment Format** | Python checkpoint (.pt) | ONNX (model_encoder.onnx + model_decoder.onnx) |
| **Encode Mode** | Fixed | Selectable (g1, teleop, smpl) |

---

## 🎯 Conceptual Flow Comparison

### Current Implementation

```
Egocentric Teleoperation Data (Parquet)
    ↓
EgocentricDataset (load 57-dim obs vectors)
    ↓
DataLoader (batch, shuffle, split 80/20)
    ↓
SonicActionPredictor or SonicMLP
    ↓
    Forward:  obs_context[4] → action_pred[8]
    Loss:     MSE(action_pred, action_target)
    Backward: optimize weights
    ↓
Checkpoint save (best_model.pt)
    ↓
Inference: Load .pt → predict actions
```

**Key trait**: End-to-end supervised prediction. Input and output are both raw motion sequences. No intermediate representation.

---

### Full SONIC Flow

```
Multi-Modal Motion Dataset (CSV files: g_r, g_h, g_m)
    ↓
SonicMotionDataset (load g_r, g_h, g_m as synchronized triplets)
    ↓
DataLoader (batch, shuffle, split 80/20)
    ↓
┌─────────────────────────────────────────────────────────┐
│  Forward Pass (each training step)                      │
├─────────────────────────────────────────────────────────┤
│  1. Encode: E_r(g_r) → z_r                             │
│  2. Encode: E_h(g_h) → z_h                             │
│  3. Encode: E_m(g_m) → z_m                             │
│  4. Decode (recon): D_r(z_r), D_r(z_h), D_r(z_m) → ĝ_r│
│  5. Policy: Decoder(z + proprioception) → action       │
│  6. Simulate: MuJoCo rollout with action               │
│  7. Reward: Track motion (joint pos, root, balance)    │
└─────────────────────────────────────────────────────────┘
    ↓
Loss Computation (4 objectives)
    ├─ L_PPO:   standard PPO surrogate
    ├─ L_recon: ||D_r(z_r) - g_r||² + ||D_r(z_h) - g_r||² + ||D_r(z_m) - g_r||²
    ├─ L_token: ||z_r - z_h||²
    └─ L_cycle: ||E_r(D_r(z_h)) - z_r||²
    ↓
    L_total = L_PPO + λ_recon·L_recon + λ_token·L_token + λ_cycle·L_cycle
    ↓
Backward (updates all: E_r, E_h, E_m, D_r, Policy)
    ↓
Checkpoint save (best weights across E_r, E_h, E_m, D_r, Policy)
    ↓
ONNX Export:
    ├─ model_encoder.onnx (with encode_mode routing)
    └─ model_decoder.onnx (policy + D_r)
    ↓
Deployment:
    ├─ encode_mode=0 (g1):     E_r(g_r) → z → policy
    ├─ encode_mode=1 (teleop): E_m(g_m) → z → policy
    └─ encode_mode=2 (smpl):   E_h(g_h) → z → policy
```

**Key trait**: Multi-stage architecture with learned cross-modal alignment. Physics simulation provides the reward signal. Inference is modular: select encoder → generate token → run policy.

---

## 🔑 Core Architectural Differences

### 1. **Encoders**

**Current**: None. Observation fed directly to policy.

**SONIC**:
- **E_r (Robot Encoder)**: Compresses retargeted robot motion g_r → z_r
- **E_h (Human Encoder)**: Compresses SMPL motion g_h → z_h
- **E_m (Mixed Encoder)**: Compresses VR + lower-body g_m → z_m

All three map to the same latent dimension (e.g., 64-dim token) so the decoder is agnostic to source.

---

### 2. **Decoders**

**Current**: Single-stage policy decoder.
```python
SonicMLP(obs_dim=57, action_dim=35)
# obs[4] → action[8]
```

**SONIC**: Two-stage decoding.
- **D_r (Motion Decoder)**: Reconstructs robot trajectory from token.
  ```
  z [64-dim] → ĝ_r [10 frames × 29 joints]
  ```
- **Policy Decoder**: Executes actions in simulation.
  ```
  (z [64-dim] + proprioception) → action [29-dim]
  ```

---

### 3. **Loss Functions**

**Current**:
```python
loss = MSE(predicted_action, target_action)
```
Single supervised objective.

**SONIC**:
```python
L_total = L_PPO + λ_recon * L_recon + λ_token * L_token + λ_cycle * L_cycle
```

where:
- **L_PPO**: `∑ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)` (PPO surrogate)
- **L_recon**: `||D_r(z_r) - g_r||² + ||D_r(z_h) - g_r||² + ||D_r(z_m) - g_r||²` (forces all encoders to converge to same latent space)
- **L_token**: `||z_r - z_h||²` (direct cross-modal alignment in latent space)
- **L_cycle**: `||E_r(D_r(z_h)) - z_r||²` (consistency: encoding a z_h-decoded motion recovers z_r)

---

### 4. **Physics Simulation**

**Current**: ❌ None. Supervised on ground-truth actions.

**SONIC**: ✅ MuJoCo / Isaac Sim required.
- Roll out policy in physics environment
- Collect trajectory rollouts
- Compute motion-tracking reward
- Back-propagate PPO loss through simulation

---

### 5. **Data Representation**

**Current**:
```
Parquet (egocentric teleoperation)
├── observation.left_arm, right_arm, left_gripper, right_gripper
├── observation.left_ee, right_ee, body
└── action.left_arm, right_arm, ... (35-dim total)

→ Single flat vector per timestep
```

**SONIC**:
```
Three synchronized representations per motion:
├── g_r: Retargeted robot joint trajectory [T, 29]  (PyRoki output)
├── g_h: SMPL human motion [T, 24, 3]  (from MoCap)
└── g_m: Mixed (3-point VR + lower-body) [T, 11] (teleop input)

All prepared offline; correspondence pre-aligned
```

---

## 🚨 Key Gaps to Bridge

### Gap 1: Encoder Architecture
**Current**: No encoders.  
**Required**: Implement E_r, E_h, E_m as learnable modules.
```python
# pseudo-code
class RobotEncoder(nn.Module):
    def forward(self, g_r: [B, T, 29]) -> [B, 64]:
        # Compress T-frame robot trajectory to 64-dim token

class HumanEncoder(nn.Module):
    def forward(self, g_h: [B, T, 72]) -> [B, 64]:
        # Compress T-frame SMPL (24*3) to same 64-dim token
```

### Gap 2: Reconstruction Decoder
**Current**: Only policy; no explicit reconstruction.  
**Required**: Implement D_r that decodes any token back to robot motion.
```python
class MotionDecoder(nn.Module):
    def forward(self, z: [B, 64]) -> [B, T, 29]:
        # Reconstruct robot trajectory from token
```

### Gap 3: PPO Simulation Loop
**Current**: Supervised forward/backward pass.  
**Required**: Full PPO training loop with MuJoCo rollouts.
```python
# pseudo-code
for step in num_steps:
    trajectory = self.rollout_policy_in_mujoco(...)
    reward = self.compute_motion_tracking_reward(trajectory)
    advantage = self.compute_gae(reward)
    ppo_loss = self.compute_ppo_loss(advantage)
    
    # Back-prop all losses together
    loss = ppo_loss + L_recon + L_token + L_cycle
    loss.backward()
```

### Gap 4: Multi-Modal Data Pipeline
**Current**: Single flat observation from Parquet.  
**Required**: Load and align g_r, g_h, g_m triplets.
```python
class SonicMotionDataset(Dataset):
    def __getitem__(self, idx):
        g_r = load_from_csv("joint_pos.csv")     # robot
        g_h = load_from_csv("smpl_joint.csv")    # human
        g_m = combine_body_and_lower()           # mixed
        return {"g_r": g_r, "g_h": g_h, "g_m": g_m}
```

### Gap 5: Combined Loss & Hyper-Parameters
**Current**: Single MSE loss.  
**Required**: Implement combined loss with λ tuning.
```python
def sonic_loss(z_r, z_h, z_m, g_r, D_r, E_r, ppo_loss, lambda_recon=1.0, ...):
    L_recon = ...
    L_token = ...
    L_cycle = ...
    return ppo_loss + lambda_recon * L_recon + lambda_token * L_token + ...
```

### Gap 6: ONNX Export & Encode-Mode Routing
**Current**: Simple .pt checkpoint load.  
**Required**: Export model_encoder.onnx with runtime encode_mode selection.
```python
# Pseudo-code
def export_encoder(E_r, E_h, E_m, encode_mode=0):
    # Wrap all three encoders + router in a single ONNX module
    # Router selects E_r, E_h, or E_m based on encode_mode input
    return onnx_model
```

---

## 🗂️ Implementation Checklist

| Item | Current | Todo | Priority |
|------|---------|------|----------|
| Supervised baseline (SonicMLP) | ✅ Done | — | — |
| Egocentric dataset loader | ✅ Done | — | — |
| Test training on parquet data | ✅ Running | — | — |
| **E_r, E_h, E_m encoders** | ❌ — | ⬜ Create encoders.py | 🔴 HIGH |
| **D_r motion decoder** | ❌ — | ⬜ Create decoders.py | 🔴 HIGH |
| **SonicMotionDataset** | ❌ — | ⬜ Extend data_loader.py | 🔴 HIGH |
| **PPO training loop** | ❌ — | ⬜ Create ppo_trainer.py | 🔴 HIGH |
| **Motion tracking rewards** | ❌ — | ⬜ Create rewards.py | 🔴 HIGH |
| **Combined loss** | ❌ — | ⬜ Create losses.py | 🔴 HIGH |
| **MuJoCo simulation bridge** | ⚠️ Partial | ⬜ Integrate full PPO loop | 🟠 MEDIUM |
| **ONNX export** | ❌ — | ⬜ Create export.py | 🟠 MEDIUM |
| **Encode-mode routing** | ❌ — | ⬜ Add to model_encoder.onnx | 🟠 MEDIUM |
| **Documentation & examples** | ⚠️ Partial | ⬜ Full end-to-end example | 🟡 LOW |

---

## 📋 Deployment Readiness

| Aspect | Current Status | SONIC Full Status |
|--------|---|---|
| **Train locally** | ✅ Ready (supervised baseline) | ⏳ In progress (need encoders + PPO) |
| **Export to .onnx** | ⚠️ Partial (.pt only) | ❌ Not yet (need encode-mode router) |
| **Deploy to g1_deploy_onnx_ref** | ⚠️ Manual adapter needed | ✅ Native support (designed for) |
| **Test on real robot** | ❌ Not recommended (supervised only) | ✅ Target use case |
| **Simulation validation** | ❌ No simulator integration | ⏳ Needed for training |

---

## 🎓 Why This Progression?

1. **Current baseline** is fast to iterate:
   - No simulator needed (reduce dev friction)
   - Parquet data readily available
   - Single model easy to debug
   - Good for understanding data modalities

2. **Full SONIC** is production-grade:
   - PPO + simulation = true RL (robust to real-world deployment)
   - Multi-encoder design = flexible inference (switch modalities at runtime)
   - Cross-modal alignment = learned human-to-robot mapping (no manual retargeting)
   - ONNX export = native integration with `g1_deploy_onnx_ref`

---

## 🔮 Next Steps (in order)

1. **[HIGH]** Implement E_r, E_h, E_m encoders in `encoders.py`
2. **[HIGH]** Implement D_r motion decoder in `decoders.py`
3. **[HIGH]** Create `SonicMotionDataset` to load g_r, g_h, g_m triplets
4. **[HIGH]** Implement `ppo_trainer.py` with full PPO loop + MuJoCo
5. **[HIGH]** Create `losses.py` with combined loss function
6. **[MEDIUM]** Implement motion-tracking `rewards.py`
7. **[MEDIUM]** Create `export.py` for ONNX with encode-mode routing
8. **[LOW]** Add comprehensive examples and documentation

---

## 📚 Reference Files

- **Current code**: `gear_sonic/training/{data_loader,model,trainer,train}.py`
- **To create**: `gear_sonic/training/{encoders,decoders,ppo_trainer,losses,rewards,export}.py`
- **Config**: Update `gear_sonic/training/config.yaml` with PPO + λ hyper-parameters
- **Data**: `/home/grease/ego_dataset/work_bearlu/data/` (currently parquet; will need g_r, g_h, g_m CSVs)

---

**Status**: Baseline ready ✅ | Full SONIC in design phase 📋
