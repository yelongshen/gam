# SONIC Architecture Diagrams

## Current Implementation vs. Full SONIC Flow

### 1️⃣ CURRENT: Supervised Action Prediction

```
┌─────────────────────────────────────────────────────────────┐
│ CURRENT IMPLEMENTATION (Baseline)                           │
└─────────────────────────────────────────────────────────────┘

Egocentric Parquet Data
    ↓
    obs: [4 frames × 57-dim]    action: [8 frames × 35-dim]
    (left_arm, right_arm, left_gripper, right_gripper,
     left_ee, right_ee, body)
    ↓
┌─────────────────────────────┐
│  DataLoader                 │
│  • Load parquet             │
│  • Normalize obs/action     │
│  • 80/20 train/val split    │
└──────────────┬──────────────┘
               ↓
        ┌──────────────┐
        │   SonicMLP   │  ←  Single model (786K params)
        │   or         │     Takes raw obs, predicts action
        │ Transformer  │     No intermediate representation
        │ (1.7M params)│
        └──────┬───────┘
               ↓
        forward: obs[4, 57] → action[8, 35]
        loss:    MSE(action_pred, action_target)
        backward: update weights
               ↓
        ┌──────────────────────┐
        │  Checkpoint          │
        │  best_model.pt       │
        └──────────────────────┘
               ↓
        Inference: Load .pt → Direct prediction

⏱️  Training: ~50 hours for 200 epochs
📊 Loss curve: Supervised MSE only
🚀 Deployment: Not designed for real robot (supervised only)
```

---

### 2️⃣ FULL SONIC: Multi-Encoder PPO System

```
┌───────────────────────────────────────────────────────────────┐
│ FULL SONIC TRAINING FLOW (Production)                         │
└───────────────────────────────────────────────────────────────┘

Multi-Modal Motion Dataset
    ├─ g_r: Robot motion [T, 29]        (retargeted via PyRoki)
    ├─ g_h: Human motion [T, 24, 3]     (SMPL from MoCap)
    └─ g_m: Mixed [T, 11]               (VR 3-point + lower-body)
    ↓
┌─────────────────────────────────────┐
│  SonicMotionDataset                 │
│  • Load g_r, g_h, g_m triplets      │
│  • Align temporal correspondence    │
│  • 80/20 train/val split            │
└──────────────┬──────────────────────┘
               ↓
     ┌─────────────────────────────────────┐
     │   FORWARD PASS (each training step) │
     └─────────────────────────────────────┘
               ↓
    ┌──────────────────────────────────────┐
    │  Encoders (parallel)                 │
    ├──────────────────────────────────────┤
    │  E_r(g_r) → z_r [64-dim]            │  ← Compress 3 modalities
    │  E_h(g_h) → z_h [64-dim]            │    into shared latent
    │  E_m(g_m) → z_m [64-dim]            │    space
    └──────┬──────────────────────────────┘
           ↓
    ┌──────────────────────────────────────┐
    │  Decoders (parallel + policy)        │
    ├──────────────────────────────────────┤
    │  D_r(z_r) → ĝ_r                     │  ← Reconstruction
    │  D_r(z_h) → ĝ_r                     │    (L_recon loss)
    │  D_r(z_m) → ĝ_r                     │
    │                                      │
    │  Policy(z + proprioception) → action │  ← RL action
    └──────┬──────────────────────────────┘
           ↓
    ┌──────────────────────────────────────┐
    │  Physics Simulation (MuJoCo/Isaac)   │
    │  • Roll out action in simulator      │
    │  • Collect trajectory                │
    │  • Compute motion-tracking reward    │
    └──────┬──────────────────────────────┘
           ↓
     ┌─────────────────────────────────────┐
     │   LOSS COMPUTATION (4 objectives)   │
     ├─────────────────────────────────────┤
     │  L_PPO   = PPO surrogate            │  ← From RL
     │  L_recon = ||D_r(z_r) - g_r||²     │  ← Reconstruction
     │          + ||D_r(z_h) - g_r||²     │
     │          + ||D_r(z_m) - g_r||²     │
     │  L_token = ||z_r - z_h||²           │  ← Cross-modal align
     │  L_cycle = ||E_r(D_r(z_h))-z_r||²  │  ← Consistency
     │                                      │
     │  L_total = L_PPO                    │
     │          + λ_recon·L_recon          │
     │          + λ_token·L_token          │
     │          + λ_cycle·L_cycle          │
     └──────┬──────────────────────────────┘
            ↓
     ┌────────────────────────────────────┐
     │  BACKWARD PASS (single backward)   │
     │  Updates: E_r, E_h, E_m, D_r,     │
     │           Policy decoder           │
     └──────┬─────────────────────────────┘
            ↓
    ┌────────────────────────────────────┐
    │  Checkpointing                     │
    │  • best_weights_{E_r,E_h,E_m,D_r} │
    │  • checkpoint_epoch_*.pt           │
    └────────────────────────────────────┘
            ↓
     ┌─────────────────────────────────────┐
     │   ONNX EXPORT & DEPLOYMENT          │
     ├─────────────────────────────────────┤
     │  model_encoder.onnx                 │  ← E_r/E_h/E_m router
     │  model_decoder.onnx                 │  ← D_r + Policy
     │                                      │
     │  At runtime, select encoder:        │
     │  • encode_mode=0 (g1):    use E_r  │
     │  • encode_mode=1 (teleop): use E_m │
     │  • encode_mode=2 (smpl):   use E_h │
     └─────────────────────────────────────┘
            ↓
    Deploy to g1_deploy_onnx_ref (C++ binary)
    → Ready for real robot

⏱️  Training: ~100-200 hours (includes sim rollouts)
📊 Loss curve: Combined PPO + recon + alignment + consistency
🚀 Deployment: Native support in g1_deploy_onnx_ref
```

---

## 3️⃣ Side-by-Side Module Comparison

### Input/Output

```
CURRENT:
    Input:  obs_context [B, 4, 57]
    Output: action [B, 8, 35]
    
    Single end-to-end pipeline

SONIC:
    Input:  g_r [B, T, 29] OR g_h [B, T, 72] OR g_m [B, T, 11]
    ├─ Encode → z [B, 64]
    └─ Decode (z + proprioception) → action [B, 29]
    
    Modular: encoder selection at runtime
```

### Model Count

```
CURRENT (3 architectures):
    ├─ SonicMLP         : 786K params
    ├─ SonicActionPredictor (Transformer): 1.7M params
    └─ Total models: 2 (choose one per training)

SONIC (5 modules):
    ├─ E_r  (RobotEncoder)    : ~50-100K params
    ├─ E_h  (HumanEncoder)    : ~50-100K params
    ├─ E_m  (MixedEncoder)    : ~50-100K params
    ├─ D_r  (MotionDecoder)   : ~100-150K params
    ├─ Policy (Decoder + RL)  : ~500-1000K params
    └─ Total: ~1-2M params (scales with z_dim and decoder size)
```

### Data Flow

```
CURRENT:
    Parquet
      ↓
    [obs] → [Model] → [action]

SONIC:
    CSV (g_r, g_h, g_m)
      ↓
    [Encoder] → [z] → [Decoder (recon)] → [ĝ_r]
                ↓
              [Decoder (policy) + proprioception] → [action]
                ↓
              [Simulate in MuJoCo] → [reward]
```

---

## 4️⃣ Loss Function Comparison

### Current Loss

```python
# Single supervised loss
loss = MSE(predicted_action, target_action)

# Backward updates only the single model
model.backward()
```

**Graph**:
```
obs_context
    ↓
 [Model]
    ↓
action_pred
    ↓
    MSE(action_pred, action_target)
    ↓
backward() → update model weights
```

---

### SONIC Combined Loss

```python
# Four objectives computed simultaneously
L_total = L_PPO + λ_recon * L_recon + λ_token * L_token + λ_cycle * L_cycle

# Single backward updates all encoders, decoders, policy
loss.backward()
```

**Computation Graph**:
```
                g_r, g_h, g_m
                     ↓
        ┌────────────┴────────────┐
        ↓            ↓            ↓
    [E_r]        [E_h]        [E_m]
        ↓            ↓            ↓
    z_r           z_h           z_m
        ├────────────┼────────────┤
        ↓            ↓            ↓
    [D_r]        [D_r]        [D_r]    ← Shared decoder
        ↓            ↓            ↓
    ĝ_r          ĝ_r          ĝ_r
        ├────────────┼────────────┤
        └────────────┼────────────┘
                     ↓
    L_recon = MSE(ĝ_r, g_r)  (all three terms)
                     
    L_token = MSE(z_r, z_h)
    
    L_cycle = MSE(E_r(D_r(z_h)), z_r)
    
        z (select one: z_r, z_h, or z_m)
        + proprioception
        ↓
    [Policy Decoder]
        ↓
    action
        ↓
    [Simulate in MuJoCo]
        ↓
    reward
        ↓
    L_PPO = PPO surrogate
        ↓
    L_total = L_PPO + λ_recon·L_recon + λ_token·L_token + λ_cycle·L_cycle
        ↓
    backward() → update E_r, E_h, E_m, D_r, Policy
```

---

## 5️⃣ Inference Pipeline Comparison

### Current

```
Load model_checkpoint.pt
    ↓
obs_context [4, 57]
    ↓
model(obs_context)
    ↓
action [8, 35]
```

**Simple, direct, single-modality.**

---

### SONIC

```
encode_mode = select_from_config()  # 0=g1, 1=teleop, 2=smpl
    ↓
input_data = get_observation(encode_mode)
    ├─ encode_mode=0: g_r (retargeted motion)
    ├─ encode_mode=1: g_m (VR + lower-body)
    └─ encode_mode=2: g_h (SMPL from PICO)
    ↓
z = model_encoder(input_data, encode_mode)
    ↓
    [Internally: router selects E_r / E_m / E_h]
    ↓
proprioception = get_robot_state()  # Joint positions, velocities
    ↓
obs = concatenate(z, proprioception)
    ↓
action = model_decoder(obs)
    ↓
execute_action(action)
```

**Modular, multi-modality, runtime-selectable encoder.**

---

## 6️⃣ Deployment Path

### Current

```
Python Training
    ↓
best_model.pt (PyTorch checkpoint)
    ↓
Can use for inference in Python, but:
    ✗ Not designed for C++ deployment
    ✗ No ONNX export
    ✗ Supervised baseline (not for real robot)
```

### SONIC

```
Python Training (PPO + multi-modal)
    ↓
Export to ONNX:
    ├─ model_encoder.onnx (with encode_mode router)
    └─ model_decoder.onnx (with motion tracking policy)
    ↓
TensorRT compilation (on Unitree Orin)
    ↓
Deploy to g1_deploy_onnx_ref (C++ binary)
    ├─ Read observation_config.yaml (encode_mode)
    ├─ Load model_encoder.onnx + model_decoder.onnx
    └─ Real-time control loop: obs → action @ 50Hz
```

---

## Summary Table: Architecture Metrics

| Metric | Current | SONIC |
|--------|---------|-------|
| **Encoders** | 0 | 3 (E_r, E_h, E_m) |
| **Decoders** | 1 (policy only) | 2 (D_r + policy) |
| **Latent dim** | 0 (no latent) | 64 (configurable) |
| **Losses** | 1 (MSE) | 4 (PPO + recon + token + cycle) |
| **Simulation** | ❌ | ✅ |
| **RL** | ❌ | ✅ |
| **Cross-modal** | ❌ | ✅ |
| **ONNX export** | ⚠️ | ✅ |
| **C++ deployment** | ❌ | ✅ |

---

## Gradient Flow Visualization

### Current: Direct Gradient Path

```
obs ──→ [Model] ──→ action_pred ──→ MSE loss
  ↓                                     ↓
  └─ gradients ←──────────────────────┘
     update model weights only
```

**Simple**: Single gradient path through one model.

---

### SONIC: Complex Multi-Path Gradient Flow

```
    g_r              g_h              g_m
    ↓                ↓                ↓
[E_r] ──→ z_r    [E_h] ──→ z_h    [E_m] ──→ z_m
    ↓                ↓                ↓
    └────────────────────┬────────────┘
                         ↓
                      [D_r]
                    ↙  ↓  ↖
              L_recon  │  L_token
                       ↓
                    z_r,z_h
                       ↓
                    Policy (select one z)
                       ↓
                    Simulate
                       ↓
                    L_PPO, L_cycle
                       ↓
            L_total = L_PPO + λ_recon·L_recon + λ_token·L_token + λ_cycle·L_cycle
                       ↓
            backward() flows through all paths simultaneously:
            • E_r, E_h, E_m updated via L_recon, L_token, L_cycle
            • D_r updated via L_recon, L_cycle
            • Policy updated via L_PPO
```

**Complex**: Gradient flows through multiple paths and losses, updating all modules simultaneously.

---

This comparison highlights why the full SONIC system is more powerful: it leverages multi-modal alignment and physics-based learning to produce policies that transfer better to real robots.
