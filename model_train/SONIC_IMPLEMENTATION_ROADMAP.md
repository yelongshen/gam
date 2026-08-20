# 🗺️ SONIC Full Training Implementation Roadmap

## Overview

This roadmap details the step-by-step process to evolve the current supervised baseline into the full SONIC encoder-decoder PPO system for multi-modal humanoid motion learning.

---

## 🎯 Phase 1: Foundation (Weeks 1-2)

### 1.1 Design Phase

**Goal**: Finalize architecture decisions

- [ ] Review SONIC paper (Araujo et al., 2025) appendices for architecture details
- [ ] Decide on latent dimension (recommend 64 or 128)
- [ ] Decide on encoder architecture (Transformer vs. MLP vs. CNN)
- [ ] Decide on time window size (T: recommend 10-20 frames)
- [ ] Document design choices in `gear_sonic/training/DESIGN.md`

**Deliverable**: `DESIGN.md` with final architecture spec

---

### 1.2 Environment Setup

**Goal**: Ensure MuJoCo and physics simulation are ready

- [ ] Verify MuJoCo installation: `python -c "import mujoco; print(mujoco.__version__)"`
- [ ] Test `gear_sonic/utils/mujoco_sim/base_sim.py` with a simple rollout
- [ ] Ensure G1 robot model loads correctly in MuJoCo
- [ ] Document motion tracking reward computation
- [ ] Create `gear_sonic/training/utils/mujoco_wrapper.py` for training integration

**Deliverable**: Verified MuJoCo simulation environment ready for rollouts

---

### 1.3 Data Preparation

**Goal**: Prepare g_r, g_h, g_m representations

**Current state**: Parquet files with flat 57-dim observations

**Required**:
- [ ] Source SMPL sequences (g_h) - from existing teleoperation MoCap or download public dataset
- [ ] Source retargeted robot sequences (g_r) - use PyRoki to convert g_h to G1
- [ ] Create mixed representation (g_m) - extract 3-point VR + lower body
- [ ] Create CSV file format for g_r, g_h, g_m
- [ ] Implement CSV loader in data preparation script

**Deliverable**: Sample CSV data triplets (g_r, g_h, g_m) for one motion

---

## 🏗️ Phase 2: Encoder Implementation (Weeks 2-3)

### 2.1 RobotEncoder (E_r)

**File**: `gear_sonic/training/encoders.py`

```python
class RobotEncoder(nn.Module):
    """Encodes retargeted robot joint trajectory to latent token."""
    
    def __init__(self, input_dim=29, time_steps=10, latent_dim=64):
        # input: [B, T=10, 29] (robot joints)
        # output: [B, 64] (latent token)
        pass
    
    def forward(self, g_r):
        # Compress temporal sequence to fixed-size token
        pass
```

**Implementation options**:
- Transformer encoder (1-2 layers) + pooling to get token
- CNN (1D convolution) + pooling
- MLP with temporal positional encoding

**Testing**:
- [ ] Test forward pass: `E_r(torch.randn(4, 10, 29))` → `[4, 64]`
- [ ] Test gradients: `loss.backward()` flows through E_r
- [ ] Profile speed: should be <1ms per forward pass

**Deliverable**: Working `RobotEncoder` with tests

---

### 2.2 HumanEncoder (E_h)

**File**: `gear_sonic/training/encoders.py`

```python
class HumanEncoder(nn.Module):
    """Encodes SMPL human motion to same latent space as robot."""
    
    def __init__(self, input_dim=72, time_steps=10, latent_dim=64):
        # input: [B, T=10, 72] (SMPL 24 joints × 3 coords)
        # output: [B, 64] (same latent space as E_r)
        pass
    
    def forward(self, g_h):
        # Compress to same 64-dim space
        pass
```

**Key requirement**: Output must be same dimension as E_r for L_token alignment

**Testing**:
- [ ] Test forward pass: `E_h(torch.randn(4, 10, 72))` → `[4, 64]`
- [ ] Verify output dimension matches E_r output

**Deliverable**: Working `HumanEncoder` with dimension alignment

---

### 2.3 MixedEncoder (E_m)

**File**: `gear_sonic/training/encoders.py`

```python
class MixedEncoder(nn.Module):
    """Encodes mixed VR + lower-body representation."""
    
    def __init__(self, input_dim=11, time_steps=10, latent_dim=64):
        # input: [B, T=10, 11] (3-point VR + 8 lower-body joints)
        # output: [B, 64] (same latent space)
        pass
    
    def forward(self, g_m):
        pass
```

**Composition of g_m**:
- Head position + orientation: 6-dim (or 3D position)
- Left wrist position + orientation: 6-dim
- Right wrist position + orientation: 6-dim
- Lower-body robot state: 8-dim (hip, knee, ankle joints)
- **Total**: 11-26 dim depending on representation choice

**Deliverable**: Working `MixedEncoder`

---

## 🎯 Phase 3: Decoder Implementation (Weeks 3-4)

### 3.1 MotionDecoder (D_r)

**File**: `gear_sonic/training/decoders.py`

```python
class MotionDecoder(nn.Module):
    """Decodes latent token to robot motion trajectory."""
    
    def __init__(self, latent_dim=64, time_steps=10, output_dim=29):
        # input: [B, 64] (latent token)
        # output: [B, T=10, 29] (reconstructed robot trajectory)
        pass
    
    def forward(self, z):
        # Expand token to full trajectory
        pass
```

**Architecture choices**:
- Transformer decoder (expands token to sequence)
- MLP + reshape (expand to [B, 29*T])
- RNN decoder (recurrent generation)

**Key point**: This decoder is **shared** across all three encoders (E_r, E_h, E_m)

**Testing**:
- [ ] Test forward: `D_r(torch.randn(4, 64))` → `[4, 10, 29]`
- [ ] Test reconstruction loss: `L_recon = MSE(D_r(z_r), g_r)`
- [ ] Verify D_r can reconstruct from all three encoder outputs

**Deliverable**: Working `MotionDecoder` with reconstruction capability

---

### 3.2 RL Policy Decoder

**File**: `gear_sonic/training/decoders.py`

```python
class PolicyDecoder(nn.Module):
    """Decodes latent token + proprioception to motor action."""
    
    def __init__(self, latent_dim=64, proprioception_dim=50, action_dim=29):
        # input: [B, 64 + 50] (latent + proprioception)
        # output: [B, 29] (motor command)
        pass
    
    def forward(self, z, proprioception):
        # Generate action
        pass
```

**Proprioception components**:
- Current joint positions (29-dim)
- Current joint velocities (29-dim)
- Base height, orientation (6-dim)
- Contact state (4-dim, 4 feet)
- History (e.g., last 3 actions: 3×29-dim)

**Total**: ~100-150-dim (depends on how much history to include)

**Deliverable**: Working `PolicyDecoder`

---

## 🔄 Phase 4: PPO Training Loop (Weeks 4-6)

### 4.1 Create PPOTrainer Class

**File**: `gear_sonic/training/ppo_trainer.py`

```python
class PPOTrainer:
    """PPO training loop with physics simulation."""
    
    def __init__(self, env, policy, num_steps=1000):
        self.env = env  # MuJoCo simulator
        self.policy = policy  # E_r/E_h/E_m + D_r + PolicyDecoder
        pass
    
    def collect_rollout(self, num_steps):
        """Roll out policy in MuJoCo, collect (s,a,r,s',d) tuples."""
        pass
    
    def compute_gae(self, trajectories, gamma=0.99, lambda_=0.95):
        """Compute GAE advantages."""
        pass
    
    def compute_ppo_loss(self, trajectories, old_policy):
        """Compute PPO surrogate loss."""
        pass
    
    def train_step(self):
        """Single training step: collect rollout, compute loss, backward."""
        pass
```

**Key components**:
- [ ] MuJoCo environment wrapper
- [ ] Rollout collection loop
- [ ] Value function network (for advantage computation)
- [ ] GAE (Generalized Advantage Estimation)
- [ ] PPO clipped surrogate loss

**Deliverable**: Working `PPOTrainer` that can run one full step

---

### 4.2 Reward Function Implementation

**File**: `gear_sonic/training/rewards.py`

```python
class MotionTrackingReward:
    """Compute rewards for motion tracking task."""
    
    def __init__(self, ref_trajectory, weights=None):
        self.ref = ref_trajectory  # [T, 29]
        self.weights = weights or {
            'joint_tracking': 1.0,
            'root_tracking': 0.5,
            'balance': 0.5,
            'smoothness': 0.1,
            'torque': 0.01,
        }
    
    def compute(self, q_robot, q_dot, tau, t):
        """Compute total reward at timestep t."""
        r_joint = self.r_joint_tracking(q_robot, t)
        r_root = self.r_root_tracking(q_robot, t)
        r_balance = self.r_balance(q_dot)
        r_smooth = self.r_smoothness(tau)
        r_torque = self.r_torque_penalty(tau)
        
        return (self.weights['joint_tracking'] * r_joint +
                self.weights['root_tracking'] * r_root +
                self.weights['balance'] * r_balance +
                self.weights['smoothness'] * r_smooth +
                self.weights['torque'] * r_torque)
    
    def r_joint_tracking(self, q_robot, t):
        """Joint position tracking: exp(-||q - q_ref||^2)"""
        pass
    
    def r_root_tracking(self, q_robot, t):
        """Root pose tracking (height, orientation)"""
        pass
    
    def r_balance(self, q_dot):
        """Penalize large angular velocity"""
        pass
    
    def r_smoothness(self, tau):
        """Penalize action discontinuity"""
        pass
    
    def r_torque_penalty(self, tau):
        """Penalize large motor torques"""
        pass
```

**Reward tuning**: Test different weight combinations to get good motion tracking

**Deliverable**: Working `MotionTrackingReward` that produces reasonable reward signals

---

### 4.3 Combined Loss Function

**File**: `gear_sonic/training/losses.py`

```python
def sonic_combined_loss(z_r, z_h, z_m, g_r, D_r, E_r,
                       ppo_loss, advantages,
                       lambda_recon=1.0, lambda_token=0.5, lambda_cycle=0.1):
    """Compute total SONIC loss."""
    
    # Reconstruction loss: all encoders must reconstruct g_r via shared D_r
    g_r_pred_from_z_r = D_r(z_r)
    g_r_pred_from_z_h = D_r(z_h)
    g_r_pred_from_z_m = D_r(z_m)
    
    L_recon = (MSE(g_r_pred_from_z_r, g_r) +
               MSE(g_r_pred_from_z_h, g_r) +
               MSE(g_r_pred_from_z_m, g_r))
    
    # Token alignment: pull robot and human tokens together
    L_token = MSE(z_r, z_h)
    
    # Cycle consistency: E_r(D_r(z_h)) should recover z_r
    z_h_reconstructed = E_r(D_r(z_h))
    L_cycle = MSE(z_h_reconstructed, z_r.detach())
    
    # Total loss
    L_total = (ppo_loss +
               lambda_recon * L_recon +
               lambda_token * L_token +
               lambda_cycle * L_cycle)
    
    return L_total, {
        'L_PPO': ppo_loss.item(),
        'L_recon': L_recon.item(),
        'L_token': L_token.item(),
        'L_cycle': L_cycle.item(),
        'L_total': L_total.item(),
    }
```

**Hyperparameter tuning**:
- λ_recon: Higher → stronger reconstruction constraint
- λ_token: Higher → stronger cross-modal alignment
- λ_cycle: Higher → stronger consistency

**Deliverable**: Working combined loss function

---

## 📊 Phase 5: Data Pipeline (Weeks 3-4, parallel)

### 5.1 SonicMotionDataset

**File**: `gear_sonic/training/data_loader.py` (extend)

```python
class SonicMotionDataset(Dataset):
    """Load triplets of synchronized motion representations."""
    
    def __init__(self, data_root, context_length=10):
        # data_root/
        #   ├── motion_001/
        #   │   ├── g_r.csv  [T, 29]
        #   │   ├── g_h.csv  [T, 72]
        #   │   └── g_m.csv  [T, 11]
        #   ├── motion_002/
        #   └── ...
        
        self.motions = self.load_all_motions(data_root)
        self.context_length = context_length
    
    def __getitem__(self, idx):
        motion_idx, start_frame = self.index_to_motion_frame(idx)
        motion = self.motions[motion_idx]
        
        window = self.context_length
        g_r = motion['g_r'][start_frame : start_frame + window]
        g_h = motion['g_h'][start_frame : start_frame + window]
        g_m = motion['g_m'][start_frame : start_frame + window]
        
        return {
            'g_r': torch.from_numpy(g_r).float(),
            'g_h': torch.from_numpy(g_h).float(),
            'g_m': torch.from_numpy(g_m).float(),
        }
```

**Key requirements**:
- [ ] Load g_r, g_h, g_m from CSV with shape preservation
- [ ] Handle variable-length sequences
- [ ] Ensure temporal alignment across triplets
- [ ] Create train/val split

**Deliverable**: Working `SonicMotionDataset` that yields aligned triplets

---

## 🔧 Phase 6: Integration & Testing (Weeks 6-7)

### 6.1 End-to-End Training Script

**File**: `gear_sonic/training/train_sonic_ppo.py`

```python
def train_sonic_ppo(config_path):
    """Main SONIC PPO training loop."""
    config = load_yaml(config_path)
    
    # Setup
    device = torch.device('cuda')
    
    # Create dataset
    train_loader = SonicMotionDataset(config['data_root'], 
                                      context_length=10)
    
    # Create models
    E_r = RobotEncoder(latent_dim=64)
    E_h = HumanEncoder(latent_dim=64)
    E_m = MixedEncoder(latent_dim=64)
    D_r = MotionDecoder(latent_dim=64)
    policy = PolicyDecoder(latent_dim=64, action_dim=29)
    
    # Create trainer
    trainer = PPOTrainer(
        env=MuJoCoSimulator(config),
        encoders={'E_r': E_r, 'E_h': E_h, 'E_m': E_m},
        decoders={'D_r': D_r, 'policy': policy},
        config=config
    )
    
    # Training loop
    for epoch in range(config['num_epochs']):
        loss_dict = trainer.train_step()
        print(f"Epoch {epoch}: {loss_dict}")
        
        if epoch % config['val_interval'] == 0:
            val_loss = trainer.validate()
            trainer.save_checkpoint(f"checkpoint_{epoch}.pt")
```

**Testing checklist**:
- [ ] Test forward pass through all encoders + decoders
- [ ] Test rollout in MuJoCo
- [ ] Test reward computation
- [ ] Test combined loss computation
- [ ] Test backward pass updates all parameters
- [ ] Test one full training step end-to-end

**Deliverable**: End-to-end training script that runs without errors

---

### 6.2 Configuration File

**File**: `gear_sonic/training/config_sonic_ppo.yaml`

```yaml
# Data
data_root: /path/to/motion/triplets
context_length: 10

# Encoders
encoder:
  type: transformer  # or mlp
  latent_dim: 64
  hidden_dim: 256
  num_layers: 2

# Decoders
decoder:
  type: transformer
  hidden_dim: 256
  num_layers: 2

# PPO
ppo:
  learning_rate: 3e-4
  clip_ratio: 0.2
  epochs_per_rollout: 10
  num_mini_batches: 32
  gamma: 0.99
  lambda_gae: 0.95

# Losses
losses:
  lambda_recon: 1.0
  lambda_token: 0.5
  lambda_cycle: 0.1

# Training
num_epochs: 200
batch_size: 32
num_workers: 4
checkpointing:
  save_interval: 10
  val_interval: 5

# Simulation
mujoco:
  timestep: 0.01
  steps_per_second: 100

# Reward weights
reward:
  joint_tracking: 1.0
  root_tracking: 0.5
  balance: 0.5
  smoothness: 0.1
  torque: 0.01
```

**Deliverable**: Complete YAML config for SONIC PPO training

---

## 📤 Phase 7: Export & Deployment (Weeks 7-8)

### 7.1 ONNX Export

**File**: `gear_sonic/training/export.py`

```python
def export_encoder_with_router(E_r, E_h, E_m, output_dir):
    """Export encoders as single ONNX module with encode_mode routing."""
    
    # Wrap in router module
    class EncoderRouter(nn.Module):
        def __init__(self, E_r, E_h, E_m):
            super().__init__()
            self.E_r = E_r
            self.E_h = E_h
            self.E_m = E_m
        
        def forward(self, x, encode_mode):
            if encode_mode == 0:
                return self.E_r(x)
            elif encode_mode == 1:
                return self.E_m(x)
            else:  # encode_mode == 2
                return self.E_h(x)
    
    router = EncoderRouter(E_r, E_h, E_m)
    
    # Export to ONNX
    torch.onnx.export(
        router,
        (torch.randn(1, 10, 72), torch.tensor(0)),
        f"{output_dir}/model_encoder.onnx",
        input_names=['input_data', 'encode_mode'],
        output_names=['token'],
        ...
    )

def export_decoder(D_r, policy_decoder, output_dir):
    """Export decoder (D_r + policy) as ONNX."""
    # Similar export logic
    pass
```

**Verification**:
- [ ] Load exported ONNX with onnx runtime
- [ ] Test forward pass
- [ ] Verify output shapes match deployment expectations

**Deliverable**: Valid ONNX files (model_encoder.onnx, model_decoder.onnx)

---

### 7.2 TensorRT Compilation

**File**: `gear_sonic/training/compile_tensorrt.py`

```bash
# On Unitree Orin (or compatible device):
trtexec --onnx=model_encoder.onnx --saveEngine=model_encoder.trt
trtexec --onnx=model_decoder.onnx --saveEngine=model_decoder.trt
```

**Deliverable**: Compiled .trt files for deployment

---

### 7.3 Integration with g1_deploy_onnx_ref

**File**: `gear_sonic_deploy/policy/release/` (already exists)

- [ ] Place model_encoder.onnx and model_decoder.onnx in deployment directory
- [ ] Update observation_config.yaml with correct encoder dimension
- [ ] Test with `./target/release/g1_deploy_onnx_ref` C++ binary
- [ ] Run on real G1 robot

**Deliverable**: Deployed model on G1 robot

---

## 📋 Implementation Checklist

| Phase | Milestone | Status | ETA |
|-------|-----------|--------|-----|
| **Phase 1** | Foundation | ⬜ | Week 1-2 |
| 1.1 | Design finalized | ⬜ | Day 2 |
| 1.2 | MuJoCo verified | ⬜ | Day 3 |
| 1.3 | Data prepared | ⬜ | Day 5 |
| **Phase 2** | Encoders | ⬜ | Week 2-3 |
| 2.1 | RobotEncoder | ⬜ | Day 6 |
| 2.2 | HumanEncoder | ⬜ | Day 8 |
| 2.3 | MixedEncoder | ⬜ | Day 9 |
| **Phase 3** | Decoders | ⬜ | Week 3-4 |
| 3.1 | MotionDecoder | ⬜ | Day 10 |
| 3.2 | PolicyDecoder | ⬜ | Day 12 |
| **Phase 4** | PPO Loop | ⬜ | Week 4-6 |
| 4.1 | PPOTrainer | ⬜ | Day 15 |
| 4.2 | Reward function | ⬜ | Day 18 |
| 4.3 | Combined loss | ⬜ | Day 20 |
| **Phase 5** | Data Pipeline | ⬜ | Week 3-4 (parallel) |
| 5.1 | SonicMotionDataset | ⬜ | Day 12 |
| **Phase 6** | Integration | ⬜ | Week 6-7 |
| 6.1 | End-to-end training | ⬜ | Day 25 |
| 6.2 | Configuration | ⬜ | Day 26 |
| **Phase 7** | Deployment | ⬜ | Week 7-8 |
| 7.1 | ONNX export | ⬜ | Day 28 |
| 7.2 | TensorRT compilation | ⬜ | Day 30 |
| 7.3 | Robot deployment | ⬜ | Day 32 |

**Total estimated timeline**: 8 weeks (with some parallelization)

---

## 🚀 Quick Start Template

To get started immediately:

```bash
# 1. Create the files
touch gear_sonic/training/{encoders,decoders,ppo_trainer,rewards,losses}.py
touch gear_sonic/training/{train_sonic_ppo,export,compile_tensorrt}.py

# 2. Implement Phase 1: Foundation
# - Document design choices
# - Verify MuJoCo
# - Prepare data

# 3. Implement Phase 2: Encoders (start with E_r)
# - Copy skeleton code
# - Implement forward pass
# - Test with random inputs

# 4. Iteratively add other phases
```

---

**Next action**: Start with Phase 1.1 (Design finalization). Review the SONIC paper appendices and document final architecture choices in `gear_sonic/training/DESIGN.md`.
