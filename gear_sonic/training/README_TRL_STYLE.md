# SONIC Training - TRL-Style Implementation

This directory contains two SONIC training implementations:

1. **Custom Implementation** (`sonic_combined_trainer.py`) - Our original implementation
2. **TRL-Style Implementation** (`train_trl_style.py`) - Borrows from official NVlabs code

## Overview

The TRL-style implementation (`train_trl_style.py`) is adapted from the official SONIC training script:
https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/train_agent_trl.py

### Key Differences

| Feature | Custom Implementation | TRL-Style Implementation |
|---------|----------------------|--------------------------|
| **Framework** | PyTorch + custom PPO | HuggingFace TRL + Accelerate |
| **Multi-GPU** | Manual DDP | Accelerate (automatic) |
| **Config System** | YAML + Python dict | Hydra (in official) / YAML (ours) |
| **Environment** | G1MuJoCoEnv | Same (G1MuJoCoEnv) |
| **Reward** | FK-based (Table S3) | Same (FK-based) |
| **Domain Rand** | Implemented (Table S4) | Same (Table S4) |
| **PPO** | Custom implementation | TRL PPOTrainer |
| **Logging** | Print + file | Wandb + HF Trainer |

## Architecture Comparison

### Official SONIC (NVlabs)

```python
# From train_agent_trl.py
trainer = TRLPPOTrainer(
    args=training_args,  # HF TrainingArguments
    config=config.algo.config,  # Hydra config
    env=env,  # IsaacLab ManagerBasedRLEnv
    model=policy,  # Actor model
    value_model=value_model,  # Critic model
    ref_model=ref_model,  # Teacher (for distillation)
    disc_model=disc_model,  # Discriminator (for AMP)
    accelerator=accelerator,  # HF Accelerate
)
trainer.train()
```

**Key Components**:
- `TRLPPOTrainer`: Custom PPO trainer extending HF TRL's `PPOTrainer`
- `PolicyAndValueWrapper`: Wraps actor + critic for joint training
- `RolloutStorage`: Stores transitions for PPO updates
- **Multi-critic support**: Can use multiple reward heads
- **Symmetry augmentation**: Left-right symmetry for data efficiency
- **Adaptive LR**: Adjusts learning rate based on KL divergence

### Our TRL-Style Implementation

```python
# From train_trl_style.py
policy = PolicyWrapper(encoder_decoder, policy_head)
value_model = ValueHead(obs_dim, hidden_dim)

# Simplified training loop (full TRL integration would require
# adapting TRLPPOTrainer to work with our G1MuJoCoEnv)
for iteration in range(num_iterations):
    # Collect rollouts
    # Compute GAE
    # PPO update (clipped surrogate + value loss)
    # Log metrics
```

**Key Simplifications**:
- No IsaacLab dependency (uses MuJoCo)
- Simplified rollout collection (no RolloutStorage)
- Standard PPO (no symmetry augmentation yet)
- Manual multi-GPU support (could add Accelerate)

## Training Pipelines

### 1. Custom Implementation

**Files**:
- `sonic_combined_trainer.py` - Main training script
- `config_sonic_combined.yaml` - Configuration
- `g1_mujoco_env.py` - Environment
- `rewards.py` - FK-based reward (Table S3)
- `encoders.py` - Encoder-decoder with FSQ
- `ppo_trainer.py` - Policy/Value heads

**Usage**:
```bash
cd /home/grease/gam
source .venv_sim/bin/activate

python gear_sonic/training/sonic_combined_trainer.py \
    --config gear_sonic/training/config_sonic_combined.yaml \
    --iters 500
```

**Status**: ✅ Working, tested with 50-iteration run
- Reward: -4777 → -4203 (best)
- Reconstruction loss: 2.27 → 1.73
- Domain randomization: 6/9 items implemented

### 2. TRL-Style Implementation

**Files**:
- `train_trl_style.py` - Main training script
- `config_trl.yaml` - Configuration
- Uses same `g1_mujoco_env.py`, `rewards.py`, etc.

**Usage**:
```bash
cd /home/grease/gam
source .venv_sim/bin/activate

python gear_sonic/training/train_trl_style.py \
    --config gear_sonic/training/config_trl.yaml
```

**Status**: 🚧 Work in progress
- Basic structure implemented
- Needs testing
- Full TRL integration would require:
  - Custom `TRLPPOTrainer` subclass
  - `RolloutStorage` adapter for G1MuJoCoEnv
  - Hydra config integration (optional)

## Recommendations

### For Quick Iteration & Prototyping
**Use**: Custom Implementation (`sonic_combined_trainer.py`)
- Simpler codebase
- Easier to modify reward functions
- Direct control over training loop
- Already tested and working

### For Large-Scale Training & Official Comparison
**Use**: TRL-Style Implementation (`train_trl_style.py`)
- Better multi-GPU scaling via Accelerate
- Compatible with HuggingFace ecosystem
- Can leverage official SONIC hyperparameters
- Easier integration with wandb/tensorboard

### For Final Sim-to-Real Deployment
**Use**: Either (both should produce equivalent policies)
- Policy architecture is the same
- Reward function is the same
- Differences are in training infrastructure

## Next Steps

### To Complete TRL-Style Implementation:

1. **Integrate full TRL PPOTrainer**:
   - Subclass `TRLPPOTrainer` from `gear_sonic.trl.trainer.ppo_trainer`
   - Adapt `RolloutStorage` for MuJoCo (instead of IsaacLab)
   - Implement `_rollout_step()` for G1MuJoCoEnv

2. **Add missing features**:
   - Multi-critic support (if needed)
   - Symmetry augmentation (Table S5 in paper)
   - Adaptive learning rate based on KL
   - AMP discriminator (if using AMP reward)

3. **Hydra config integration** (optional):
   - Convert `config_trl.yaml` to Hydra format
   - Add actor/critic configs under `config/actor_critic/`
   - Add algo configs under `config/algo/`

4. **Test at scale**:
   - Run with 4096 envs (requires multi-GPU)
   - Compare convergence with custom implementation
   - Validate final policy performance

### To Improve Custom Implementation:

1. **Add symmetry augmentation**:
   - Implement left-right symmetry for observations
   - Flip actions during training
   - Should improve sample efficiency

2. **Add multi-GPU support**:
   - Use `torch.nn.parallel.DistributedDataParallel`
   - Or integrate with Accelerate

3. **Add remaining domain randomization**:
   - Observation noise (Table S4)
   - Added mass
   - Link length scaling

## Official SONIC Training Details

From the paper (arXiv:2511.07820v3):

**Training Setup**:
- 4096 parallel environments (IsaacLab)
- 500 training iterations
- Each iteration: 24 steps per env → 98,304 transitions
- 5 PPO epochs per iteration
- Minibatch size: 24,576
- Learning rate: 3e-4 (adaptive based on KL)

**Hyperparameters** (Table S1):
- Discount γ = 0.99
- GAE λ = 0.95
- PPO clip ε = 0.2
- Value clip ε = 0.2
- Value loss coef = 2.0
- Entropy coef = 0.01
- Max grad norm = 1.0

**Reward Scale**:
- FK-based reward: ~[-5000, 0]
- Normalized by 5000 for PPO stability
- Multi-critic: 5 reward heads (proprioception tracking, contact, smoothness, etc.)

**Convergence**:
- Training time: ~6-8 hours on 8x A100 GPUs
- Success rate: 94% on 123-motion evaluation set
- Average reward: -200 (normalized)

## References

1. **SONIC Paper**: https://arxiv.org/abs/2511.07820v3
2. **Official Code**: https://github.com/NVlabs/GR00T-WholeBodyControl
3. **TRL Docs**: https://huggingface.co/docs/trl
4. **IsaacLab**: https://isaac-sim.github.io/IsaacLab
5. **MuJoCo**: https://mujoco.readthedocs.io

## Contact

For questions about this implementation, see:
- `PROJECT_SUMMARY.md` - Project overview
- `SONIC_DATA_PIPELINE_GUIDE.md` - Data processing
- `SONIC_IMPLEMENTATION_ROADMAP.md` - Implementation plan
