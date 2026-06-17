#!/bin/bash
# GEAR-SONIC Training Quick Reference
# Common commands for training Sonic models on egocentric data

# ============================================================================
# SETUP
# ============================================================================

# Activate environment
source /home/grease/gam/.venv_sim/bin/activate

# Navigate to repo
cd /home/grease/gam

# ============================================================================
# CHECK STATUS
# ============================================================================

# View current training progress (test run)
tail -f outputs/sonic_training_test.log

# Get epoch count
grep "Epoch" outputs/sonic_training_test.log | tail -5

# Check validation losses
grep "Validation Metrics" outputs/sonic_training_test.log

# View TensorBoard
tensorboard --logdir outputs/sonic_training_test/logs

# ============================================================================
# START TRAINING
# ============================================================================

# Test run (50 episodes, 50 epochs) - CURRENTLY RUNNING
python gear_sonic/training/train.py --config gear_sonic/training/config_test.yaml

# Full training (all episodes, 200 epochs) - RECOMMENDED
python gear_sonic/training/train.py --config gear_sonic/training/config.yaml

# Custom batch size
python gear_sonic/training/train.py \
  --config gear_sonic/training/config.yaml \
  --batch-size 64

# Custom learning rate
python gear_sonic/training/train.py \
  --config gear_sonic/training/config.yaml \
  --learning-rate 5e-4

# Multiple overrides
python gear_sonic/training/train.py \
  --config gear_sonic/training/config.yaml \
  --batch-size 64 \
  --num-epochs 300 \
  --learning-rate 5e-4 \
  --output-dir outputs/sonic_v1

# Run in background (so you can close terminal)
nohup python gear_sonic/training/train.py \
  --config gear_sonic/training/config.yaml \
  > outputs/training.log 2>&1 &

# ============================================================================
# USE TRAINED MODEL
# ============================================================================

# Python script to load and use model
python << 'EOF'
import torch
from gear_sonic.training import SonicMLP

# Load best model
model = SonicMLP(obs_dim=57, action_dim=35)
ckpt = torch.load("outputs/sonic_training_test/best_model.pt")
model.load_state_dict(ckpt["model_state"])
model.eval()

# Run inference
with torch.no_grad():
    obs = torch.randn(1, 4, 57)  # batch=1, context=4 frames
    actions = model(obs)           # predict 8 actions
    print(f"Input: {obs.shape}, Output: {actions.shape}")
EOF

# ============================================================================
# MONITORING
# ============================================================================

# GPU status (if available)
nvidia-smi -l 1

# CPU status
top -b -n 1 | head -20

# Disk usage
du -sh outputs/sonic_training_test/

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Check data loading works
python -c "
from gear_sonic.training import EgocentricDataset
d = EgocentricDataset(
    data_root='/home/grease/ego_dataset/work_bearlu/data',
    task_name='G1_Dex1_Fold_Towel',
    max_episodes=2
)
print(f'Dataset size: {len(d)}')
obs, action = d[0]
print(f'Obs shape: {obs.shape}, Action shape: {action.shape}')
"

# Check model creation
python -c "
from gear_sonic.training import SonicMLP
model = SonicMLP(obs_dim=57, action_dim=35)
params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {params:,}')
"

# Check CUDA available
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ============================================================================
# CLEANUP
# ============================================================================

# Remove test training outputs
rm -rf outputs/sonic_training_test/

# Remove old checkpoints (keep best_model.pt)
cd outputs/sonic_training
rm -f checkpoint_epoch_*.pt
cd -

# ============================================================================
# REFERENCES
# ============================================================================

# Full documentation
cat gear_sonic/training/README.md

# Getting started guide  
cat gear_sonic/training/GETTING_STARTED.md

# Config options
cat gear_sonic/training/config.yaml

# ============================================================================
# EXAMPLES
# ============================================================================

# Train transformer model for 500 epochs
python gear_sonic/training/train.py \
  --config gear_sonic/training/config.yaml \
  --model-type transformer \
  --num-epochs 500 \
  --output-dir outputs/sonic_transformer_500ep

# Train with different hyperparameters
python gear_sonic/training/train.py \
  --config gear_sonic/training/config.yaml \
  --batch-size 128 \
  --learning-rate 1e-4 \
  --num-epochs 300 \
  --output-dir outputs/sonic_tuned

# Train on subset of data for quick testing
python gear_sonic/training/train.py \
  --config gear_sonic/training/config.yaml \
  --max-episodes 20 \
  --num-epochs 10 \
  --output-dir outputs/sonic_quick_test
