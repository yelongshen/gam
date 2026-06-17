# GEAR-SONIC Training Pipeline - Getting Started Guide

## ✅ Setup Complete

Your GEAR-SONIC training pipeline has been successfully set up! This document provides step-by-step instructions to train models on your egocentric dataset.

## 📁 What Was Created

The following training infrastructure has been added to `/home/grease/gam/gear_sonic/training/`:

```
training/
├── __init__.py                 # Module initialization
├── data_loader.py              # Egocentric dataset loader
├── model.py                    # Action prediction models
├── trainer.py                  # Training loop and utilities
├── train.py                    # Main training script
├── config.yaml                 # Default training config
├── config_test.yaml            # Quick test config
├── README.md                   # Detailed documentation
└── GETTING_STARTED.md          # This file
```

## 🚀 Quick Start

### 1. Check Training Status

Your test training is currently running! Check progress:

```bash
# View live logs
tail -f outputs/sonic_training_test.log

# Or check the log file
cat outputs/sonic_training_test.log
```

### 2. Configure Full Training

Edit `gear_sonic/training/config.yaml` for your full training:

```yaml
# Data paths
data_root: /home/grease/ego_dataset/work_bearlu/data
task_name: G1_Dex1_Fold_Towel

# Model and hyperparameters
model_type: transformer    # "transformer" or "mlp"
num_epochs: 200
batch_size: 32
learning_rate: 1.0e-3

# Use all episodes (remove or set to null)
max_episodes: null
```

### 3. Launch Full Training

```bash
cd /home/grease/gam
source .venv_sim/bin/activate

# Standard training
python gear_sonic/training/train.py --config gear_sonic/training/config.yaml

# Or with command-line overrides
python gear_sonic/training/train.py \
  --config gear_sonic/training/config.yaml \
  --batch-size 64 \
  --num-epochs 300 \
  --output-dir outputs/my_sonic_model
```

## 📊 Monitoring Progress

### TensorBoard

View training metrics in real-time:

```bash
# In a separate terminal
cd /home/grease/gam
source .venv_sim/bin/activate
tensorboard --logdir outputs/sonic_training_test/logs
```

Then open `http://localhost:6006` in your browser.

### Log Files

Check training progress:

```bash
# Real-time
tail -f outputs/sonic_training_test.log

# Count completed epochs
grep "Validation Metrics" outputs/sonic_training_test.log | wc -l

# View best loss
grep "Saved best model" outputs/sonic_training_test.log | tail -1
```

## 🏗️ Architecture Overview

### Data Pipeline

```
Parquet Files (200 episodes)
    ↓
EgocentricDataset
  - Loads observations & actions
  - Applies normalization
  - Creates (context, targets) pairs
    ↓
PyTorch DataLoader
  - Batching
  - Multi-worker loading
  - Train/val split
```

### Model Architectures

**Transformer Model** (Recommended):
- Encodes 4 frames of observations
- Multi-head self-attention
- Predicts 8 future actions
- ~200K parameters
- Better for temporal patterns

**MLP Model** (Baseline):
- Flattens observation context
- 3-layer feedforward network
- Predicts 8 future actions
- ~200K parameters
- Faster training/inference

### Training Loop

```
For each epoch:
  1. Train on all batches
     - Forward pass
     - Compute MSE loss
     - Backward pass
     - Update weights
  
  2. Every 5 epochs: Validate
     - Evaluate on val set
     - Save best checkpoint
     - Log metrics to TensorBoard
  
  3. Every 10 epochs: Save checkpoint
```

## 📈 Expected Performance

Based on the test run, you should see:

- **Epoch 0**: Loss drops from ~0.94 to ~0.09
- **Epoch 5**: Validation loss ~0.06-0.07
- **Epoch 10**: Validation loss ~0.05-0.06
- **Epoch 50+**: Continued improvement, converges

**Best model location**: `outputs/sonic_training_test/best_model.pt`

## 🔧 Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `context_length` | 4 | More = longer temporal context but slower training |
| `action_horizon` | 8 | Frames to predict (8 = ~0.4 sec at 20Hz) |
| `hidden_dim` | 256 | Larger = more capacity, slower training |
| `batch_size` | 32 | Larger = better gradient estimates, more memory |
| `learning_rate` | 1e-3 | Start here, reduce if loss oscillates |
| `num_epochs` | 100 | 50-200 typical; stop early if val loss plateaus |

## 💡 Tips & Tricks

### For Better Results
1. **Increase context length**: `context_length: 8` captures more history
2. **Longer training**: Set `num_epochs: 300-500` for full convergence
3. **Use transformer**: Usually outperforms MLP after 50+ epochs
4. **Monitor validation loss**: Stop if it plateaus for 20+ epochs

### If Training is Slow
1. Reduce `num_workers: 2` (less I/O overhead)
2. Reduce `context_length: 2` (smaller inputs)
3. Use MLP model (faster than transformer)
4. Increase `batch_size: 64` (fewer gradient updates)

### If Out of Memory
1. Reduce `batch_size: 16`
2. Reduce `hidden_dim: 128`
3. Use MLP instead of transformer
4. Reduce `max_episodes: 100` for testing

## 📂 Output Structure

After training, check:

```
outputs/sonic_training_test/
├── config.json              # Training config saved
├── best_model.pt            # Best checkpoint (lowest val loss)
├── checkpoint_epoch_010.pt  # Periodic checkpoints
├── checkpoint_epoch_020.pt
├── logs/
│   └── events.out.tfevents.* # TensorBoard logs
└── sonic_training_test.log  # Training output (if background)
```

## 🔌 Using Trained Models

### Load a Checkpoint

```python
import torch
from gear_sonic.training import SonicMLP, SonicActionPredictor

# Load model
model = SonicMLP(obs_dim=57, action_dim=35)
checkpoint = torch.load("outputs/sonic_training_test/best_model.pt")
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Predict actions
with torch.no_grad():
    obs_context = torch.randn(1, 4, 57)  # (batch, context_len, obs_dim)
    actions = model(obs_context)  # (batch, action_horizon, action_dim)
```

### Export for Deployment

To use trained models in deployment:

1. **ONNX Export** (for C++ deployment):
   ```python
   torch.onnx.export(model, dummy_input, "policy_decoder.onnx")
   ```

2. **TorchScript** (for Python deployment):
   ```python
   scripted = torch.jit.script(model)
   scripted.save("policy.pt")
   ```

## 🐛 Troubleshooting

### Training won't start
```bash
# Check data directory exists
ls /home/grease/ego_dataset/work_bearlu/data/unitreerobotics_datasets/G1_Dex1_Fold_Towel/data/chunk-000/ | head

# Verify CUDA available
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of memory error
```bash
# Reduce batch size in config
batch_size: 16

# Or reduce model size
hidden_dim: 128
```

### Loss not decreasing
```bash
# Try lower learning rate
learning_rate: 5.0e-4

# Check data normalization
python -c "from gear_sonic.training import EgocentricDataset; d = EgocentricDataset(...); print(d.obs_mean)"
```

## 📚 Next Steps

1. **Let test training finish** (~1-2 hours)
2. **Review results**: Check `outputs/sonic_training_test/`
3. **Train full model**: Update config and run full training (8-24 hours)
4. **Evaluate performance**: Use `best_model.pt` on validation set
5. **Deploy**: Export model for real robot control

## 📖 Reference

For detailed documentation, see:
- `gear_sonic/training/README.md` - Full architecture details
- `gear_sonic/training/config.yaml` - All configuration options
- `gear_sonic/training/train.py` - Command-line arguments

## ✉️ Questions?

Check the training logs for errors:
```bash
grep "ERROR\|Traceback" outputs/sonic_training_test.log
```

Or review specific sections of `gear_sonic/training/trainer.py` for the training loop logic.

---

**Status**: ✅ Training pipeline ready
**Data**: 200 episodes × ~1432 frames = ~286K trajectory samples
**Model**: MLP (fast test) / Transformer (production)
**GPU**: CUDA available for acceleration
