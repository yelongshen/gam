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
✅ Ready for production training

---

**Status**: Ready for training ✅  
**Data**: 496GB egocentric teleoperation  
**Compute**: GPU accelerated (CUDA available)  
**Timeline**: 50 hours for 200 epoch run  

See [TRAINING_PIPELINE_SUMMARY.md](TRAINING_PIPELINE_SUMMARY.md) for full details.
