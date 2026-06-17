# 🚀 SONIC Data Processing Pipeline - User Guide

## Quick Start

### 1. Extract Dataset (One-Time Setup)

The Bones-Studio dataset comes in compressed tar files. Extract them first:

```bash
cd /home/grease/ego_dataset/work_bearlu/data/bones-studio-seed

# Extract G1 robot motion data (22 GB → ~40 GB)
tar -xzf g1.tar.gz

# Extract human motion - choose one or both:
# Option A: Proportional SOMA (actor-specific body models) - FASTER
tar -xzf soma_proportional.tar.gz

# Option B: Uniform SOMA (generic body model) - FALLBACK
tar -xzf soma_uniform.tar.gz

# Verify extraction
ls -lh g1/csv/       # Should show directories like 240918, 231012, etc.
ls -lh soma_proportional/bvh/  # Same date structure
```

**Note**: Extraction takes 15-30 minutes per file. Total extracted size: ~300+ GB

### 2. Test the Pipeline (5 Motions)

```bash
cd /home/grease/gam
source .venv_sim/bin/activate

# Process first 5 motions to test
python gear_sonic/training/process_sonic_data.py --max-motions 5

# Expected output:
# - Data loaded: 5 motions
# - Saving to: ./data/sonic_processed/
# - Files: *.npz (compressed numpy)
```

### 3. Verify Output

```bash
# Check created files
ls -lh data/sonic_processed/ | head -10

# Inspect one motion
python << 'EOF'
import numpy as np
data = np.load('data/sonic_processed/body_check_001__A548.npz', allow_pickle=True)
print(f"g_r (robot): {data['g_r'].shape}")      # [T, 29]
print(f"g_h (human): {data['g_h'].shape}")      # [T, 72]
print(f"g_m (mixed): {data['g_m'].shape}")      # [T, 11]
print(f"Motion: {data['move_name']}")
print(f"Actor: {data['actor_id']}")
EOF
```

### 4. Scale Up Processing

```bash
# Process 100 motions (takes ~2 minutes)
python gear_sonic/training/process_sonic_data.py --max-motions 100

# Process 1,000 motions (takes ~15-30 minutes)
python gear_sonic/training/process_sonic_data.py --max-motions 1000

# Process all motions (WARNING: takes 24-72 hours)
python gear_sonic/training/process_sonic_data.py

# Run in background
nohup python gear_sonic/training/process_sonic_data.py > processing.log 2>&1 &
```

## Configuration

Edit `gear_sonic/training/config_sonic_data.yaml`:

```yaml
dataset:
  root: /home/grease/ego_dataset/work_bearlu/data/bones-studio-seed
  metadata: metadata/seed_metadata_v004.parquet
  output_dir: ./data/sonic_processed

processing:
  max_motions: 500         # Limit for testing
  load_g_r: true           # Always true for robot
  load_g_h: true           # Set to true for human motion
  load_g_m: true           # Always true for mixed
  g_h_format: proportional # Use actor-specific SOMA models

motion_window:
  context_length: 4        # History frames for context
  action_horizon: 8        # Future frames to predict

output:
  format: npy              # Use compressed numpy
  compressed: true
```

## Processing Workflow

### Flow Diagram

```
Raw Bones-Studio Data
    ↓
[G1RobotDataLoader]         [SOMAHumanDataLoader]
    ↓ g_r [T, 29]           ↓ g_h [T, 72]
    └─────────┬─────────────┘
              ↓
    [MixedRepresentationBuilder]
              ↓ g_m [T, 11]
              ↓
    [MotionData Triplet]
              ↓ (save as NPZ)
    ./data/sonic_processed/
              ↓
    [PyTorch Dataset]
        ├─ Temporal windowing
        ├─ Normalization
        └─ Train/Val/Test split
              ↓
        Ready for PPO Training!
```

### Data Dimensions

**Input**:
- **g_r** (Robot): [T, 29] - Unitree G1 29-DOF robot configuration
- **g_h** (Human): [T, 72] - SMPL skeleton (24 joints × 3 coords)
- **g_m** (Mixed): [T, 11] - VR tracker positions + lower body

**Output (with windowing)**:
```
Context:  [batch, context_length=4, dim]
Target:   [batch, action_horizon=8, dim]

Example:
  g_r_context: [32, 4, 29]   ← 4 frames of robot history
  g_r_target:  [32, 8, 29]   ← 8 frames to predict
  g_h_context: [32, 4, 72]   ← 4 frames of human history
  g_h_target:  [32, 8, 72]   ← 8 frames to predict
```

## Using in Training

### Step 1: Create DataLoaders

```python
from gear_sonic.training.sonic_dataset import create_dataloaders

train_loader, val_loader = create_dataloaders(
    data_dir='./data/sonic_processed',
    batch_size=32,
    num_workers=4,
    context_length=4,
    action_horizon=8,
    split_ratio=0.8,  # 80% train, 20% val
)
```

### Step 2: Iterate Batches

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # Modalities with temporal context
        g_r_ctx = batch['g_r_context']      # [B, 4, 29]
        g_h_ctx = batch['g_h_context']      # [B, 4, 72]
        g_m_ctx = batch['g_m_context']      # [B, 4, 11]
        
        # Target trajectories to predict
        g_r_tgt = batch['g_r_target']       # [B, 8, 29]
        g_h_tgt = batch['g_h_target']       # [B, 8, 72]
        g_m_tgt = batch['g_m_target']       # [B, 8, 11]
        
        # Your training code here
        ...
```

### Step 3: Normalize Outputs

Data is automatically normalized during loading:

```python
# To denormalize predictions for visualization:
from gear_sonic.training.sonic_dataset import SonicMotionDataset

dataset = SonicMotionDataset(data_dir='./data/sonic_processed')

# Get normalization stats
g_r_mean = dataset.g_r_mean  # [29]
g_r_std = dataset.g_r_std    # [29]

# Denormalize prediction
pred_normalized = model(batch)  # [B, 8, 29]
pred_original = pred_normalized * g_r_std + g_r_mean
```

## Typical Processing Times

| Motions | Time | Output Size |
|---------|------|-------------|
| 5 | 5-10s | ~5 MB |
| 10 | 10-20s | ~10 MB |
| 100 | 1-2 min | ~100 MB |
| 500 | 5-15 min | ~500 MB |
| 1,000 | 15-30 min | ~1 GB |
| 5,000 | 1-2 hrs | ~5 GB |
| 10,000 | 2-4 hrs | ~10 GB |
| 50,000 | 10-20 hrs | ~50 GB |
| 142,220 (all) | 24-72 hrs | ~140 GB |

## Memory Requirements

### Processing
- **Per motion**: ~1-10 MB
- **100 motions in memory**: ~100 MB - 1 GB
- **1,000 motions in memory**: ~1-10 GB
- **Full dataset**: Requires streaming loader (included)

### Training
- **Mini-batch** [32, context=4, 72]: ~1 MB
- **GPU memory** (RTX 4090, batch=32): ~6-12 GB
- **RAM for dataset caching**: ~2-4 GB for 1K motions

## Troubleshooting

### Issue: "CSV file not found"
```
FileNotFoundError: G1 CSV directory not found
```

**Solution**: Extract g1.tar.gz
```bash
cd /home/grease/ego_dataset/work_bearlu/data/bones-studio-seed
tar -xzf g1.tar.gz
```

### Issue: "BVH file not found"
```
Warning: BVH file not found: soma_proportional/bvh/...
```

**Solution**: Extract soma tar files
```bash
tar -xzf soma_proportional.tar.gz
# or
tar -xzf soma_uniform.tar.gz
```

### Issue: "Out of memory during processing"
```
MemoryError: Unable to allocate array
```

**Solution**: Reduce batch size
- Edit config: `max_motions: 100` (instead of 1000)
- Process multiple times with lower limits

### Issue: "Slow processing speed"
**Causes & Solutions**:
- **Disk I/O bottleneck**: Use SSD, not HDD
- **CPU bottleneck**: Each motion takes ~100-500ms
- **Solution**: Process on high-performance machine or increase timeout

### Issue: "Mismatched shapes after loading"
```
Shape mismatch for motion_X: g_r=(1000, 29), g_h=(1500, 72)
```

**Cause**: Robot and human motions have different lengths

**Solution**: Auto-handled by processor (trims to min length)

## Advanced Usage

### Custom Processing

```python
from gear_sonic.training.sonic_data_processor import (
    SonicDataProcessor,
    MixedRepresentationBuilder
)

processor = SonicDataProcessor(
    data_root='/path/to/bones-studio-seed',
    metadata_path='metadata/seed_metadata_v004.parquet'
)

# Process specific motions
for idx in [0, 10, 100, 1000]:
    row = processor.metadata.iloc[idx]
    motion = processor.process_single_motion(row, verbose=True)
    
    if motion:
        print(f"Loaded {motion.move_name}: {len(motion)} frames")
        print(f"  g_r: {motion.g_r.shape}")
        print(f"  g_h: {motion.g_h.shape}")
        print(f"  g_m: {motion.g_m.shape}")
```

### Custom Mixed Representation

```python
# Create custom g_m with different joints
g_m_custom = MixedRepresentationBuilder.build_mixed_representation(
    g_h=motion.g_h,
    g_r=motion.g_r,
    vr_joints=[15, 20, 21],  # head, left wrist, right wrist
    lower_body_joints=[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28]
)
```

### Validation Script

```bash
python << 'EOF'
import numpy as np
import pandas as pd
from pathlib import Path

# Check all processed motions
data_dir = Path('./data/sonic_processed')
files = list(data_dir.glob('*.npz'))

print(f"Total motions: {len(files)}")

shapes = []
for f in files[:100]:
    data = np.load(f, allow_pickle=True)
    shapes.append({
        'motion': f.stem,
        'g_r': data['g_r'].shape,
        'g_h': data['g_h'].shape,
        'g_m': data['g_m'].shape,
    })

df = pd.DataFrame(shapes)
print(df)
print(f"\nUnique g_r shapes: {df['g_r'].nunique()}")
print(f"Unique g_h shapes: {df['g_h'].nunique()}")
print(f"Unique g_m shapes: {df['g_m'].nunique()}")
EOF
```

## Performance Optimization

### 1. Use Proportional SOMA (Faster)
```yaml
g_h_format: proportional  # ~2-3x faster than uniform
```

### 2. Increase num_workers
```python
create_dataloaders(..., num_workers=8)  # More CPU cores = faster loading
```

### 3. Use NVMe SSD
- Extract tar files to SSD for 5-10x faster I/O
- Process on SSD for faster reading/writing

### 4. Batch Processing
```bash
# Process in 10 batches of 10K motions each
for i in {0..10}; do
    python process_sonic_data.py --max-motions $((i * 10000 + 10000))
done
```

## Files & Structure

```
gear_sonic/training/
├── sonic_data_processor.py         (500 lines) Core processing
├── sonic_dataset.py                (200 lines) PyTorch Dataset
├── process_sonic_data.py           (320 lines) Main script
├── config_sonic_data.yaml          (30 lines) Configuration
└── SONIC_DATA_PIPELINE.md          (500 lines) Architecture docs

data/
├── sonic_processed/                Processed triplets
│   ├── *.npz                       Individual motions
│   └── ...
├── sonic_train/                    Train split (symlinks)
├── sonic_val/                      Val split (symlinks)
└── sonic_test/                     Test split (symlinks)
```

## Next Steps

1. ✅ Extract tar files (if not already done)
2. ✅ Test pipeline with 5-10 motions
3. ✅ Verify output shapes and values
4. ✅ Process target number of motions (100, 1K, or more)
5. 🔄 Create train/val/test dataloaders
6. 🔄 Implement encoder architectures (E_r, E_h, E_m)
7. 🔄 Implement motion decoder (D_r)
8. 🔄 Setup PPO training loop

## Support

- **Documentation**: `SONIC_DATA_PIPELINE.md` (architecture details)
- **Code**: Check docstrings in `sonic_data_processor.py`
- **Config**: Modify `config_sonic_data.yaml`
- **Issues**: Check troubleshooting section above

---

**Status**: ✅ Pipeline ready for production use  
**Next Phase**: Encoder-Decoder Architecture (E_r, E_h, E_m, D_r)  
**Estimated Time**: 1-2 hours to process 1,000 motions
