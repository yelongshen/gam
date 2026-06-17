# GEAR-SONIC Training Pipeline

This directory contains the training infrastructure for GEAR-SONIC action prediction models on egocentric teleoperation datasets.

## Overview

The training pipeline consists of:

1. **Data Loader** (`data_loader.py`): 
   - Loads egocentric teleoperation data from Parquet files
   - Handles observation/action normalization
   - Supports configurable context and action horizons
   - Automatic train/val split

2. **Models** (`model.py`):
   - `SonicActionPredictor`: Transformer-based action prediction model
   - `SonicMLP`: Simple MLP baseline for comparison

3. **Trainer** (`trainer.py`):
   - Full training loop with validation
   - Checkpoint saving and logging
   - TensorBoard integration for monitoring

## Quick Start

### 1. Install Dependencies

```bash
cd /home/grease/gam
source .venv_sim/bin/activate
pip install -e gear_sonic[sim]
pip install tensorboard
```

### 2. Configure Training

Edit `gear_sonic/training/config.yaml` to set:
- `data_root`: Path to your dataset
- `task_name`: Task folder name
- `model_type`: "transformer" or "mlp"
- Hyperparameters (batch size, learning rate, etc.)

### 3. Launch Training

```bash
python gear_sonic/training/train.py --config gear_sonic/training/config.yaml
```

**With overrides:**
```bash
python gear_sonic/training/train.py \
  --config gear_sonic/training/config.yaml \
  --batch-size 64 \
  --num-epochs 200 \
  --output-dir outputs/my_training_run
```

## Data Format

The training pipeline expects data in the following structure:

```
data_root/
├── unitreerobotics_datasets/
│   └── G1_Dex1_Fold_Towel/
│       └── data/
│           └── chunk-000/
│               ├── episode_000000.parquet
│               ├── episode_000001.parquet
│               └── ...
```

Each parquet file contains timesteps with:
- **Observations**: `observation.left_arm`, `observation.right_arm`, `observation.left_gripper`, etc.
- **Actions**: `action.left_arm`, `action.right_arm`, `action.left_gripper`, etc.
- **Metadata**: `timestamp`, `frame_index`, `episode_index`, etc.

## Configuration Options

### Data Settings
- `data_root`: Root directory containing datasets
- `task_name`: Task folder name (default: "G1_Dex1_Fold_Towel")
- `context_length`: Frames of observation context (default: 4)
- `action_horizon`: Frames to predict (default: 8)
- `train_split`: Train/validation split ratio (default: 0.8)
- `max_episodes`: Max episodes to load (default: all)

### Model Architecture
- `model_type`: "transformer" or "mlp" (default: "transformer")
- `hidden_dim`: Hidden layer dimension (default: 256)
- `num_layers`: Number of layers (default: 2)
- `num_heads`: Attention heads for transformer (default: 4)
- `dropout`: Dropout rate (default: 0.1)

### Training Hyperparameters
- `num_epochs`: Total epochs (default: 100)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Initial learning rate (default: 1e-3)
- `weight_decay`: L2 regularization (default: 1e-4)
- `num_workers`: Data loading workers (default: 4)

### Checkpointing
- `output_dir`: Save location for checkpoints (default: "outputs/sonic_training")
- `val_interval`: Validate every N epochs (default: 5)
- `save_interval`: Save checkpoint every N epochs (default: 10)

## Training Output

After training, you'll find:

```
outputs/sonic_training/
├── config.json                      # Training configuration
├── best_model.pt                    # Best checkpoint (lowest val loss)
├── checkpoint_epoch_010.pt          # Periodic checkpoints
├── checkpoint_epoch_020.pt
└── logs/                            # TensorBoard logs
    └── events.out.tfevents.*
```

## Monitoring Training

View training progress in TensorBoard:

```bash
tensorboard --logdir outputs/sonic_training/logs
```

Then open `http://localhost:6006` in your browser.

## Model Architectures

### Transformer Model (Default)
- Encodes observations with a linear layer
- Applies sinusoidal positional encoding
- Processes context with multi-head transformer
- Decodes final hidden state to action horizon

**Advantages:**
- Better for longer sequences
- Captures temporal dependencies
- Scales to larger context lengths

### MLP Baseline
- Flattens observation context
- Passes through 3+ layer MLP
- Outputs flattened action sequence

**Advantages:**
- Faster training
- Lower memory usage
- Good baseline for comparison

## Tips for Training

1. **Start with smaller dataset**: Use `max_episodes: 100` for quick experimentation
2. **Monitor validation loss**: Early stopping can prevent overfitting
3. **Adjust learning rate**: If loss oscillates, try 5e-4 or 1e-4
4. **Increase context length**: `context_length: 8` helps with temporal patterns
5. **Use mixed precision**: Consider `torch.cuda.amp` for faster training

## Troubleshooting

### Out of Memory
- Reduce `batch_size` (try 16)
- Reduce `hidden_dim` (try 128)
- Use "mlp" model instead of "transformer"

### Slow Training
- Increase `num_workers` (up to CPU count)
- Reduce `max_episodes` for testing
- Use GPU: training will use CUDA if available

### Poor Validation Loss
- Check data normalization in `data_loader.py`
- Increase `num_epochs` to 200-500
- Try lower learning rate (1e-4 or 5e-4)
- Verify observation/action dimensions are correct

## License

Same as parent GEAR-SONIC project (Apache 2.0)
