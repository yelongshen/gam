#!/bin/bash
# Verify GEAR-SONIC training pipeline setup

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  GEAR-SONIC Training Pipeline Verification                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check training module files
echo "✓ Checking training module files..."
files=(
    "gear_sonic/training/__init__.py"
    "gear_sonic/training/data_loader.py"
    "gear_sonic/training/model.py"
    "gear_sonic/training/trainer.py"
    "gear_sonic/training/train.py"
    "gear_sonic/training/config.yaml"
    "gear_sonic/training/config_test.yaml"
    "gear_sonic/training/README.md"
    "gear_sonic/training/GETTING_STARTED.md"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ MISSING: $file"
        all_exist=false
    fi
done

echo ""
echo "✓ Checking documentation files..."
docs=(
    "TRAINING_README.md"
    "TRAINING_PIPELINE_SUMMARY.md"
    "TRAINING_COMMANDS.sh"
)

for file in "${docs[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ MISSING: $file"
        all_exist=false
    fi
done

echo ""
echo "✓ Checking data directory..."
data_dir="/home/grease/ego_dataset/work_bearlu/data/unitreerobotics_datasets/G1_Dex1_Fold_Towel/data/chunk-000"
if [ -d "$data_dir" ]; then
    count=$(find "$data_dir" -name "episode_*.parquet" | wc -l)
    echo "  ✓ Found $count episode parquet files"
else
    echo "  ✗ Data directory not found: $data_dir"
    all_exist=false
fi

echo ""
echo "✓ Checking Python dependencies..."
python3 -c "
import torch
import pandas as pd
import numpy as np
from pathlib import Path

print('  ✓ PyTorch:', torch.__version__)
print('  ✓ Pandas:', pd.__version__)
print('  ✓ NumPy:', np.__version__)
print('  ✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('    - GPU:', torch.cuda.get_device_name(0))
" 2>&1 | sed 's/^/  /'

echo ""
echo "✓ Checking training module imports..."
python3 -c "
from gear_sonic.training import EgocentricDataset, EgoDataLoader, SonicActionPredictor, SonicMLP, SonicTrainer
print('  ✓ All training modules import successfully')
" 2>&1 | sed 's/^/  /'

echo ""
echo "✓ Checking test dataset loading..."
python3 -c "
from gear_sonic.training import EgocentricDataset
try:
    dataset = EgocentricDataset(
        data_root='/home/grease/ego_dataset/work_bearlu/data',
        task_name='G1_Dex1_Fold_Towel',
        max_episodes=2
    )
    obs, action = dataset[0]
    print(f'  ✓ Dataset loads successfully')
    print(f'    - Samples: {len(dataset)}')
    print(f'    - Obs shape: {obs.shape}')
    print(f'    - Action shape: {action.shape}')
except Exception as e:
    print(f'  ✗ Error loading dataset: {e}')
" 2>&1 | sed 's/^/  /'

echo ""
echo "✓ Checking model creation..."
python3 -c "
import torch
from gear_sonic.training import SonicMLP, SonicActionPredictor

mlp = SonicMLP(obs_dim=57, action_dim=35)
transformer = SonicActionPredictor(obs_dim=57, action_dim=35)

mlp_params = sum(p.numel() for p in mlp.parameters())
trans_params = sum(p.numel() for p in transformer.parameters())

print(f'  ✓ SonicMLP: {mlp_params:,} parameters')
print(f'  ✓ SonicActionPredictor: {trans_params:,} parameters')

# Test forward pass
obs = torch.randn(1, 4, 57)
actions_mlp = mlp(obs)
actions_trans = transformer(obs)
print(f'  ✓ Forward pass successful')
print(f'    - Output shape: {actions_mlp.shape}')
" 2>&1 | sed 's/^/  /'

echo ""
if [ "$all_exist" = true ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "✅ All checks passed! Training pipeline is ready."
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "  1. Monitor test training: tail -f outputs/sonic_training_test.log"
    echo "  2. Start full training: python gear_sonic/training/train.py --config gear_sonic/training/config.yaml"
    echo "  3. View documentation: cat TRAINING_README.md"
    echo ""
else
    echo "═══════════════════════════════════════════════════════════════"
    echo "⚠️  Some files are missing. Please check the errors above."
    echo "═══════════════════════════════════════════════════════════════"
fi
