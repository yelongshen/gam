#!/usr/bin/env python3
"""Training script for GEAR-SONIC action prediction models.

Usage:
    python train.py --config gear_sonic/training/config.yaml
    python train.py --config gear_sonic/training/config.yaml --model-type mlp
    python train.py --config gear_sonic/training/config.yaml --batch-size 64
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add repo to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from gear_sonic.training.data_loader import EgoDataLoader
from gear_sonic.training.model import SonicActionPredictor, SonicMLP
from gear_sonic.training.trainer import SonicTrainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train GEAR-SONIC action prediction models on egocentric data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="gear_sonic/training/config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["transformer", "mlp"],
        help="Model architecture (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Max episodes to load (overrides config)",
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override with CLI args
    if args.model_type:
        config["model_type"] = args.model_type
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.num_epochs:
        config["num_epochs"] = args.num_epochs
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_episodes:
        config["max_episodes"] = args.max_episodes
    
    logger.info(f"Config: {config}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = EgoDataLoader.create(
        data_root=config["data_root"],
        task_name=config.get("task_name", "G1_Dex1_Fold_Towel"),
        context_length=config.get("context_length", 4),
        action_horizon=config.get("action_horizon", 8),
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 4),
        train_split=config.get("train_split", 0.8),
        max_episodes=config.get("max_episodes"),
    )
    logger.info(
        f"Train: {len(train_loader)} batches, "
        f"Val: {len(val_loader)} batches"
    )
    
    # Get dimensions
    sample_obs, sample_action = next(iter(train_loader))
    obs_dim = sample_obs.shape[-1]
    action_dim = sample_action.shape[-1]
    logger.info(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create model
    model_type = config.get("model_type", "transformer")
    if model_type == "transformer":
        model = SonicActionPredictor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            context_length=config.get("context_length", 4),
            action_horizon=config.get("action_horizon", 8),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
            num_heads=config.get("num_heads", 4),
            dropout=config.get("dropout", 0.1),
        )
    elif model_type == "mlp":
        model = SonicMLP(
            obs_dim=obs_dim,
            action_dim=action_dim,
            context_length=config.get("context_length", 4),
            action_horizon=config.get("action_horizon", 8),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model.__class__.__name__} with {num_params:,} parameters")
    
    # Create trainer
    trainer = SonicTrainer(
        model=model,
        device=device,
        output_dir=config.get("output_dir", "outputs/sonic_training"),
        learning_rate=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    
    # Save config
    trainer.save_config(config)
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get("num_epochs", 100),
        val_interval=config.get("val_interval", 5),
        save_interval=config.get("save_interval", 10),
    )


if __name__ == "__main__":
    main()
