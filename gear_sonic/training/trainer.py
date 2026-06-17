"""Trainer for GEAR-SONIC models."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data_loader import EgoDataLoader
from .model import SonicActionPredictor, SonicMLP


logger = logging.getLogger(__name__)


class SonicTrainer:
    """Trainer for GEAR-SONIC action prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str = "outputs/sonic_training",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_tensorboard: bool = True,
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: torch.device (cuda or cpu)
            output_dir: Directory for checkpoints and logs
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            use_tensorboard: Whether to log to tensorboard
        """
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Tensorboard
        self.use_tensorboard = use_tensorboard
        self.writer = None
        if use_tensorboard:
            log_dir = self.output_dir / "logs"
            self.writer = SummaryWriter(str(log_dir))
        
        self.global_step = 0
        self.epoch = 0
    
    def _get_scheduler(self, num_epochs: int, steps_per_epoch: int):
        """Create learning rate scheduler."""
        total_steps = num_epochs * steps_per_epoch
        return CosineAnnealingLR(self.optimizer, T_max=total_steps)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.
        
        Returns:
            Average loss over the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (obs_context, action_targets) in enumerate(train_loader):
            obs_context = obs_context.to(self.device)
            action_targets = action_targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            action_preds = self.model(obs_context)
            loss = self.criterion(action_preds, action_targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Tensorboard logging
            if self.use_tensorboard and batch_idx % 10 == 0:
                self.writer.add_scalar(
                    "train/loss",
                    loss.item(),
                    self.global_step,
                )
            
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        for obs_context, action_targets in val_loader:
            obs_context = obs_context.to(self.device)
            action_targets = action_targets.to(self.device)
            
            # Forward pass
            action_preds = self.model(obs_context)
            loss = self.criterion(action_preds, action_targets)
            mae = torch.abs(action_preds - action_targets).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return {
            "val_loss": avg_loss,
            "val_mae": avg_mae,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        val_interval: int = 5,
        save_interval: int = 10,
    ):
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            val_interval: Validation frequency (epochs)
            save_interval: Checkpoint save frequency (epochs)
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model.__class__.__name__}")
        
        # Create scheduler
        scheduler = self._get_scheduler(num_epochs, len(train_loader))
        
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}")
            
            if self.use_tensorboard:
                self.writer.add_scalar("train/epoch_loss", train_loss, epoch)
            
            # Validation
            if (epoch + 1) % val_interval == 0:
                val_metrics = self.validate(val_loader)
                logger.info(f"Validation Metrics: {val_metrics}")
                
                if self.use_tensorboard:
                    for key, val in val_metrics.items():
                        self.writer.add_scalar(f"val/{key}", val, epoch)
                
                # Save best checkpoint
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    self._save_checkpoint("best_model.pt")
                    logger.info(f"Saved best model with val_loss: {best_val_loss:.6f}")
            
            # Periodic checkpoint
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1:03d}.pt")
            
            # Update learning rate
            scheduler.step()
        
        logger.info("Training finished!")
        self.writer.close()
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save(
            {
                "epoch": self.epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_config(self, config: Dict):
        """Save training configuration."""
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved config to {config_path}")


def train_sonic_model(config_path: str):
    """Train a SONIC model from config file.
    
    Args:
        config_path: Path to training config YAML/JSON
    """
    import yaml
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Config: {config}")
    
    # Setup
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
    
    # Get dimensions from dataloader
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
    
    logger.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
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
    logging.basicConfig(level=logging.INFO)
    import sys
    
    if len(sys.argv) > 1:
        train_sonic_model(sys.argv[1])
    else:
        print("Usage: python trainer.py <config_path>")
