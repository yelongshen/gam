"""PyTorch Dataset classes for SONIC multi-modal training."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class SonicMotionDataset(Dataset):
    """PyTorch Dataset for SONIC multi-modal motion training.
    
    Loads triplets of (g_r, g_h, g_m) with context and action horizons.
    """
    
    def __init__(
        self,
        data_dir: str,
        context_length: int = 4,
        action_horizon: int = 8,
        normalize: bool = True,
        split: str = 'train',
        split_ratio: float = 0.8,
    ):
        """Initialize SONIC dataset.
        
        Args:
            data_dir: Directory containing .npz motion files
            context_length: Number of past frames for context
            action_horizon: Number of future frames to predict
            normalize: Whether to normalize data
            split: 'train', 'val', or 'all'
            split_ratio: Train/val split (default 0.8/0.2)
        """
        self.data_dir = Path(data_dir)
        self.context_length = context_length
        self.action_horizon = action_horizon
        self.normalize = normalize
        self.split = split
        self.split_ratio = split_ratio
        
        # Find all motion files
        self.motion_files = sorted(self.data_dir.glob('*.npz'))
        logger.info(f"Found {len(self.motion_files)} motion files")
        
        # Split into train/val
        split_idx = int(len(self.motion_files) * split_ratio)
        if split == 'train':
            self.motion_files = self.motion_files[:split_idx]
        elif split == 'val':
            self.motion_files = self.motion_files[split_idx:]
        # else 'all' uses all files
        
        # Load all motions into memory
        self.motions = {}
        self.sample_indices = []
        self._load_all_motions()
        
        # Compute normalization statistics
        if self.normalize:
            self._compute_stats()
    
    def _load_all_motions(self):
        """Load all motion files into memory."""
        logger.info(f"Loading {len(self.motion_files)} motion files...")
        
        for motion_file in self.motion_files:
            try:
                data = np.load(motion_file, allow_pickle=True)
                
                motion_name = motion_file.stem
                self.motions[motion_name] = {
                    'g_r': data['g_r'].astype(np.float32),
                    'g_h': data['g_h'].astype(np.float32),
                    'g_m': data['g_m'].astype(np.float32),
                }
                
                # Record valid sample indices for this motion
                T = len(data['g_r'])
                valid_frames = max(0, T - self.context_length - self.action_horizon)
                
                for frame_idx in range(valid_frames):
                    self.sample_indices.append((motion_name, frame_idx))
                
            except Exception as e:
                logger.warning(f"Error loading {motion_file}: {e}")
        
        logger.info(f"Loaded {len(self.motions)} motions with {len(self.sample_indices)} samples")
    
    def _compute_stats(self):
        """Compute normalization statistics."""
        logger.info("Computing normalization statistics...")
        
        all_g_r = []
        all_g_h = []
        all_g_m = []
        
        for motion in self.motions.values():
            all_g_r.append(motion['g_r'])
            all_g_h.append(motion['g_h'])
            all_g_m.append(motion['g_m'])
        
        all_g_r = np.concatenate(all_g_r, axis=0)
        all_g_h = np.concatenate(all_g_h, axis=0)
        all_g_m = np.concatenate(all_g_m, axis=0)
        
        # Compute statistics
        self.g_r_mean = all_g_r.mean(axis=0)
        self.g_r_std = all_g_r.std(axis=0) + 1e-8
        
        self.g_h_mean = all_g_h.mean(axis=0)
        self.g_h_std = all_g_h.std(axis=0) + 1e-8
        
        self.g_m_mean = all_g_m.mean(axis=0)
        self.g_m_std = all_g_m.std(axis=0) + 1e-8
        
        logger.info(f"  g_r: mean={self.g_r_mean.shape}, std={self.g_r_std.shape}")
        logger.info(f"  g_h: mean={self.g_h_mean.shape}, std={self.g_h_std.shape}")
        logger.info(f"  g_m: mean={self.g_m_mean.shape}, std={self.g_m_std.shape}")
    
    def __len__(self) -> int:
        """Total number of samples."""
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with:
            - 'g_r_context': [context_length, 29]
            - 'g_h_context': [context_length, 72]
            - 'g_m_context': [context_length, 11]
            - 'g_r_target': [action_horizon, 29]
            - 'g_h_target': [action_horizon, 72]
            - 'g_m_target': [action_horizon, 11]
        """
        motion_name, frame_idx = self.sample_indices[idx]
        motion = self.motions[motion_name]
        
        # Context frames
        context_start = frame_idx
        context_end = frame_idx + self.context_length
        
        # Target frames
        target_start = context_end
        target_end = target_start + self.action_horizon
        
        # Extract
        g_r_context = motion['g_r'][context_start:context_end]
        g_h_context = motion['g_h'][context_start:context_end]
        g_m_context = motion['g_m'][context_start:context_end]
        
        g_r_target = motion['g_r'][target_start:target_end]
        g_h_target = motion['g_h'][target_start:target_end]
        g_m_target = motion['g_m'][target_start:target_end]
        
        # Normalize
        if self.normalize:
            g_r_context = (g_r_context - self.g_r_mean) / self.g_r_std
            g_h_context = (g_h_context - self.g_h_mean) / self.g_h_std
            g_m_context = (g_m_context - self.g_m_mean) / self.g_m_std
            
            g_r_target = (g_r_target - self.g_r_mean) / self.g_r_std
            g_h_target = (g_h_target - self.g_h_mean) / self.g_h_std
            g_m_target = (g_m_target - self.g_m_mean) / self.g_m_std
        
        return {
            'g_r_context': torch.from_numpy(g_r_context).float(),
            'g_h_context': torch.from_numpy(g_h_context).float(),
            'g_m_context': torch.from_numpy(g_m_context).float(),
            'g_r_target': torch.from_numpy(g_r_target).float(),
            'g_h_target': torch.from_numpy(g_h_target).float(),
            'g_m_target': torch.from_numpy(g_m_target).float(),
            'motion_name': motion_name,
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    context_length: int = 4,
    action_horizon: int = 8,
    split_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.
    
    Args:
        data_dir: Directory with processed motion files
        batch_size: Batch size
        num_workers: Number of worker processes
        context_length: Context frames
        action_horizon: Prediction horizon
        split_ratio: Train/val split
        
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = SonicMotionDataset(
        data_dir=data_dir,
        context_length=context_length,
        action_horizon=action_horizon,
        normalize=True,
        split='train',
        split_ratio=split_ratio,
    )
    
    val_dataset = SonicMotionDataset(
        data_dir=data_dir,
        context_length=context_length,
        action_horizon=action_horizon,
        normalize=True,
        split='val',
        split_ratio=split_ratio,
    )
    
    # Share normalization stats
    val_dataset.g_r_mean = train_dataset.g_r_mean
    val_dataset.g_r_std = train_dataset.g_r_std
    val_dataset.g_h_mean = train_dataset.g_h_mean
    val_dataset.g_h_std = train_dataset.g_h_std
    val_dataset.g_m_mean = train_dataset.g_m_mean
    val_dataset.g_m_std = train_dataset.g_m_std
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    return train_loader, val_loader
