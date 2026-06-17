"""Data loading utilities for egocentric dataset."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset


class EgocentricDataset(Dataset):
    """PyTorch Dataset for egocentric teleoperation data stored in Parquet format.
    
    Expects data structure:
    ```
    data_root/
    ├── unitreerobotics_datasets/
    │   └── [TASK_NAME]/
    │       └── data/
    │           └── chunk-000/
    │               ├── episode_000000.parquet
    │               ├── episode_000001.parquet
    │               └── ...
    ```
    
    Each parquet file contains columns:
    - observation.*: body pose, gripper state, end-effector position
    - action.*: joint commands and gripper targets
    - timestamp, frame_index, episode_index, etc.
    """

    def __init__(
        self,
        data_root: str,
        task_name: str = "G1_Dex1_Fold_Towel",
        context_length: int = 4,
        action_horizon: int = 8,
        max_episodes: Optional[int] = None,
        normalize: bool = True,
    ):
        """Initialize the egocentric dataset.
        
        Args:
            data_root: Root directory containing datasets
            task_name: Name of the task folder
            context_length: Number of past frames to include as context
            action_horizon: Number of future actions to predict
            max_episodes: Max episodes to load (None = all)
            normalize: Whether to normalize observations/actions
        """
        self.data_root = Path(data_root)
        self.task_name = task_name
        self.context_length = context_length
        self.action_horizon = action_horizon
        self.normalize = normalize
        
        # Find all parquet files
        data_dir = self.data_root / task_name / "data" / "chunk-000"
        if not data_dir.exists():
            # Try alternate path structure
            data_dir = self.data_root / "unitreerobotics_datasets" / task_name / "data" / "chunk-000"
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Could not find data directory at {data_dir}")
        
        self.episode_files = sorted(
            [f for f in data_dir.glob("episode_*.parquet")],
            key=lambda x: int(x.stem.split("_")[1])
        )
        
        if max_episodes:
            self.episode_files = self.episode_files[:max_episodes]
        
        print(f"Found {len(self.episode_files)} episodes for task '{task_name}' at {data_dir}")
        
        # Load all episodes into memory
        self.episodes = []
        self._load_episodes()
        
        # Compute normalization statistics
        if self.normalize:
            self._compute_normalization_stats()
    
    def _load_episodes(self):
        """Load all episodes from parquet files."""
        for ep_file in self.episode_files:
            try:
                df = pd.read_parquet(ep_file)
                self.episodes.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {ep_file}: {e}")
    
    def _parse_observation(self, row: pd.Series) -> np.ndarray:
        """Parse observation from a row."""
        obs_parts = []
        
        # Extract left/right arm, end-effectors, body pose
        for key in [
            "observation.left_arm",
            "observation.right_arm",
            "observation.left_gripper",
            "observation.right_gripper",
            "observation.left_ee",
            "observation.right_ee",
            "observation.body",
        ]:
            if key in row.index:
                val = row[key]
                if isinstance(val, (list, np.ndarray)):
                    obs_parts.append(np.array(val, dtype=np.float32))
                else:
                    obs_parts.append(np.array([val], dtype=np.float32))
        
        obs = np.concatenate(obs_parts)
        return obs.astype(np.float32)
    
    def _parse_action(self, row: pd.Series) -> np.ndarray:
        """Parse action from a row."""
        action_parts = []
        
        # Extract left/right arm, gripper, ee commands
        for key in [
            "action.left_arm",
            "action.right_arm",
            "action.left_gripper",
            "action.right_gripper",
            "action.left_ee",
            "action.right_ee",
            "action.body",
        ]:
            if key in row.index:
                val = row[key]
                if isinstance(val, (list, np.ndarray)):
                    action_parts.append(np.array(val, dtype=np.float32))
                else:
                    action_parts.append(np.array([val], dtype=np.float32))
        
        action = np.concatenate(action_parts)
        return action.astype(np.float32)
    
    def _compute_normalization_stats(self):
        """Compute mean/std for normalization."""
        all_obs = []
        all_actions = []
        
        for ep in self.episodes:
            for _, row in ep.iterrows():
                all_obs.append(self._parse_observation(row))
                all_actions.append(self._parse_action(row))
        
        if not all_obs:
            print("Warning: No data found for normalization")
            return
        
        all_obs = np.array(all_obs)
        all_actions = np.array(all_actions)
        
        self.obs_mean = all_obs.mean(axis=0)
        self.obs_std = all_obs.std(axis=0) + 1e-8
        self.action_mean = all_actions.mean(axis=0)
        self.action_std = all_actions.std(axis=0) + 1e-8
        
        print(f"Observation shape: {self.obs_mean.shape}, Action shape: {self.action_mean.shape}")
    
    def __len__(self) -> int:
        """Total number of valid samples across all episodes."""
        total = 0
        for ep in self.episodes:
            # A sample requires context_length + action_horizon frames
            ep_len = len(ep)
            valid_frames = max(0, ep_len - self.context_length - self.action_horizon)
            total += valid_frames
        return total
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample with context and action horizon.
        
        Returns:
            obs_context: (context_length, obs_dim) tensor
            action_targets: (action_horizon, action_dim) tensor
        """
        # Find which episode and frame this index corresponds to
        start_frame = 0
        ep = None
        remaining_idx = idx
        
        for episode in self.episodes:
            ep_len = len(episode)
            valid_frames = max(0, ep_len - self.context_length - self.action_horizon)
            
            if remaining_idx < valid_frames:
                # This sample is in this episode
                start_frame = remaining_idx
                ep = episode
                break
            else:
                remaining_idx -= valid_frames
        
        if ep is None:
            raise IndexError(f"Invalid index {idx}")
        
        # Get context frames
        obs_context = []
        for i in range(self.context_length):
            frame = start_frame + i
            obs = self._parse_observation(ep.iloc[frame])
            if self.normalize:
                obs = (obs - self.obs_mean) / self.obs_std
            obs_context.append(obs)
        
        obs_context = np.stack(obs_context)  # (context_length, obs_dim)
        
        # Get action targets
        action_targets = []
        for i in range(self.action_horizon):
            frame = start_frame + self.context_length + i
            action = self._parse_action(ep.iloc[frame])
            if self.normalize:
                action = (action - self.action_mean) / self.action_std
            action_targets.append(action)
        
        action_targets = np.stack(action_targets)  # (action_horizon, action_dim)
        
        return (
            torch.from_numpy(obs_context).float(),
            torch.from_numpy(action_targets).float(),
        )


class EgoDataLoader:
    """Helper class to create DataLoaders for egocentric data."""
    
    @staticmethod
    def create(
        data_root: str,
        task_name: str = "G1_Dex1_Fold_Towel",
        context_length: int = 4,
        action_horizon: int = 8,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
        max_episodes: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders.
        
        Args:
            data_root: Root directory containing datasets
            task_name: Task folder name
            context_length: Frames of context
            action_horizon: Frames to predict
            batch_size: Batch size
            num_workers: Number of data loading workers
            train_split: Train/val split ratio
            max_episodes: Max episodes to load
        
        Returns:
            (train_loader, val_loader)
        """
        dataset = EgocentricDataset(
            data_root=data_root,
            task_name=task_name,
            context_length=context_length,
            action_horizon=action_horizon,
            max_episodes=max_episodes,
            normalize=True,
        )
        
        # Split dataset
        total_len = len(dataset)
        train_len = int(total_len * train_split)
        val_len = total_len - train_len
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_len, val_len]
        )
        
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
        
        return train_loader, val_loader
