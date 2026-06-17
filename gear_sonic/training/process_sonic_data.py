#!/usr/bin/env python3
"""SONIC Data Processing Pipeline - Main Script.

This script orchestrates the complete data processing workflow:
1. Load raw motion data (G1 robot, SOMA human, mixed)
2. Process and align all modalities
3. Save processed triplets for training
4. Create train/val/test splits
5. Generate data statistics and reports
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import pandas as pd

from gear_sonic.training.sonic_data_processor import (
    SonicDataProcessor,
    MotionData,
)
from gear_sonic.training.sonic_dataset import SonicMotionDataset


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def process_dataset(
    config: dict,
    max_motions: Optional[int] = None,
) -> list:
    """Process dataset according to configuration.
    
    Args:
        config: Configuration dictionary
        max_motions: Override max_motions from config
        
    Returns:
        List of processed MotionData objects
    """
    dataset_config = config['dataset']
    processing_config = config['processing']
    
    data_root = dataset_config['root']
    metadata_path = os.path.join(data_root, dataset_config['metadata'])
    
    # Create processor
    processor = SonicDataProcessor(
        data_root=data_root,
        metadata_path=metadata_path,
    )
    
    # Get max motions
    if max_motions is None:
        max_motions = processing_config.get('max_motions', None)
    
    logger.info(f"Processing dataset with max_motions={max_motions}")
    
    # Process batch
    motions = processor.process_batch(
        max_motions=max_motions,
        verbose=True,
    )
    
    logger.info(f"Successfully processed {len(motions)} motions")
    logger.info(f"Failed: {len(processor.stats['failed_motions'])}")
    
    return motions


def save_processed_data(
    motions: list,
    output_dir: str,
    format: str = 'npy',
) -> None:
    """Save processed motions to disk.
    
    Args:
        motions: List of MotionData objects
        output_dir: Output directory path
        format: Output format (npy, csv, etc.)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(motions)} motions to {output_path}...")
    
    saved_count = 0
    
    for motion in motions:
        try:
            if format == 'npy':
                # Save as NPZ (compressed numpy)
                np.savez_compressed(
                    output_path / f"{motion.move_name}.npz",
                    g_r=motion.g_r,
                    g_h=motion.g_h,
                    g_m=motion.g_m,
                    move_name=motion.move_name,
                    actor_id=motion.actor_id,
                    date=motion.date,
                )
            elif format == 'csv':
                # Save as CSV (human readable)
                for name, data in [
                    ('g_r', motion.g_r),
                    ('g_h', motion.g_h),
                    ('g_m', motion.g_m),
                ]:
                    df = pd.DataFrame(data)
                    df.to_csv(
                        output_path / f"{motion.move_name}_{name}.csv",
                        index=False
                    )
            
            saved_count += 1
            
        except Exception as e:
            logger.error(f"Error saving {motion.move_name}: {e}")
    
    logger.info(f"Saved {saved_count}/{len(motions)} motions")


def create_data_splits(
    processed_dir: str,
    split_config: dict,
) -> None:
    """Create train/val/test splits.
    
    Args:
        processed_dir: Directory with processed motion files
        split_config: Split configuration
    """
    processed_path = Path(processed_dir)
    
    # Load all motion files
    motion_files = sorted(processed_path.glob('*.npz'))
    logger.info(f"Found {len(motion_files)} processed motions")
    
    # Compute split indices
    train_ratio = split_config.get('train_ratio', 0.8)
    val_ratio = split_config.get('val_ratio', 0.1)
    test_ratio = split_config.get('test_ratio', 0.1)
    
    # Normalize ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total
    
    n_train = int(len(motion_files) * train_ratio)
    n_val = int(len(motion_files) * val_ratio)
    
    # Create directories
    splits = {
        'train': motion_files[:n_train],
        'val': motion_files[n_train:n_train + n_val],
        'test': motion_files[n_train + n_val:],
    }
    
    for split_name, files in splits.items():
        split_dir = processed_path.parent / f"sonic_{split_name}"
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks or copy
        for motion_file in files:
            target = split_dir / motion_file.name
            if not target.exists():
                # Copy instead of symlink for portability
                import shutil
                shutil.copy(motion_file, target)
        
        logger.info(f"{split_name}: {len(files)} motions → {split_dir}")


def generate_statistics(
    processed_dir: str,
) -> dict:
    """Generate statistics about processed dataset.
    
    Args:
        processed_dir: Directory with processed motion files
        
    Returns:
        Statistics dictionary
    """
    processed_path = Path(processed_dir)
    motion_files = list(processed_path.glob('*.npz'))
    
    stats = {
        'total_motions': len(motion_files),
        'total_frames': 0,
        'g_r_shape': None,
        'g_h_shape': None,
        'g_m_shape': None,
        'motion_durations': [],
    }
    
    for motion_file in motion_files[:100]:  # Sample first 100
        try:
            data = np.load(motion_file, allow_pickle=True)
            T = len(data['g_r'])
            stats['total_frames'] += T
            stats['motion_durations'].append(T)
            
            if stats['g_r_shape'] is None:
                stats['g_r_shape'] = data['g_r'].shape[1:]
                stats['g_h_shape'] = data['g_h'].shape[1:]
                stats['g_m_shape'] = data['g_m'].shape[1:]
        except Exception as e:
            logger.warning(f"Error reading {motion_file}: {e}")
    
    if stats['motion_durations']:
        stats['avg_motion_duration'] = np.mean(stats['motion_durations'])
        stats['min_motion_duration'] = np.min(stats['motion_durations'])
        stats['max_motion_duration'] = np.max(stats['motion_durations'])
    
    return stats


def print_statistics(stats: dict) -> None:
    """Print statistics in readable format."""
    logger.info("=" * 60)
    logger.info("SONIC Dataset Statistics")
    logger.info("=" * 60)
    logger.info(f"Total motions: {stats['total_motions']}")
    logger.info(f"Total frames: {stats['total_frames']}")
    logger.info(f"g_r shape per frame: {stats['g_r_shape']}")
    logger.info(f"g_h shape per frame: {stats['g_h_shape']}")
    logger.info(f"g_m shape per frame: {stats['g_m_shape']}")
    
    if 'avg_motion_duration' in stats:
        logger.info(f"Avg motion duration: {stats['avg_motion_duration']:.1f} frames")
        logger.info(f"Min motion duration: {stats['min_motion_duration']} frames")
        logger.info(f"Max motion duration: {stats['max_motion_duration']} frames")
    logger.info("=" * 60)


def main():
    """Main pipeline."""
    parser = argparse.ArgumentParser(
        description='SONIC Data Processing Pipeline'
    )
    parser.add_argument(
        '--config',
        default='gear_sonic/training/config_sonic_data.yaml',
        help='Config file path'
    )
    parser.add_argument(
        '--max-motions',
        type=int,
        default=None,
        help='Max motions to process (override config)'
    )
    parser.add_argument(
        '--skip-processing',
        action='store_true',
        help='Skip processing, only create splits and stats'
    )
    parser.add_argument(
        '--skip-splits',
        action='store_true',
        help='Skip creating splits'
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    output_dir = config['dataset']['output_dir']
    
    # Process dataset
    if not args.skip_processing:
        logger.info("Starting data processing...")
        motions = process_dataset(config, max_motions=args.max_motions)
        
        # Save processed data
        save_processed_data(
            motions,
            output_dir,
            format=config['output']['format'],
        )
    else:
        logger.info("Skipping processing step")
    
    # Create splits
    if not args.skip_splits:
        logger.info("Creating train/val/test splits...")
        create_data_splits(output_dir, config['split'])
    
    # Generate statistics
    logger.info("Generating statistics...")
    stats = generate_statistics(output_dir)
    print_statistics(stats)
    
    logger.info("Data processing pipeline complete!")
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
