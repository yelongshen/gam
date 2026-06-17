"""SONIC Data Processing Pipeline for Full Multi-Modal Training.

This module handles:
1. Loading G1 robot motion data (g_r) from CSV files
2. Loading SOMA human motion data (g_h) from BVHS/proportional format
3. Creating mixed representations (g_m) from VR + lower body
4. Synchronizing and aligning all three modalities
5. Saving processed data for training
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MotionData:
    """Container for motion data triplet (g_r, g_h, g_m)."""
    
    g_r: np.ndarray  # [T, 29] robot joint positions
    g_h: np.ndarray  # [T, 72] SMPL human joint positions (3 coords × 24 joints)
    g_m: np.ndarray  # [T, 11] mixed representation (head + wrists + lower body)
    
    # Metadata
    move_name: str
    actor_id: str
    date: str
    duration_frames: int
    
    # Statistics
    g_r_range: Tuple[float, float] = (0.0, 0.0)  # Joint range for normalization
    
    def __len__(self):
        """Length of motion (assumes all have same length after processing)."""
        return len(self.g_r)
    
    def validate(self) -> bool:
        """Validate shapes match."""
        T = len(self.g_r)
        if len(self.g_h) != T or len(self.g_m) != T:
            logger.warning(
                f"Shape mismatch for {self.move_name}: "
                f"g_r={self.g_r.shape}, g_h={self.g_h.shape}, g_m={self.g_m.shape}"
            )
            return False
        return True


class G1RobotDataLoader:
    """Load G1 robot motion data from CSV files."""
    
    def __init__(self, data_root: Path):
        """Initialize G1 data loader.
        
        Args:
            data_root: Path to bones-studio-seed dataset root
        """
        self.data_root = Path(data_root)
        self.g1_csv_root = self.data_root / "g1" / "csv"
        
        if not self.g1_csv_root.exists():
            raise FileNotFoundError(f"G1 CSV directory not found: {self.g1_csv_root}")
        
        # G1 robot has 29 DOF: 7 per arm + 7 per leg + 3 base + 5 spine
        self.joint_order = [
            # Left arm (7 DOF)
            'L_shoulder_roll', 'L_shoulder_pitch', 'L_shoulder_yaw',
            'L_elbow', 'L_wrist_roll', 'L_wrist_pitch', 'L_wrist_yaw',
            # Right arm (7 DOF)
            'R_shoulder_roll', 'R_shoulder_pitch', 'R_shoulder_yaw',
            'R_elbow', 'R_wrist_roll', 'R_wrist_pitch', 'R_wrist_yaw',
            # Left leg (6 DOF)
            'L_hip_roll', 'L_hip_pitch', 'L_hip_yaw',
            'L_knee', 'L_ankle_pitch', 'L_ankle_roll',
            # Right leg (6 DOF)
            'R_hip_roll', 'R_hip_pitch', 'R_hip_yaw',
            'R_knee', 'R_ankle_pitch', 'R_ankle_roll',
            # Waist (1 DOF)
            'waist_yaw',
            # Head (2 DOF)
            'head_pitch', 'head_yaw',
        ]
        assert len(self.joint_order) == 29, f"Expected 29 joints, got {len(self.joint_order)}"
    
    def load_motion(self, csv_path: str) -> Optional[np.ndarray]:
        """Load G1 motion from CSV file.
        
        Args:
            csv_path: Relative path from g1/csv (e.g., "240918/body_check_001__A548.csv")
            
        Returns:
            Motion array [T, 29] or None if load fails
        """
        full_path = self.g1_csv_root / csv_path
        
        if not full_path.exists():
            logger.warning(f"G1 CSV file not found: {full_path}")
            return None
        
        try:
            df = pd.read_csv(full_path)
            
            # Extract joint columns (assuming column names match joint_order)
            motion = []
            for joint in self.joint_order:
                if joint in df.columns:
                    motion.append(df[joint].values)
                else:
                    # Try alternative naming
                    alt_names = [
                        col for col in df.columns
                        if joint.lower() in col.lower()
                    ]
                    if alt_names:
                        motion.append(df[alt_names[0]].values)
                    else:
                        logger.warning(f"Joint {joint} not found in {full_path}")
                        return None
            
            motion = np.stack(motion, axis=1).astype(np.float32)
            
            # Validate
            if motion.shape[1] != 29:
                logger.warning(f"Invalid shape {motion.shape} for {full_path}")
                return None
            
            return motion
            
        except Exception as e:
            logger.error(f"Error loading {full_path}: {e}")
            return None


class SOMAHumanDataLoader:
    """Load SOMA human motion data from BVH or proportional formats."""
    
    def __init__(self, data_root: Path):
        """Initialize SOMA data loader.
        
        Args:
            data_root: Path to bones-studio-seed dataset root
        """
        self.data_root = Path(data_root)
        
        # Extract tar files if needed
        self._ensure_extracted()
        
        self.soma_uniform_root = self.data_root / "soma_uniform"
        self.soma_proportional_root = self.data_root / "soma_proportional"
        
        # SMPL has 24 joints (including root), 3 coords each = 72 dims
        self.num_joints = 24
        self.joint_dim = 72  # 24 joints × 3 coordinates
    
    def _ensure_extracted(self):
        """Extract tar files if needed."""
        import tarfile
        
        tar_files = [
            (self.data_root / "soma_uniform.tar.gz", self.data_root / "soma_uniform"),
            (self.data_root / "soma_proportional.tar.gz", self.data_root / "soma_proportional"),
        ]
        
        for tar_path, extract_path in tar_files:
            if tar_path.exists() and not extract_path.exists():
                logger.info(f"Extracting {tar_path.name}...")
                with tarfile.open(tar_path) as tar:
                    tar.extractall(self.data_root)
    
    def load_motion(self, bvh_path: str, use_proportional: bool = True) -> Optional[np.ndarray]:
        """Load SOMA motion from BVH file.
        
        Args:
            bvh_path: Relative path from soma_*/bvh (e.g., "240918/body_check_001__A548.bvh")
            use_proportional: Use proportional fit if True, else uniform
            
        Returns:
            Motion array [T, 72] (SMPL format) or None if load fails
        """
        if use_proportional:
            full_path = self.soma_proportional_root / "bvh" / bvh_path
        else:
            full_path = self.soma_uniform_root / "bvh" / bvh_path
        
        if not full_path.exists():
            logger.warning(f"BVH file not found: {full_path}")
            return None
        
        try:
            motion = self._parse_bvh(full_path)
            if motion is None:
                return None
            
            # Ensure [T, 72] shape
            if motion.shape[1] != self.joint_dim:
                logger.warning(
                    f"Invalid shape {motion.shape} for {full_path}, "
                    f"expected (T, {self.joint_dim})"
                )
                return None
            
            return motion.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error loading {full_path}: {e}")
            return None
    
    def _parse_bvh(self, bvh_path: Path) -> Optional[np.ndarray]:
        """Parse BVH file to get joint positions.
        
        Returns motion array [T, 72] where each joint has (x, y, z).
        """
        try:
            from bvh import Bvh
        except ImportError:
            logger.warning("bvh-python not installed, trying manual parsing...")
            return self._parse_bvh_manual(bvh_path)
        
        try:
            with open(bvh_path) as f:
                mocap = Bvh(f.read())
            
            # Extract frames
            frames = mocap.get_joint_channel_index(mocap.root)
            motion = []
            
            for frame_idx in range(mocap.nframes):
                frame_data = mocap.get_joint(mocap.root, frame_idx)
                motion.append(frame_data)
            
            return np.array(motion, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"BVH parsing failed: {e}")
            return None
    
    def _parse_bvh_manual(self, bvh_path: Path) -> Optional[np.ndarray]:
        """Manual BVH parsing as fallback.
        
        Extracts position data from BVH format.
        """
        try:
            with open(bvh_path) as f:
                lines = f.readlines()
            
            # Find MOTION section
            motion_idx = None
            num_frames = None
            frame_time = None
            
            for i, line in enumerate(lines):
                if line.startswith("Frames:"):
                    num_frames = int(line.split()[1])
                elif line.startswith("Frame Time:"):
                    frame_time = float(line.split()[2])
                elif line.strip() == "MOTION":
                    motion_idx = i + 2  # Skip "MOTION" and frame info lines
                    break
            
            if motion_idx is None or num_frames is None:
                logger.warning(f"Could not parse BVH header in {bvh_path}")
                return None
            
            # Parse frame data
            motion = []
            for i in range(motion_idx, motion_idx + num_frames):
                if i < len(lines):
                    frame_vals = [float(x) for x in lines[i].split()]
                    motion.append(frame_vals)
            
            motion = np.array(motion, dtype=np.float32)
            
            # Ensure 72 dimensions
            if motion.shape[1] >= 72:
                motion = motion[:, :72]  # Take first 72 dims
            else:
                logger.warning(
                    f"BVH has {motion.shape[1]} dims, expected 72. Padding..."
                )
                motion = np.pad(motion, ((0, 0), (0, 72 - motion.shape[1])))
            
            return motion
            
        except Exception as e:
            logger.error(f"Manual BVH parsing failed: {e}")
            return None


class MixedRepresentationBuilder:
    """Create mixed representation (g_m) from VR + lower body."""
    
    @staticmethod
    def build_mixed_representation(
        g_h: np.ndarray,
        g_r: np.ndarray,
        vr_joints: Optional[List[int]] = None,
        lower_body_joints: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Build mixed representation combining VR trackers and lower body.
        
        Args:
            g_h: SMPL human motion [T, 72]
            g_r: G1 robot motion [T, 29]
            vr_joints: SMPL joint indices for VR (head, left wrist, right wrist)
                Default: [head=15, l_wrist=20, r_wrist=21] × 3 coords = 27 dims
            lower_body_joints: Lower body joint indices from g_r
                Default: left_leg (6) + right_leg (6) + waist (1) = 13 dims
        
        Returns:
            Mixed representation [T, 11] approximately
            - 3D head position (3)
            - 3D left wrist position (3)
            - 3D right wrist position (3)
            - Lower body state from g_r (remaining)
        """
        T = g_h.shape[0]
        
        # Default VR joint indices (SMPL format)
        if vr_joints is None:
            # Head (joint 15), left wrist (joint 20), right wrist (joint 21)
            vr_joints = [15, 20, 21]
        
        # Extract VR part from g_h
        vr_data = []
        for joint_idx in vr_joints:
            start = joint_idx * 3
            end = start + 3
            vr_data.append(g_h[:, start:end])
        
        vr_features = np.concatenate(vr_data, axis=1)  # [T, 9]
        
        # Default lower body from g_r: legs + waist
        if lower_body_joints is None:
            # Left leg (6), right leg (6), waist (1) = 13
            # But we only take key joints
            lower_body_joints = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28]
        
        lower_body_data = g_r[:, lower_body_joints]  # [T, n_lower]
        
        # Combine: normalize dimensions by averaging
        # Target: 11 dims
        # Strategy: 9 from VR + 2 compressed from lower body
        
        if lower_body_data.shape[1] > 2:
            # PCA or averaging to compress
            lower_compressed = lower_body_data.mean(axis=1, keepdims=True)  # [T, 1]
            lower_compressed = np.concatenate([
                lower_compressed,
                lower_body_data[:, -1:],  # Keep last joint
            ], axis=1)  # [T, 2]
        else:
            lower_compressed = lower_body_data
        
        # Final mixed: [T, 9+2] = [T, 11]
        g_m = np.concatenate([vr_features, lower_compressed], axis=1)
        
        return g_m.astype(np.float32)


class SonicDataProcessor:
    """Main processor coordinating all data loading and processing."""
    
    def __init__(self, data_root: str, metadata_path: str):
        """Initialize SONIC data processor.
        
        Args:
            data_root: Path to bones-studio-seed dataset root
            metadata_path: Path to metadata parquet file
        """
        self.data_root = Path(data_root)
        
        # Load metadata
        self.metadata = pd.read_parquet(metadata_path)
        logger.info(f"Loaded metadata with {len(self.metadata)} entries")
        
        # Initialize sub-loaders
        self.g1_loader = G1RobotDataLoader(self.data_root)
        self.soma_loader = SOMAHumanDataLoader(self.data_root)
        
        # Statistics
        self.stats = {
            'total_motions': len(self.metadata),
            'loaded_g_r': 0,
            'loaded_g_h': 0,
            'successful_triplets': 0,
            'failed_motions': [],
        }
    
    def process_single_motion(
        self,
        row: pd.Series,
        verbose: bool = False,
    ) -> Optional[MotionData]:
        """Process a single motion row from metadata.
        
        Args:
            row: Metadata row containing paths and info
            verbose: Print debug info
            
        Returns:
            MotionData triplet or None if processing fails
        """
        move_name = row['move_name']
        
        # Load g_r (robot)
        g_r = self.g1_loader.load_motion(row['move_g1_path'])
        if g_r is None:
            if verbose:
                logger.warning(f"Failed to load g_r for {move_name}")
            self.stats['failed_motions'].append(move_name)
            return None
        
        # Load g_h (human - proportional)
        g_h = self.soma_loader.load_motion(
            row['move_soma_proportional_path'],
            use_proportional=True
        )
        if g_h is None:
            # Fallback to uniform
            g_h = self.soma_loader.load_motion(
                row['move_soma_uniform_path'],
                use_proportional=False
            )
        
        if g_h is None:
            if verbose:
                logger.warning(f"Failed to load g_h for {move_name}")
            self.stats['failed_motions'].append(move_name)
            return None
        
        # Align lengths
        min_len = min(len(g_r), len(g_h))
        g_r = g_r[:min_len]
        g_h = g_h[:min_len]
        
        # Build g_m (mixed)
        g_m = MixedRepresentationBuilder.build_mixed_representation(g_h, g_r)
        assert g_m.shape[0] == min_len
        
        # Create MotionData
        motion_data = MotionData(
            g_r=g_r,
            g_h=g_h,
            g_m=g_m,
            move_name=move_name,
            actor_id=row.get('take_actor', 'unknown'),
            date=row.get('take_date', 'unknown'),
            duration_frames=row.get('move_duration_frames', min_len),
        )
        
        if motion_data.validate():
            self.stats['successful_triplets'] += 1
            return motion_data
        else:
            self.stats['failed_motions'].append(move_name)
            return None
    
    def process_batch(
        self,
        indices: Optional[List[int]] = None,
        max_motions: Optional[int] = None,
        verbose: bool = False,
    ) -> List[MotionData]:
        """Process batch of motions.
        
        Args:
            indices: Specific row indices to process
            max_motions: Limit number of motions to process
            verbose: Print progress
            
        Returns:
            List of successfully processed MotionData objects
        """
        if indices is None:
            indices = list(range(len(self.metadata)))
        
        if max_motions:
            indices = indices[:max_motions]
        
        motions = []
        total_indices = len(indices)
        
        for i, idx in enumerate(indices):
            if verbose and i % 100 == 0:
                logger.info(f"Processing motion {i}/{total_indices}...")
            
            row = self.metadata.iloc[idx]
            motion = self.process_single_motion(row, verbose=False)
            
            if motion:
                motions.append(motion)
        
        logger.info(
            f"Processed {len(motions)}/{total_indices} motions successfully. "
            f"Failed: {len(self.stats['failed_motions'])}"
        )
        
        return motions
    
    def save_motions(
        self,
        motions: List[MotionData],
        output_dir: str,
        format: str = 'npy',
    ) -> None:
        """Save processed motions to disk.
        
        Args:
            motions: List of MotionData objects
            output_dir: Output directory
            format: 'npy', 'h5', or 'csv'
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(motions)} motions to {out_path}...")
        
        for motion in motions:
            if format == 'npy':
                # Save as NPZ (compressed)
                np.savez(
                    out_path / f"{motion.move_name}.npz",
                    g_r=motion.g_r,
                    g_h=motion.g_h,
                    g_m=motion.g_m,
                    move_name=motion.move_name,
                    actor_id=motion.actor_id,
                )
            elif format == 'csv':
                # Save as CSV (human readable)
                for name, data in [('g_r', motion.g_r), ('g_h', motion.g_h), ('g_m', motion.g_m)]:
                    df = pd.DataFrame(data)
                    df.to_csv(
                        out_path / f"{motion.move_name}_{name}.csv",
                        index=False
                    )
        
        logger.info(f"Saved {len(motions)} motions")


def main():
    """Example usage of SONIC data processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SONIC data processor')
    parser.add_argument(
        '--data-root',
        default='/home/grease/ego_dataset/work_bearlu/data/bones-studio-seed',
        help='Path to dataset root'
    )
    parser.add_argument(
        '--output-dir',
        default='./data/sonic_processed',
        help='Output directory'
    )
    parser.add_argument(
        '--max-motions',
        type=int,
        default=10,
        help='Max motions to process'
    )
    parser.add_argument(
        '--format',
        choices=['npy', 'csv'],
        default='npy',
        help='Output format'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create processor
    processor = SonicDataProcessor(
        data_root=args.data_root,
        metadata_path=os.path.join(
            args.data_root,
            'metadata/seed_metadata_v004.parquet'
        ),
    )
    
    # Process batch
    logger.info(f"Processing up to {args.max_motions} motions...")
    motions = processor.process_batch(
        max_motions=args.max_motions,
        verbose=True,
    )
    
    # Save
    processor.save_motions(motions, args.output_dir, format=args.format)
    
    logger.info(f"Done! Processed {len(motions)} motions")


if __name__ == '__main__':
    main()
