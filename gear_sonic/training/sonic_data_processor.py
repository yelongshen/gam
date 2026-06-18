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
        
        # G1 robot has 29 DOF - joint order matches CSV columns from Bones-Studio
        self.joint_order = [
            # Left leg (6 DOF)
            'left_hip_pitch_joint_dof',
            'left_hip_roll_joint_dof',
            'left_hip_yaw_joint_dof',
            'left_knee_joint_dof',
            'left_ankle_pitch_joint_dof',
            'left_ankle_roll_joint_dof',
            # Right leg (6 DOF)
            'right_hip_pitch_joint_dof',
            'right_hip_roll_joint_dof',
            'right_hip_yaw_joint_dof',
            'right_knee_joint_dof',
            'right_ankle_pitch_joint_dof',
            'right_ankle_roll_joint_dof',
            # Waist (3 DOF)
            'waist_yaw_joint_dof',
            'waist_roll_joint_dof',
            'waist_pitch_joint_dof',
            # Left arm (7 DOF)
            'left_shoulder_pitch_joint_dof',
            'left_shoulder_roll_joint_dof',
            'left_shoulder_yaw_joint_dof',
            'left_elbow_joint_dof',
            'left_wrist_roll_joint_dof',
            'left_wrist_pitch_joint_dof',
            'left_wrist_yaw_joint_dof',
            # Right arm (7 DOF)
            'right_shoulder_pitch_joint_dof',
            'right_shoulder_roll_joint_dof',
            'right_shoulder_yaw_joint_dof',
            'right_elbow_joint_dof',
            'right_wrist_roll_joint_dof',
            'right_wrist_pitch_joint_dof',
            'right_wrist_yaw_joint_dof',
        ]
        assert len(self.joint_order) == 29, f"Expected 29 joints, got {len(self.joint_order)}"
    
    def load_motion(self, csv_path: str) -> Optional[np.ndarray]:
        """Load G1 motion from CSV file.
        
        Args:
            csv_path: Path to CSV, can be:
                - Relative: "240918/body_check_001__A548.csv"  
                - Full relative: "g1/csv/240918/body_check_001__A548.csv"
                
        Returns:
            Motion array [T, 29] or None if load fails
        """
        csv_path = str(csv_path)
        
        # Handle paths that include "g1/csv/" prefix
        if csv_path.startswith("g1/csv/"):
            csv_path = csv_path.replace("g1/csv/", "", 1)
        
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
            bvh_path: Path to BVH file, can be:
                - Relative: "240918/body_check_001__A548.bvh"  
                - Full relative: "soma_proportional/bvh/240918/body_check_001__A548.bvh"
            use_proportional: Use proportional fit if True, else uniform
            
        Returns:
            Motion array [T, 72] (SMPL format) or None if load fails
        """
        bvh_path = str(bvh_path)
        
        # Handle paths that include directory prefix
        if bvh_path.startswith("soma_proportional/bvh/"):
            bvh_path = bvh_path.replace("soma_proportional/bvh/", "", 1)
            use_proportional = True
        elif bvh_path.startswith("soma_uniform/bvh/"):
            bvh_path = bvh_path.replace("soma_uniform/bvh/", "", 1)
            use_proportional = False
        
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
            # bvh-python not installed — fall through to manual FK parser (expected path)
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
        """
        Parse BVH and return SMPL-compatible world-space joint positions.

        Returns: np.ndarray [T, 72]  (24 SMPL joints × 3 coords, in BVH world units)

        The previous implementation incorrectly returned raw BVH channel values
        (rotation angles), which are meaningless as positions.  This version runs
        proper forward kinematics to compute the 3-D world position of each joint.
        """
        try:
            with open(bvh_path) as f:
                lines = f.read().split('\n')

            # ── 1. Parse hierarchy ────────────────────────────────────────────
            joints, offsets, channels, parents = [], {}, {}, {}
            stack, ch_idx, i = [], 0, 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('ROOT ') or line.startswith('JOINT '):
                    name = line.split()[1]
                    joints.append(name)
                    parents[name] = stack[-1] if stack else None
                    stack.append(name)
                elif line.startswith('OFFSET') and stack:
                    p = line.split()
                    offsets[stack[-1]] = np.array([float(p[1]), float(p[2]), float(p[3])])
                elif line.startswith('CHANNELS') and stack:
                    p = line.split(); n = int(p[1])
                    channels[stack[-1]] = {'start': ch_idx, 'types': p[2:2+n]}
                    ch_idx += n
                elif line.startswith('End Site'):   # consume without touching stack
                    i += 1
                    while i < len(lines):
                        if lines[i].strip() == '}': break
                        i += 1
                elif line == '}' and stack:
                    stack.pop()
                elif line.strip() == 'MOTION':
                    break
                i += 1

            # ── 2. Load motion frames ─────────────────────────────────────────
            motion_start_line = None
            motion_idx = None
            num_frames = None
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped == 'MOTION':
                    motion_start_line = i
                elif motion_start_line is not None and stripped.startswith('Frames:'):
                    num_frames = int(stripped.split()[1])
                elif motion_start_line is not None and stripped.startswith('Frame Time:'):
                    motion_idx = i + 1
                    break

            if motion_idx is None or num_frames is None:
                logger.warning(f"Could not parse BVH header in {bvh_path}")
                return None

            frame_data = []
            for k in range(motion_idx, motion_idx + num_frames):
                vals = lines[k].split()
                if vals:
                    frame_data.append([float(v) for v in vals])
            frame_data = np.array(frame_data, dtype=np.float64)

            # ── 3. Forward kinematics ─────────────────────────────────────────
            def Rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
            def Ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
            def Rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

            def fk_frame(frame):
                wp, wr = {}, {}
                for j in joints:
                    if j not in channels:
                        wp[j] = wp.get(parents.get(j), np.zeros(3)).copy()
                        wr[j] = wr.get(parents.get(j), np.eye(3)).copy()
                        continue
                    ch = channels[j]; start, types = ch['start'], ch['types']
                    pos_v = [None, None, None]; rot_a, rot_t = [], []
                    for k, t in enumerate(types):
                        v = frame[start + k]
                        if 'position' in t.lower():
                            pos_v['XYZ'.index(t[0].upper())] = v
                        else:
                            rot_a.append(v); rot_t.append(t)
                    R = np.eye(3)
                    for ang, t in zip(rot_a, rot_t):
                        a = np.radians(ang)
                        if t == 'Xrotation': R = R @ Rx(a)
                        elif t == 'Yrotation': R = R @ Ry(a)
                        elif t == 'Zrotation': R = R @ Rz(a)
                    parent = parents[j]
                    if parent is None:
                        wp[j] = np.array([v if v is not None else 0.0 for v in pos_v])
                        wr[j] = R
                    else:
                        lp = np.array([v if v is not None else 0.0 for v in pos_v])
                        wp[j] = wp.get(parent, np.zeros(3)) + wr.get(parent, np.eye(3)) @ offsets.get(j, np.zeros(3)) + lp
                        wr[j] = wr.get(parent, np.eye(3)) @ R
                return wp

            # ── 4. Map SOMA BVH joints → 24 SMPL indices ─────────────────────
            # SOMA BVH joint name → SMPL joint index (0-based, standard ordering)
            SOMA_TO_SMPL = {
                'Hips': 0,          # Pelvis
                'LeftLeg': 1,       # L_Hip
                'RightLeg': 2,      # R_Hip
                'Spine1': 3,        # Spine1
                'LeftShin': 4,      # L_Knee
                'RightShin': 5,     # R_Knee
                'Spine2': 6,        # Spine2
                'LeftFoot': 7,      # L_Ankle
                'RightFoot': 8,     # R_Ankle
                'Chest': 9,         # Spine3
                'LeftToeBase': 10,  # L_Foot
                'RightToeBase': 11, # R_Foot
                'Neck1': 12,        # Neck
                'LeftShoulder': 13, # L_Collar
                'RightShoulder': 14,# R_Collar
                'Head': 15,         # Head
                'LeftArm': 16,      # L_Shoulder
                'RightArm': 17,     # R_Shoulder
                'LeftForeArm': 18,  # L_Elbow
                'RightForeArm': 19, # R_Elbow
                'LeftHand': 20,     # L_Wrist
                'RightHand': 21,    # R_Wrist
                # joints 22-23 (hands) are finger-tip approx; leave as zero if absent
            }

            # ── 5. Compute positions for all frames ───────────────────────────
            result = np.zeros((len(frame_data), 72), dtype=np.float32)
            for fi, frame in enumerate(frame_data):
                wp = fk_frame(frame)
                for bvh_name, smpl_idx in SOMA_TO_SMPL.items():
                    if bvh_name in wp:
                        result[fi, smpl_idx * 3: smpl_idx * 3 + 3] = wp[bvh_name]

            return result

        except Exception as e:
            logger.error(f"Manual BVH FK parsing failed: {e}")
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
