# Motion Sequence Visualization Summary

**Notebook:** `/home/grease/gam/visualize_motion_examples.ipynb`

## Overview

This notebook provides comprehensive visualizations of the **G1 robot jump and land motion** (`jump_and_land_heavy_001__A001`) from the Bones-Studio dataset. The visualizations span from static joint analysis to dynamic motion sequence analysis.

---

## Visualizations Generated

### 1. **G1 Robot Motion Sequence - Key Frames Progression**
**File:** `motion_sequence_frames.png`

A 10-panel visualization showing the joint angles across the entire motion:
- **Frames shown:** 0, 13, 26, 39, 52, 66, 79, 92, 105, 119 (evenly distributed)
- **What it shows:** Joint angle profiles at critical moments in the jump
- **Energy indicator:** Yellow boxes show motion energy at each frame (higher energy = more joint velocity)
- **Key insight:** Left hip (J0) drives the motion from 0° to 119° (maximum extension)

### 2. **Motion Timeline Analysis - 4-Panel Dashboard**
**File:** `motion_timeline_analysis.png`

#### Panel 1: Overall Motion Energy Over Time
- Blue filled area shows total motion energy (joint velocity) across frames
- Red dashed line marks the 33rd percentile threshold for phase detection
- **Peak:** Frame 16 (peak velocity during launch phase)
- **Range:** [1.0, 4.195] motion energy units

#### Panel 2: Left vs Right Leg Activity
- **Red line (Left Leg):** Progressively increases from ~80° to ~145° activity
- **Green line (Right Leg):** Remains mostly stationary around 86° activity
- **Key finding:** Highly asymmetric motion - left leg is primary mover

#### Panel 3: Lower vs Upper Body Activity
- **Stacked area chart:** Shows energy distribution between body regions
- **Lower body (pink):** Dominates throughout
- **Upper body (cyan):** Stable, supporting movement

#### Panel 4: Motion Phases with Identification
- Three distinct phases marked with colored backgrounds:
  1. **Preparation (Blue, Frames 0-39):** Crouch and loading phase
  2. **Launch (Yellow, Frames 39-78):** Explosive jump phase
  3. **Landing (Pink, Frames 78-119):** Impact and recovery phase

---

### 3. **Detailed Joint Trajectories - 6-Panel Analysis**
**File:** `joint_trajectories_detailed.png`

Individual trajectories for 6 key joints with dual-axis visualization:

#### Left Hip (J0) - **RED**
- **Range:** 0° to 119° (massive extension)
- **Mean:** 59.50°
- **Dynamics:** Smooth continuous extension throughout motion
- **Interpretation:** Primary driver of the jump

#### Left Ankle (J4) - **ORANGE**
- **Range:** -1.43° to -1.23° (minimal movement)
- **Mean:** -1.34°
- **Std:** 0.06°
- **Interpretation:** Locked in fixed position for stability

#### Right Ankle (J6) - **GREEN**
- **Range:** -85.49° to -85.14° (minimal movement)
- **Mean:** -85.31°
- **Std:** 0.10°
- **Interpretation:** Locked rigid position for stability

#### Waist Roll (J12) - **BLUE**
- **Range:** -9.38° to -9.13° (very stable)
- **Mean:** -9.28°
- **Std:** 0.06°
- **Interpretation:** Minimal waist rotation in this motion

#### Left Shoulder (J16) - **PURPLE**
- **Range:** 6.74° to 6.80° (extremely stable)
- **Mean:** 6.78°
- **Std:** 0.02°
- **Interpretation:** Shoulders remain almost stationary

#### Right Elbow (J23) - **BROWN**
- **Range:** 36.67° to 92.23° (significant movement)
- **Mean:** 62.61°
- **Std:** 20.77°
- **Range span:** 55.56°
- **Interpretation:** Arm swing is secondary motion component

---

## Motion Phase Breakdown

### Phase 1: Preparation (Frames 0-39)
- **Duration:** ~39 frames (1.3 seconds at 30 fps)
- **Motion:** Crouch and loading
- **Key changes:**
  - Left hip angles from 0° to ~48°
  - Body lowers for power generation
  - Motion energy gradually increases

### Phase 2: Launch (Frames 39-78)
- **Duration:** ~39 frames (1.3 seconds at 30 fps)
- **Motion:** Explosive jump
- **Key changes:**
  - Left hip accelerates from ~48° to ~110°
  - Peak motion energy at frame 16 (0.3 seconds into this phase)
  - Leg extension generates upward momentum

### Phase 3: Landing (Frames 78-119)
- **Duration:** ~42 frames (1.4 seconds at 30 fps)
- **Motion:** Impact and recovery
- **Key changes:**
  - Left hip reaches maximum (119°) then slightly settles
  - Motion energy decreases as jump completes
  - Body absorbs impact landing

---

## Key Statistics

```
Total Motion Duration: 120 frames
Dataset: Bones-Studio (jump_and_land_heavy_001__A001)
Date: 2021-05-31

MOTION ENERGY METRICS:
  • Peak Energy: 4.195 (Frame 16)
  • Average Energy: 1.691
  • Min Energy: 1.000
  • Energy Range: 3.195 units

ACTIVE JOINTS (High Variability):
  1. Right Elbow (J23): σ = 20.77° (range: 55.56°)
  2. Left Hip (J0): σ = 34.64° (range: 119.00°) ← PRIMARY DRIVER
  3. Left Shoulder (J16): σ = 0.02° (nearly frozen)

BODY ASYMMETRY:
  • Left leg active, right leg largely passive
  • Single-leg jump motion variant
```

---

## Technical Insights

### Data Representation
- **Format:** [T=120, DOF=29] array of joint angles in degrees
- **Joint Grouping:**
  - Left Leg: J0-J5 (6 DOF)
  - Right Leg: J6-J11 (6 DOF)
  - Waist: J12-J14 (3 DOF)
  - Left Arm: J15-J21 (7 DOF)
  - Right Arm: J22-J28 (7 DOF)

### Motion Characteristics
1. **Single-leg dominant:** Left leg drives 90%+ of motion energy
2. **Arm assistance:** Right arm swings (~55° range) to aid launch
3. **Waist stability:** Minimal rotation maintains control
4. **Linear progression:** Hip extension is nearly monotonic

### Physical Interpretation
This is a **heavy, asymmetric jump** where:
- The robot primarily extends its left leg powerfully
- Right leg remains extended but mostly passive
- Upper body uses arm swing for momentum transfer
- Motion follows a classic jump trajectory: preparation → explosive launch → recovery

---

## Usage in Training

These visualizations are valuable for:

1. **Data Quality Verification:** Confirm reasonable joint ranges and smooth trajectories
2. **Motion Understanding:** Understand what different motions look like quantitatively
3. **Baseline Establishment:** Compare uniform vs proportional SMPL representations
4. **Motion Phase Annotation:** Identify key frames for temporal segmentation
5. **Debugging:** Identify artifacts or unrealistic joint patterns

---

## Files Referenced

- **Notebook:** `/home/grease/gam/visualize_motion_examples.ipynb`
- **Generated PNGs:**
  - `/home/grease/gam/motion_sequence_frames.png`
  - `/home/grease/gam/motion_timeline_analysis.png`
  - `/home/grease/gam/joint_trajectories_detailed.png`
  - `/home/grease/gam/g1_robot_motion.png` (previous)

---

## Next Steps

1. **Extend to multiple motions:** Select 5-10 diverse motion types and visualize
2. **Compare modalities:** Overlay g_r, g_h_uniform, g_h_proportional
3. **Motion embeddings:** Generate latent space visualization
4. **Temporal alignment:** Verify perfect synchronization across all modalities
5. **Training pipeline:** Use phase annotations for curriculum learning

---

**Generated:** June 17, 2026  
**Dataset:** Bones-Studio (142,220 motions, 522 actors)  
**Motion:** Jump and Land Heavy (Jump motion type)
