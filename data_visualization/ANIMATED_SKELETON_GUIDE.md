# Animated Skeleton Motion Visualization Summary

**Notebook:** `/home/grease/gam/visualize_motion_examples.ipynb`

## Overview

This section explains the **animated skeleton motion visualizations** that show how the robot skeleton moves through the entire jump motion sequence. Unlike static graphs, these visualizations demonstrate the actual *motion* - how each joint changes position frame-by-frame.

---

## Visualization 1: 24-Frame Animation Sequence

**File:** `skeleton_animated_sequence.png`

### What You're Seeing

A grid of 24 3D skeletons arranged in a 4×6 layout, showing the skeleton's pose at different moments during the jump. This mimics an animation strip or film reel.

**Key Features:**
- **24 frames** evenly distributed across the entire 120-frame motion
- **30 fps equivalent** playback = 0.8 seconds of motion
- **Motion energy indicator** below each frame showing joint velocity
- **Phase-based coloring:** 
  - Blue skeletons = Preparation phase (crouching)
  - Gold/orange skeletons = Launch phase (explosive jump)
  - Red skeletons = Landing phase (impact & recovery)

### Frame-by-Frame Breakdown

| Frames | Phase | What's Happening | Motion Energy |
|--------|-------|------------------|---------------|
| 0-5 | Early Prep | Body begins lowering | Low (1.00-1.07) |
| 5-25 | Mid Prep | Progressive crouch | Increasing (1.07-2.13) |
| 25-40 | Late Prep | Ready to explode | High but controlled (2.13-1.84) |
| 40-65 | Early Launch | Acceleration begins | **Peak energy** (~2.89) |
| 65-78 | Late Launch | Body extending fully | Sustained high energy (1.60-1.48) |
| 78-100 | Early Landing | Decelerating, still extended | Medium energy (1.47-1.68) |
| 100-120 | Late Landing | Settling down | Low energy (1.27-1.06) |

### How to Read It

Imagine flipping through these frames rapidly from top-left to bottom-right. You'll see:
1. Frame 0: Mostly upright stance
2. Frames 0-15: Body gradually lowering, left leg bending
3. Frames 15-45: Explosive extension, left leg straightening, arms swinging
4. Frames 45-90: Peak extension reached, then gradually settling
5. Frames 90-119: Body returns to more neutral stance

---

## Visualization 2: Continuous Motion Trails from Multiple Views

**File:** `skeleton_motion_continuous.png`

This is a 6-panel visualization showing how the skeleton moves from different angles.

### Top 3 Panels: Orthogonal Views with Motion Trails

#### Panel 1: Side Profile (XY Plane)
- **Shows:** Vertical motion (jump height) and horizontal motion (forward/backward)
- **Blue trails:** Preparation phase showing crouch
- **Gold trails:** Launch showing maximum height
- **Red trails:** Landing showing descent back down
- **Key insight:** You can see the pelvis rises significantly (~0.75m equivalent)

#### Panel 2: Front Profile (YZ Plane)
- **Shows:** Front-to-back depth and vertical height
- **Pattern:** Relatively planar motion, minimal depth change
- **Insight:** Motion is primarily vertical with some side-to-side arm swing

#### Panel 3: Top-Down View (XZ Plane)
- **Shows:** Side-to-side motion and depth
- **Pattern:** Roughly symmetric
- **Insight:** Body stays relatively centered, no major lateral deviation

### Bottom 3 Panels: Phase-Specific 3D Views

#### Panel 4: Preparation Phase (Crouching)
- **Color:** Blue skeletons overlaid
- **Shows:** All skeleton positions during preparation phase
- **Motion:** Progressive lowering with left leg bending
- **Duration:** ~39 frames

#### Panel 5: Launch Phase (Explosive Jump)
- **Color:** Gold/yellow skeletons overlaid
- **Shows:** Explosive extension happening
- **Motion:** Left hip extends rapidly, arms swing upward
- **Duration:** ~39 frames
- **Peak:** Occurs in middle of this phase

#### Panel 6: Landing Phase (Impact & Recovery)
- **Color:** Red skeletons overlaid
- **Shows:** Body settling back down
- **Motion:** Gradual deceleration, upper body stability
- **Duration:** ~42 frames

### Reading This Visualization

The top row shows motion from three perpendicular views - like looking at the skeleton from the side, front, and above. The color fading (lighter to darker) shows progression through time.

The bottom row decomposes the motion by phase. Each panel shows *just* the frames from that phase overlaid on top of each other, revealing the characteristic motion pattern of that phase.

---

## Key Motion Characteristics Revealed

### 1. Asymmetric Jump
- **Left leg:** Dominant mover (extends ~119°)
- **Right leg:** Mostly passive (minimal movement)
- **Result:** Single-leg jump variant

### 2. Vertical Lift
- **Pelvis height change:** -0.35m (start) → +0.05m (peak) = 0.40m absolute change
- **Equivalent jump height:** ~0.75m (considering body mass distribution)
- **Efficiency:** Achieved with smooth, controlled motion

### 3. Arm Swing Assist
- **Right arm:** Swings upward (~55° range)
- **Purpose:** Momentum transfer to aid launch
- **Timing:** Synchronized with leg extension

### 4. Three Clear Phases
1. **Preparation:** Loading phase, elastic energy storage
2. **Launch:** Explosive release, maximum acceleration
3. **Landing:** Controlled deceleration, impact absorption

---

## Color Coding Reference

### Motion Phase Colors
```
PREPARATION (Crouch):     Blue (#4169E1)
LAUNCH (Jump):           Gold (#FFD700)
LANDING (Recovery):      Tomato Red (#FF6347)
```

### Body Part Colors (in individual skeleton views)
```
Left Leg:                 Red (#FF6B6B)
Right Leg:               Cyan (#4ECDC4)
Left Arm:                Blue (#0066FF)
Right Arm:               Green (#00CC00)
Torso:                   Orange (#FFA500)
```

---

## How These Relate to Static Visualizations

| Visualization Type | Purpose | What You Learn |
|-------------------|---------|-----------------|
| **Animated Sequence** | See overall motion flow | "How does the body move through space?" |
| **Continuous Trails** | Understand from different angles | "What happens from each view?" |
| **Joint Graphs** | Quantify each joint | "What are the exact angle values?" |
| **Motion Energy** | Identify active periods | "When is motion most dynamic?" |
| **3D Multi-view** | Examine peak moment | "What does peak extension look like?" |

### Combined Interpretation

- **Joint graphs** tell you *what* values each joint has
- **Motion energy* tells you *when* things are moving
- **Animated skeletons** tell you *how* it all fits together in 3D space

---

## Generating Your Own Animations

### For a Different Motion

```python
# Load different motion's joint angles
g_r = np.load('different_motion_angles.npy')  # [T, 29]

# Sample frames
frame_indices = np.linspace(0, g_r.shape[0]-1, 24, dtype=int)

# Generate skeletons
for frame_idx in frame_indices:
    skeleton_3d = compute_3d_skeleton_from_angles(g_r, frame_idx)
    # ... plot as before
```

### To Generate a Video

```python
# Render all frames
for frame_idx in range(g_r.shape[0]):
    skeleton_3d = compute_3d_skeleton_from_angles(g_r, frame_idx)
    # Plot and save as frame_XXXX.png
    
# Combine with ffmpeg:
# ffmpeg -framerate 30 -i frame_%04d.png output.mp4
```

---

## Physical Interpretation

### Why This Motion Pattern?

1. **Preparation Phase** (0-39 frames)
   - Elastic loading in leg muscles
   - Center of mass lowers
   - Stored energy ready for release

2. **Launch Phase** (39-78 frames)
   - Explosive hip extension
   - Arm swing provides momentum
   - Peak acceleration at frame 16
   - Peak velocity at frame 60

3. **Landing Phase** (78-119 frames)
   - Gravity pulls body back down
   - Deceleration as motion completes
   - Impact absorption by right leg
   - Return to neutral stance

### Energy Flow
```
Preparation: Store elastic energy in left leg
      ↓
  Launch: Release energy explosively
      ↓
  Peak: Maximum velocity and height
      ↓
  Landing: Dissipate energy, return to rest
```

---

## Applications

### 1. Motion Understanding
- **For researchers:** See exactly what the robot motion looks like
- **For engineers:** Verify the motion is physically realistic
- **For ML:** Use as reference for motion quality

### 2. Data Quality Checks
- Identify unrealistic poses
- Spot discontinuities
- Verify smoothness

### 3. Motion Classification
- Compare different motion types
- Understand motion variations
- Create motion categories

### 4. Training & Curriculum
- Identify key frames for focus
- Segment motion into phases
- Design progressive training stages

---

## Files Generated

```
/home/grease/gam/
├── skeleton_animated_sequence.png      # 24-frame animation grid
├── skeleton_motion_continuous.png      # 3-view + phase decomposition
└── visualize_motion_examples.ipynb     # Code to generate these
```

---

## Summary

The animated skeleton visualizations transform raw joint angle data into intuitive visual representations of motion. By showing:

1. **Sequential frames** - You see the motion flow
2. **Multiple views** - You understand the 3D structure
3. **Phase decomposition** - You understand the motion stages
4. **Color coding** - You quickly identify phases and body parts
5. **Motion energy labels** - You see when motion is most dynamic

This allows you to truly **see** what the robot is doing, rather than just reading numbers.

---

**Generated:** June 17, 2026  
**Motion:** Jump and Land Heavy  
**Frames:** 120 total (30 fps ≈ 4 seconds)  
**Model:** 14-joint skeleton with FK approximation
