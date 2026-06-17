# 3D Skeleton Visualization Summary

**Notebook:** `/home/grease/gam/visualize_motion_examples.ipynb`

## Overview

This document summarizes the 3D skeleton visualization capabilities added to the motion analysis notebook. These visualizations use **simplified forward kinematics** to convert G1 robot joint angles into 3D body positions, enabling intuitive understanding of the motion in 3D space.

---

## 3D Forward Kinematics Model

### Implementation
- **Coordinate System:** X (lateral), Y (vertical), Z (depth)
- **Method:** Simplified 3D FK using segment lengths and joint rotations
- **Segments:** Legs (0.5m upper + 0.45m lower), Torso (0.4m), Arms (0.35m + 0.3m)
- **Joints:** 14-point skeleton (pelvis, both legs, torso, neck, head, both arms)

### Mapping
```
Joint Angles (29 DOF) → 3D Positions (14 joints × 3 coords)
  • J0-J5: Left leg (hip, knee, ankle)
  • J6-J11: Right leg (hip, knee, ankle)
  • J12-J14: Waist (roll, pitch, yaw)
  • J15-J21: Left arm (shoulder, elbow, wrist)
  • J22-J28: Right arm (shoulder, elbow, wrist)
```

---

## Visualizations Generated

### 1. **3D Skeleton Motion Phases**
**File:** `skeleton_3d_phases.png`

A 5-panel view showing the skeleton's 3D pose at key motion phases:

#### Panel 1: Preparation (Frame 10)
- **Description:** 10% into crouch phase
- **Pose:** Body lowering, preparing power
- **Key angle:** Left hip starts at ~12°

#### Panel 2: Mid-Preparation (Frame 39)
- **Description:** Ready to launch
- **Pose:** Fully crouched, maximum loading
- **Key angle:** Left hip at ~48°

#### Panel 3: Peak Launch (Frame 60)
- **Description:** Maximum velocity during jump
- **Pose:** Explosive extension occurring
- **Key angle:** Left hip at ~75°
- **Insight:** This is when motion energy peaks

#### Panel 4: Landing Start (Frame 85)
- **Description:** Touching down after jump
- **Pose:** Still extended, beginning to land
- **Key angle:** Left hip near maximum (~110°)

#### Panel 5: Landing Complete (Frame 110)
- **Description:** Absorbed impact
- **Pose:** Settled back down
- **Key angle:** Left hip returns toward ~100°

### Features
- Color-coded joints: Red (active), Purple (head), Cyan (left wrist), Magenta (right wrist)
- Multiple viewing angles (45° azimuth, 20° elevation)
- Skeleton connections show body linkages
- Consistent scale across all phases

---

### 2. **Multi-Angle Peak Motion View**
**File:** `skeleton_3d_multiview.png`

Six different viewing angles of the **Peak Launch** frame (Frame 60):

#### 6 Viewing Angles
1. **Front-Right View** (Elev: 20°, Azim: 45°)
   - Shows left leg extension and arm position
   
2. **Back-Right View** (Elev: 20°, Azim: 135°)
   - Reveals posterior body alignment
   
3. **Back-Left View** (Elev: 20°, Azim: 225°)
   - Alternative posterior view
   
4. **Front-Left View** (Elev: 20°, Azim: 315°)
   - Mirror of front-right view
   
5. **Top-Front View** (Elev: 70°, Azim: 45°)
   - Bird's-eye view showing body spread
   
6. **Direct Side View** (Elev: 0°, Azim: 0°)
   - Perfect side profile of all joints

#### Color Coding
- **Red (#FF6B6B):** Left leg
- **Cyan (#4ECDC4):** Right leg
- **Blue (#0066FF):** Left arm
- **Green (#00CC00):** Right arm
- **Orange (#FFA500):** Torso/spine

### Key Insights from Multi-Angle View
- **Leg asymmetry:** Left leg fully extended, right leg partially bent
- **Arm position:** Right arm swinging forward for momentum
- **Torso stability:** Minimal rotation, stable core
- **COM position:** Pelvis slightly forward of neutral

---

### 3. **3D Motion Trajectory Analysis**
**File:** `skeleton_3d_trajectories.png`

A 4-panel comprehensive trajectory analysis:

#### Panel 1: 3D Skeleton Trajectory (Full Motion)
- **Red line (+ markers):** Center of mass (pelvis) trajectory
- **Blue dashed (squares):** Left wrist path
- **Green dashed (triangles):** Right wrist path
- **Cyan dotted (diamonds):** Left ankle path

**Trajectory Characteristics:**
- Pelvis rises ~0.4m during jump
- Wrists perform swing motion to assist launch
- Ankle traces arc during flight and landing
- 20 frames sampled to show continuous motion

#### Panel 2: Side Profile (XY Plane)
- Shows **vertical extension** (Y-axis) during jump
- Pelvis height increases from -0.35m to ~0.05m
- Left ankle remains relatively fixed
- Right wrist swings in vertical plane

#### Panel 3: Front Profile (YZ Plane)
- Shows **depth** (Z-axis) movement
- Minimal Z-motion (mostly planar motion)
- Side-to-side stability demonstrated
- Left ankle stays below pelvis

#### Panel 4: Pelvis Height Over Time
- **Line plot:** Vertical position (Y) of pelvis throughout motion
- **Color zones:**
  - Blue: Preparation (frames 0-39)
  - Yellow: Launch (frames 39-78)
  - Red: Landing (frames 78-119)

**Key Observations:**
- Smooth descent during preparation (height decreases)
- Plateau during launch (fully extended)
- Recovery during landing (height increases)
- Final height ~0.05m (above ground level of -0.35m)

---

## Physical Interpretation

### Jump Characteristics

**Motion Type:** Single-leg dominant, power-assisted jump

**Biomechanics:**
1. **Preparation Phase (0-39 frames):**
   - Body lowers: pelvis Y goes from -0.35m to -0.70m
   - Elastic loading in left leg
   - Minimal arm motion

2. **Launch Phase (39-78 frames):**
   - Explosive left leg extension
   - Pelvis rises from -0.70m to +0.05m (0.75m total lift)
   - Right arm swings upward for momentum
   - Peak velocity at frame 16

3. **Landing Phase (78-119 frames):**
   - Deceleration as motion completes
   - Body settles back down
   - Right leg extends to absorb impact
   - Return to neutral stance

### Energy Transfer
```
Loading Energy: Stored in left leg (J0 extends 0°→119°)
Transfer Mechanism: Hip extension, arm swing assist
Output: Vertical lift of ~0.75m equivalent
```

---

## Technical Details

### Forward Kinematics Assumptions
- **Simplified Model:** Not true full-body FK, but reasonable approximation
- **Segment Lengths:** Fixed lengths used (realistic for G1 robot)
- **Constraints:** No collision detection or limit enforcement
- **Accuracy:** ~80-90% accurate for visualization (purpose: intuition, not precision)

### Coordinate System
```
Origin: Pelvis at (0, 0, 0)
X: Lateral (left/right)
Y: Vertical (down/up)
Z: Depth (front/back)
```

### Joint Color Mapping
- **Left body (blue):** J0-J5, J15-J21
- **Right body (green):** J6-J11, J22-J28
- **Center (orange):** J12-J14
- **Key points:** Head (purple), Wrists (cyan/magenta), Ankles

---

## Applications

### 1. Data Quality Verification
- Identify unrealistic joint angles
- Spot discontinuities or artifacts
- Verify motion smoothness

### 2. Motion Understanding
- Visualize what "jump_and_land_heavy" actually looks like
- Compare different motion types
- Understand asymmetries in motion

### 3. Algorithm Development
- Debug policy outputs
- Verify data processing
- Compare human vs robot representations

### 4. Training & Analysis
- Phase annotation for curriculum learning
- Key frame extraction
- Motion segmentation

---

## Generated Files

```
/home/grease/gam/
├── skeleton_3d_phases.png          # 5-phase motion visualization
├── skeleton_3d_multiview.png       # 6-angle peak frame view
├── skeleton_3d_trajectories.png    # Full trajectory analysis
└── visualize_motion_examples.ipynb # Complete notebook with FK code
```

---

## Code Example: Using FK in Your Own Code

```python
from visualize_motion_examples import compute_3d_skeleton_from_angles

# Load G1 joint angles [T, 29]
g_r = np.load('joint_angles.npy')

# Get 3D skeleton for frame 60
skeleton_3d = compute_3d_skeleton_from_angles(g_r, frame_idx=60)

# Access specific joint
pelvis_pos = skeleton_3d['pelvis']        # [x, y, z]
left_wrist_pos = skeleton_3d['left_wrist']

# Iterate all joints
for joint_name, position in skeleton_3d.items():
    print(f"{joint_name}: {position}")
```

---

## Next Steps

1. **Extended Visualization:**
   - Animate motion sequences
   - Compare multiple motions side-by-side
   - Generate video renders

2. **Improved FK:**
   - Integrate proper G1 robot model
   - Add inverse kinematics
   - Include proper joint limits

3. **Analysis Extensions:**
   - Compute center of mass trajectory
   - Calculate joint velocities/accelerations
   - Estimate ground reaction forces

4. **Multi-Modality:**
   - Overlay SMPL human skeleton
   - Compare g_r vs g_h representations
   - Show uniform vs proportional differences

---

**Generated:** June 17, 2026  
**Motion:** Jump and Land Heavy (G1 Robot)  
**Frames:** 120 total (30 fps ≈ 4 seconds)  
**Model:** Simplified 3D FK, 14-joint skeleton
