# Animation Titles Updated with Source and Data Path Information

## Summary of Changes

Updated all skeleton motion animations to clearly display the data source and file path in their titles.

---

## Animation Information

### 1. Single-View Corrected 3D Skeleton Animation
**File**: `skeleton_3d_corrected_animation.gif` (7.9 MB)

**Title Now Shows**:
```
3D Motion Visualization - Energy-Based Skeleton (Correct)
Motion: jump_and_land_heavy_001__A001 | Source: G1 Robot (29 DOF)
Path: /home/grease/ego_dataset/work_bearlu/data/bones-studio-seed/g1/csv/210531/jump_and_land_heavy_001__A001.csv
```

**Features**:
- Full motion capture (120 frames)
- 6 seconds duration @ 20fps
- Rotating camera perspective
- Energy-based height offset
- Single synchronized view

---

### 2. Multi-View 3D Skeleton Animation  
**File**: `skeleton_3d_multiview_animation.gif` (1.2 MB)

**Title Now Shows**:
```
3D Skeleton Motion - Multi-View Animation
Motion: jump_and_land_heavy_001__A001 | Source: G1 Robot (29 DOF)
Path: /home/grease/ego_dataset/work_bearlu/data/bones-studio-seed/g1/csv/210531/jump_and_land_heavy_001__A001.csv
```

**Features**:
- 4 synchronized views (Front, Side, Isometric, Top-Down)
- 32 frames sampled from full motion
- 1.6 seconds duration @ 20fps
- All views display source and data path information

---

## Data Source Information

### Motion Data Details
- **Source**: G1 Robot (Humanoid Biped)
- **DOF**: 29 Degrees of Freedom
  - Left leg: 6 DOF (angles 0-5)
  - Right leg: 6 DOF (angles 6-11)
  - Waist: 3 DOF (angles 12-14)
  - Left arm: 7 DOF (angles 15-21)
  - Right arm: 1 DOF (angles 22)
  - Additional joints: angles 23-28

- **Motion Name**: `jump_and_land_heavy_001__A001`
- **Motion Type**: Jump with heavy landing
- **Duration**: 120 frames @ 20fps = 6 seconds
- **Dataset Path**: `/home/grease/ego_dataset/work_bearlu/data/bones-studio-seed/`

### Data Structure
```
/home/grease/ego_dataset/work_bearlu/data/bones-studio-seed/
├── g1/csv/                           # G1 Robot joint angles (THIS SOURCE)
│   └── 210531/
│       └── jump_and_land_heavy_001__A001.csv
├── soma_uniform/bvh/                 # SOMA human generic body
│   └── [not used in current visualization]
└── soma_proportional/bvh/            # SOMA human actor-specific body
    └── [not used in current visualization]
```

---

## Title Components Explained

Each animation title now includes:

1. **Visualization Type**
   - "3D Motion Visualization" or "3D Skeleton Motion"
   - Indicates the rendering method

2. **Motion Name**
   - `jump_and_land_heavy_001__A001`
   - Identifies the specific motion capture

3. **Data Source**
   - "Source: G1 Robot (29 DOF)"
   - Specifies the robot platform and joint count
   - Alternative sources: SOMA Uniform, SOMA Proportional

4. **Full Data Path**
   - Complete file path to CSV data
   - Allows direct access to original data
   - Format: `/path/to/data/source/date/motion_name.csv`

---

## Why This Information Matters

### For Research & Documentation
- Reproducibility: Readers can locate exact motion data
- Attribution: Credit to data source (G1 Robot platform)
- Traceability: Date and motion type clearly visible

### For Analysis
- Identify which robot/body model generated the motion
- Understand DOF constraints (29 DOF for G1)
- Access raw data for further processing

### For Data Management
- Quick verification of data source when working with multiple datasets
- Prevents confusion between G1 (robot), SOMA Uniform (generic human), and SOMA Proportional (actor-specific)

---

## Animation Viewing Instructions

### Viewing the Animations
```bash
# Single-view animation
open skeleton_3d_corrected_animation.gif

# Multi-view animation  
open skeleton_3d_multiview_animation.gif

# In Python
from PIL import Image
img = Image.open('skeleton_3d_multiview_animation.gif')
img.show()
```

### Interpreting the Titles
When you see an animation, read the title from top to bottom:
1. **First line**: What you're looking at (visualization type)
2. **Second line**: Which motion (with data source)
3. **Third line**: Where the original data is stored (file path)

---

## Technical Implementation

### Title Format in Code

**Single-View Animation** (Cell #VSC-aa07ffda):
```python
fig.suptitle(
    f'3D Motion Visualization - Energy-Based Skeleton (Correct)\n'
    f'Motion: {sample_motion_name} | Source: G1 Robot (29 DOF)\n'
    f'Path: {matching_g1[0] if matching_g1 else "N/A"}',
    fontsize=13, fontweight='bold'
)
```

**Multi-View Animation** (Cell #VSC-435ab893):
```python
fig_mv_updated.suptitle(
    f"3D Skeleton Motion - Multi-View Animation\n"
    f"Motion: {sample_motion_name} | Source: G1 Robot (29 DOF)\n"
    f"Path: {matching_g1[0] if matching_g1 else 'N/A'}",
    fontsize=13, fontweight='bold', y=0.995
)
```

### Variables Used
- `sample_motion_name`: Motion identifier from dataset
- `matching_g1[0]`: Full file path to G1 CSV data
- Fallback: "N/A" if file path unavailable

---

## Files Updated

1. **skeleton_3d_corrected_animation.gif**
   - Regenerated with source info in title
   - Size: 7.9 MB
   - Quality: Lossless GIF format

2. **skeleton_3d_multiview_animation.gif**
   - Regenerated with source info in title
   - Size: 1.2 MB
   - Quality: Optimized 4-view format

3. **visualize_motion_examples.ipynb**
   - Cell #VSC-aa07ffda: Updated single-view animation title
   - Cell #VSC-93d3c88d: Updated multi-view animation title definition
   - Cell #VSC-435ab893: New cell with complete multi-view animation generation

---

## Git Commit

```
7325411 feat: Add data source and file path information to animation titles
```

**Changes**:
- `skeleton_3d_corrected_animation.gif` (updated)
- `skeleton_3d_multiview_animation.gif` (updated)
- `visualize_motion_examples.ipynb` (2 cells modified, 1 cell added)

---

## Data Source References

### G1 Robot Platform
- **Type**: Humanoid biped robot
- **DOF**: 29 total degrees of freedom
- **Application**: Motion capture and analysis
- **Dataset**: bones-studio-seed collection

### Motion Capture Details
- **Motion**: Jump with heavy landing (001 variant, actor A001)
- **Frame Count**: 120 frames
- **Sample Rate**: 20 fps
- **Duration**: 6 seconds
- **Encoding**: CSV format with 29 joint angle columns

---

## Future Enhancements

- [ ] Add motion description and performance metrics to title
- [ ] Include capture date and location information
- [ ] Add links to data source documentation
- [ ] Support for multiple data sources in single comparison
- [ ] Create data catalog with source verification

