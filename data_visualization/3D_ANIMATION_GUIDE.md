# 3D Skeleton Animation Visualizations Guide

## Overview

This guide documents the comprehensive set of 3D skeleton animations created for visualizing G1 robot motion data from the Bones-Studio dataset.

---

## 📹 Animation Files

### **1. skeleton_3d_animation.gif** (7.9 MB)
**Primary 3D Animation - Single Perspective**

- **Format**: GIF animation (auto-looping)
- **Duration**: 6 seconds (120 frames @ 20 fps)
- **Perspective**: Dynamic rotating camera
  - Camera rotates around the skeleton (45° → 90°+ azimuth)
  - Elevation varies smoothly (±5° oscillation)
  - Provides excellent depth perception
  
- **Features**:
  - Real-time 3D skeleton rendering
  - Full kinematic chain visualization
  - Color-coded body parts:
    - 🔴 **Red**: Left leg (primary motion driver)
    - 🔵 **Cyan**: Right leg (mostly static)
    - 🔵 **Blue**: Left arm (balance support)
    - 🟢 **Green**: Right arm (balance support)
    - 🟠 **Orange**: Torso (vertical axis)
    - 💗 **Pink**: Head
  
  - Live data display:
    - Frame counter (0/119)
    - Motion phase (PREPARATION → LAUNCH → LANDING)
    - Left hip joint angle
    - Right hip joint angle
    - Motion energy value
    - Progress percentage
  
- **Best for**: Understanding 3D motion from single rotating perspective, presentations

---

### **2. skeleton_3d_dual_view.gif** (2.6 MB)
**Dual Perspective 3D Animation**

- **Format**: GIF animation (auto-looping)
- **Duration**: 6 seconds (120 frames @ 20 fps)
- **Left Panel**: Front-right perspective (45° azimuth, 15° elevation)
- **Right Panel**: Pure side view (0° azimuth, 0° elevation)

- **Advantages**:
  - Two simultaneous viewing angles
  - Left view shows depth and 3D structure
  - Right view shows clear side profile motion
  - Perfect for motion analysis and validation
  - Smaller file size (2.6 MB vs 7.9 MB)
  
- **Best for**: Detailed motion analysis, comparing front vs side profiles

---

### **3. stick_figure_motion.gif** (1.2 MB)
**Simplified 2D Stick Figure Animation**

- **Format**: GIF animation (auto-looping)
- **Duration**: 6 seconds (120 frames @ 20 fps)
- **Representation**: 2D stick figure approximation
- **Phase-based coloring**:
  - 🔵 Blue background: PREPARATION (frames 0-39)
  - 🟡 Yellow background: LAUNCH (frames 39-78)
  - 🔴 Red background: LANDING (frames 78-119)

- **Features**:
  - Lightweight and quick to load
  - Clear joint angle display
  - Motion energy tracking
  - Simpler representation
  
- **Best for**: Quick preview, social media sharing, bandwidth-limited scenarios

---

### **4. stick_figure_multiview.gif** (1.2 MB)
**Multi-Perspective Stick Figure**

- **Format**: GIF animation (auto-looping)
- **Duration**: 6 seconds (120 frames @ 20 fps)
- **Three panels**:
  1. **Side Profile**: Front-facing view (XY plane)
  2. **Top-Down View**: Bird's eye perspective (XZ plane)
  3. **Skeletal Structure**: Reference structure view

- **Features**:
  - Three synchronized views
  - Phase tracking
  - Energy display
  - Minimal file size
  
- **Best for**: Understanding multi-angle motion, educational use

---

## 📊 Motion Characteristics Shown

### **Motion Phases**

All animations clearly visualize the three distinct phases of the "jump_and_land_heavy" motion:

#### **Phase 1: PREPARATION (Frames 0-39)**
- Robot crouches down, loading energy
- Left leg compresses (Joint 0: 0° → ~50°)
- Right leg mostly static
- Arms prepare for support
- Duration: ~2 seconds

#### **Phase 2: LAUNCH (Frames 39-78)**
- Explosive upward motion
- Left leg fully extends (Joint 0: ~50° → 119°)
- Peak motion energy around frame 60
- Arms swing up for balance and momentum
- Body rises significantly
- Duration: ~2 seconds

#### **Phase 3: LANDING (Frames 78-119)**
- Impact absorption
- Left leg re-compresses
- Body descends and stabilizes
- Arms return to neutral
- Energy dissipates
- Duration: ~2 seconds

---

## 🎨 Color Scheme Reference

### **Body Parts** (Consistent across all animations)
| Body Part | Color | RGB | Use Case |
|-----------|-------|-----|----------|
| Left Leg | Red | #FF6B6B | Primary motion driver |
| Right Leg | Cyan | #4ECDC4 | Static reference |
| Left Arm | Blue | #0066FF | Balance/support |
| Right Arm | Green | #00CC00 | Balance/support |
| Torso | Orange | #FFA500 | Core structure |
| Head | Pink | #FF69B4 | Reference point |
| Joints | Phase-dependent | - | Current motion phase |

### **Motion Phases** (Phase-based coloring)
| Phase | Color | Background | Frames |
|-------|-------|-----------|--------|
| PREPARATION | Blue | #E6F2FF | 0-39 |
| LAUNCH | Gold | #FFFFF0 | 39-78 |
| LANDING | Red | #FFE6E6 | 78-119 |

---

## 🖥️ How to View Animations

### **Option 1: Direct File Open**
```bash
# On macOS
open skeleton_3d_animation.gif

# On Linux
display skeleton_3d_animation.gif
# or
eog skeleton_3d_animation.gif

# On Windows
# Double-click the .gif file
```

### **Option 2: Web Browser**
- Drag and drop GIF into browser window
- Most modern browsers support GIF playback
- Right-click → "Open image" for full-screen view

### **Option 3: Image Viewers**
- GIMP (advanced editing)
- ImageMagick (`display` command)
- Windows Photos App
- Preview (macOS)
- Feh (Linux)

### **Option 4: In Jupyter Notebook**
```python
from IPython.display import Image, display
display(Image('skeleton_3d_animation.gif'))
```

### **Option 5: Video Conversion**
```bash
# Convert GIF to MP4 (requires ffmpeg)
ffmpeg -i skeleton_3d_animation.gif -pix_fmt yuv420p skeleton_3d_animation.mp4

# Convert GIF to WebM (VP8/VP9 codec)
ffmpeg -i skeleton_3d_animation.gif -c:v libvpx -b:v 1M skeleton_3d_animation.webm

# View MP4 in any video player
vlc skeleton_3d_animation.mp4
```

---

## 📈 Data Specifications

### **Motion Dataset**
- **Motion Name**: `jump_and_land_heavy_001__A001`
- **Capture Date**: 2021-05-31 (210531)
- **Total Frames**: 120
- **Frame Rate**: 20 fps (6 seconds total duration)
- **Robot Model**: G1 (29 DOF)

### **Joint Configuration**
- **Left Leg** (Joints 0-5): 6 DOF
  - Joint 0 (Left Hip): 0° → 119° (primary motion)
  - Joints 1-5: Supporting joints
  
- **Right Leg** (Joints 6-11): 6 DOF
  - Joint 6 (Right Hip): ~-85° (static)
  - Joints 7-11: Supporting joints
  
- **Waist** (Joints 12-14): 3 DOF
  - Minimal motion during jump
  
- **Left Arm** (Joints 15-21): 7 DOF
  - Supporting motion for balance
  
- **Right Arm** (Joints 22-28): 7 DOF
  - Supporting motion for balance

### **Motion Energy Statistics**
- **Mean Energy**: ~1.8 units
- **Peak Energy**: ~4.2 units (frame ~60)
- **Min Energy**: ~1.0 units (frame 0 and 119)
- **Std Dev**: ~0.8 units

---

## 🔍 Interpretation Guide

### **What to Look For in 3D Animation**

1. **Left Leg Motion**
   - Watch the dramatic extension during launch
   - Notice smooth compression during landing
   - This is the primary motion driver

2. **Right Leg**
   - Remains relatively static throughout
   - Provides stability and balance
   - Small supporting movements

3. **Torso and Head**
   - Head remains relatively stable (good balance)
   - Torso follows overall body motion
   - Slight rotation with waist joint

4. **Arm Motion**
   - Arms swing up during launch phase
   - Provide momentum and balance
   - Return to neutral during landing

5. **Camera Rotation** (3D animation only)
   - Helps visualize depth
   - Clarifies 3D structure
   - Multiple angles show motion clearly

---

## 📊 Comparison: 2D vs 3D Visualizations

| Aspect | 2D Stick Figure | 3D Skeleton |
|--------|-----------------|-------------|
| **File Size** | 1.2 MB | 7.9 MB |
| **Load Time** | Fast | Slower |
| **Depth Perception** | Limited | Excellent |
| **Detail Level** | Simplified | Full kinematic chain |
| **Motion Clarity** | Good for simple analysis | Best for complex motion |
| **Best Use** | Quick preview, sharing | Detailed analysis, research |
| **Rendering** | 2D approximation | Full 3D physics-based |

---

## 🎓 Educational Use

### **For Understanding Robotics**
- Observe how joint angles translate to 3D motion
- See the kinematic chain in action
- Understand motion phases and energy flow

### **For Motion Analysis**
- Identify key motion features
- Compare different motion types
- Validate captured motion data

### **For Presentations**
- Use 3D animations for technical talks
- Use 2D for quick overviews
- Include both in comprehensive reports

---

## 🛠️ Technical Details

### **Animation Generation**
- **Library**: Matplotlib with 3D support
- **Backend**: Pillow (PIL) for GIF encoding
- **Resolution**: 1400×1000 pixels (3D) / 1000×1200 pixels (2D)
- **FPS**: 20 frames per second
- **Codec**: GIF (lossless compression)

### **3D Rendering**
- **Projection**: Perspective projection (3D space to 2D screen)
- **Lighting**: No explicit lighting (color-coded visualization)
- **View Init**: Configurable elevation and azimuth angles
- **Axis Labels**: X (lateral), Y (vertical), Z (depth)

### **Performance**
- **Generation Time**: ~11 seconds per animation
- **GIF File Size**: 1.2-7.9 MB depending on complexity
- **Playback**: Real-time in all modern browsers
- **Loop**: Infinite auto-repeat

---

## 📝 Citation

If using these visualizations in research or publications:

```bibtex
@dataset{bones_studio_2021,
  title={Bones-Studio: G1 Robot Motion Capture Dataset},
  author={[Dataset Authors]},
  year={2021},
  url={https://github.com/yelongshen/gam}
}
```

---

## 📞 Support & Troubleshooting

### **GIF Won't Play**
- Try opening in web browser (Firefox, Chrome, Safari)
- Use `ffmpeg` to convert to MP4: `ffmpeg -i file.gif output.mp4`
- Verify file is not corrupted: `file filename.gif`

### **File Size Too Large**
- Use 2D stick figure instead (1.2 MB vs 7.9 MB)
- Convert to MP4 with compression
- Use dual-view version (2.6 MB) as compromise

### **Need Different Format**
- Convert GIF to MP4: `ffmpeg -i input.gif output.mp4`
- Convert GIF to WebM: `ffmpeg -i input.gif output.webm`
- Convert GIF to PNG sequence: `ffmpeg -i input.gif frame_%04d.png`

---

## 📚 Related Files

- `visualize_motion_examples.ipynb` - Jupyter notebook with all visualizations
- `stick_figure_animation_sequence.png` - Static 32-frame grid view
- `skeleton_3d_phases.png` - Static 5-phase view
- `skeleton_3d_multiview.png` - Static multi-angle view
- `motion_timeline_analysis.png` - Motion energy and phase analysis

---

**Last Updated**: June 17, 2026  
**Animation Suite Version**: 2.0 (3D High-Quality)
