# How g_m (Mixed Representation) Data is Obtained

## Overview
The **g_m** (mixed representation) is a synchronized combination of:
- **Upper-body**: 3-point VR/PICO tracking (head + two wrists) extracted from the human motion
- **Lower-body**: Robot joint state from the retargeted robot motion

This creates a hybrid representation that simulates what a real VR teleoperation system would capture.

---

## Data Sources

### 1. **VR Component** (from g_h — human SMPL motion)
Extracts 3D positions of three key points from the original SMPL human motion:

| Joint | SMPL Index | Description | Dimensions |
|-------|-----------|-------------|-----------|
| Head | 15 | Head position | 3 (x, y, z) |
| Left Wrist | 20 | Left hand position | 3 (x, y, z) |
| Right Wrist | 21 | Right hand position | 3 (x, y, z) |
| **Total VR Features** | — | — | **9 dimensions** |

**Extraction from g_h**:
```python
# g_h is [T, 72] SMPL motion (24 joints × 3 coords)
head_pos = g_h[:, 15*3:(15+1)*3]      # Indices 45-47
l_wrist = g_h[:, 20*3:(20+1)*3]       # Indices 60-62
r_wrist = g_h[:, 21*3:(21+1)*3]       # Indices 63-65
# Result: [T, 9]
```

### 2. **Lower-Body Component** (from g_r — robot motion)
Extracts leg and waist joint states from the retargeted G1 robot motion:

| Component | G1 Robot Joints | Count |
|-----------|-----------------|-------|
| Left leg | 6 joints | 6 |
| Right leg | 6 joints | 6 |
| Waist | 1 joint | 1 |
| **Selected joints** | [14,15,16,17,18,19,20,21,22,23,24,25,28] | 13 |

**Extraction from g_r**:
```python
# g_r is [T, 29] robot joint angles
lower_body_joints = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28]
lower_body = g_r[:, lower_body_joints]  # [T, 13]
```

### 3. **Dimensionality Reduction**
The 13-dimensional lower-body data is compressed to 2 dimensions to create the final 11-dim representation:

```python
# Compression strategy:
if lower_body.shape[1] > 2:
    # Average all joints → [T, 1]
    lower_compressed = lower_body.mean(axis=1, keepdims=True)
    # Append last joint → [T, 2]
    lower_compressed = np.concatenate([
        lower_compressed,
        lower_body[:, -1:],  # Keep final waist joint
    ], axis=1)
```

**Result**: [T, 2] compressed lower body representation

---

## Final Composition: g_m

The final **g_m** is a concatenation of both components:

```python
g_m = np.concatenate([
    vr_features,      # [T, 9]  — head + L_wrist + R_wrist positions
    lower_compressed  # [T, 2]  — compressed lower-body state
], axis=1)
# Final shape: [T, 11]
```

### Semantic Breakdown:
| Indices | Content | Description |
|---------|---------|-------------|
| 0-2 | Head position | Head (x, y, z) |
| 3-5 | Left wrist position | L-Wrist (x, y, z) |
| 6-8 | Right wrist position | R-Wrist (x, y, z) |
| 9 | Lower-body average | Mean of all lower-body joints |
| 10 | Waist joint | Final waist joint angle |

---

## Why This Design?

### Purpose
**g_m** represents what a **real VR teleoperation system** would receive:
- Users wear VR trackers on head and wrists
- Lower-body motion is controlled by the planner/locomotion policy
- This is a practical constraint: VR tracking on the lower-body is expensive/unreliable

### Design Rationale
1. **VR 3-point tracking** (9 dims):
   - Directly from human motion (g_h)
   - Represents what IMU/tracker sensors would capture
   - No additional processing needed

2. **Lower-body compression** (2 dims):
   - Takes the already-retargeted robot motion (g_r)
   - Compresses 13 joints to 2 "features" (mean + last joint)
   - Serves as a summary of lower-body state for the network

3. **Asymmetry is intentional**:
   - Upper-body detailed (9 dims) → high teleoperation control
   - Lower-body abstract (2 dims) → locomotion is semi-autonomous
   - Mimics real deployment where upper-body follows user precisely

---

## Processing Pipeline

```
Raw Data
  ├─ g_h [T, 72] (SMPL human motion)
  ├─ g_r [T, 29] (Retargeted G1 motion)
  └─ (No new capture needed — purely derived!)
      ↓
MixedRepresentationBuilder.build_mixed_representation()
      ↓
g_m [T, 11] (Mixed representation)
```

### No External VR Capture Needed
Unlike some other approaches, **g_m is computed offline** during data preprocessing:
- Start with human motion (g_h) ✓ Already available
- Use retargeted robot motion (g_r) ✓ Already available
- Extract and combine → **g_m** ✓ Done!

This makes it **deterministic and reproducible** across all training runs.

---

## Summary Table

| Aspect | Detail |
|--------|--------|
| **Name** | Mixed representation (g_m) |
| **Dimensions** | [T, 11] where T = num frames |
| **Source** | Derived from g_h + g_r (no new capture) |
| **VR component** | Head & wrist positions (9 dims) from g_h |
| **Lower-body** | Compressed leg/waist state (2 dims) from g_r |
| **Creation time** | Offline, during data preprocessing |
| **Purpose** | Simulates VR teleoperation input for training |
| **Usage** | Input to E_m (mixed encoder) → z_m latent |
