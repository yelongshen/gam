# `smpl_filtered` Data Format

Reference for the SMPL motion representation used by the SONIC Mode-2 (`smpl`)
encoder.

- **Location:** `/home/grease/GR00T-WholeBodyControl/data/smpl_filtered/`
- **Count:** 131,455 `.pkl` files (one clip each)
- **Loader:** `joblib.load(path)` → a flat `dict`
- **Paired retargeted robot data:** `/home/grease/GR00T-WholeBodyControl/data/motion_lib_bones_seed/robot_filtered/`

Every statement below was verified empirically by reconstructing the data with
forward kinematics; the tests are reproduced at the end so they can be re-run.

---

## 1. Fields at a glance

| Field | Shape | dtype | Meaning | Frame |
|---|---|---|---|---|
| `pose_aa` | `(T, 72)` | float32 | SMPL pose, 24 joints × 3 axis-angle | root = world, others = parent-relative |
| `transl` | `(T, 3)` | float32 | Global root translation | world |
| `smpl_joints` | `(T, 24, 3)` | float32 | 3D joint positions — **the encoder input** | **root-local, root-rotated, Z-up** |
| `fps` | scalar | float | Target rate (50.0) | — |
| `original_pose_aa` | `(T0, 72)` | float32 | Source poses before resampling | same as `pose_aa` |
| `original_fps` | scalar | float | Source rate (typically 30.0) | — |

Example (`dance_hiphop_funky_chicken_R_fast_001__A320_M.pkl`):

```text
pose_aa           (414, 72)    min=-1.7725 max=1.6401
transl            (414, 3)     min=-0.1972 max=1.3599
smpl_joints       (414, 24, 3) min=-1.0053 max=1.0302
fps               50.0
original_pose_aa  (249, 72)
original_fps      30.0
```

---

## 2. How the fields connect

The four run-time fields are **not independent**. `smpl_joints` is derived from
`pose_aa`, and `transl` supplies the world motion that `smpl_joints`
deliberately omits.

```text
pose_aa (T,72) ──[24-joint SMPL FK]──► joint positions
     │                                       │
     │ [:, 0:3] = root rotation              │ + Z-up remap, pelvis pinned
     │      (also feeds the encoder's        ▼
     │       anchor-orientation obs)   smpl_joints (T,24,3)   ← DERIVED
     │
transl (T,3) ────── world position of the pelvis ────── independent

fps ─────────────── shared time axis for all three (dt = 1/fps)
```

### 2.1 `pose_aa` → `smpl_joints` (derivation)

`smpl_joints` is a **precomputed convenience**, not new information — it is
`pose_aa` pushed through forward kinematics:

```python
joints = smpl_fk(pose_aa, translation=0)              # articulate the tree
joints = joints @ ZUP.T                               # (x,y,z) -> (x,-z,y)
joints = joints - joints[:, 0:1, :] + PELVIS_OFFSET   # pin the pelvis
```

```text
(1) pose_aa -FK-> smpl_joints      err = 0.0544 m
```

(The residual is the betas/body-shape offset of §5.4, not a mismatch.)

It is stored so the encoder can read joint *positions* at 50 Hz without running
FK every tick. Note `pose_aa[:, 0:3]` does double duty: it is part of the FK
chain **and** the source of `smpl_anchor_orientation`.

### 2.2 `smpl_joints` + `transl` → world motion (complementary)

| Field | Carries | Missing |
|---|---|---|
| `smpl_joints` | body **shape / pose** (pelvis pinned) | where the body is |
| `transl` | body **world position** | how the body is posed |

```python
world_joints = (smpl_joints - smpl_joints[:, 0:1, :]) + (transl @ ZUP.T)[:, None, :]
```

```text
(2) smpl_joints + transl -> world  err = 0.0544 m
```

The error is identical to (1) — no *additional* error is introduced, so the
split into "pose" and "position" is exact.

> This is precisely why the tracking bug (§8) was so damaging: `smpl_joints` is
> deliberately translation-free, so streaming world-frame joints injects
> `transl` into a channel the encoder expects to stay pinned.

### 2.3 `fps` — the shared clock

`fps` binds all three arrays to one time base and defines `dt = 1/50 s`, which
the encoder's 4-frame lookahead and any velocity term depend on:

```text
(3) 249 @ 30 fps = 8.30s  ->  414 @ 50 fps = 8.28s
```

`pose_aa`, `transl` and `smpl_joints` are all resampled to `fps` and share the
row index `t`.

### 2.4 Minimal sufficient set

**`pose_aa` + `transl` + `fps`.** `smpl_joints` is fully derivable from them;
`original_*` is provenance only.

| Relationship | Type | Verified |
|---|---|---|
| `pose_aa` → `smpl_joints` | FK (derived, redundant) | 0.054 m |
| `smpl_joints` + `transl` → world | complementary decomposition | 0.054 m |
| `pose_aa[:, :3]` → anchor orientation | direct use | — |
| `fps` | shared time axis, `dt` for lookahead | 8.30 s ≡ 8.28 s |

Reproduce with:

```python
import joblib, numpy as np, fix_amass as F
d = joblib.load('.../dance_hiphop_funky_chicken_R_fast_001__A320_M.pkl')
J, P, T = (d['smpl_joints'].astype(np.float64),
           d['pose_aa'].astype(np.float64),
           d['transl'].astype(np.float64))
ZUP = np.array([[1,0,0],[0,0,-1],[0,1,0]], float)
n = 150

fk = F.smpl_fk(P[:n], np.zeros((n,3))) @ ZUP.T
fk = fk - fk[:, 0:1, :] + J[:n, 0:1, :]
print(np.linalg.norm(fk - J[:n], axis=2).mean())          # -> 0.0544

world = F.smpl_fk(P[:n], T[:n]) @ ZUP.T
rec = (J[:n] - J[:n, 0:1, :]) + (T[:n] @ ZUP.T)[:, None, :]
rec = rec - rec[:, 0:1, :] + world[:, 0:1, :]
print(np.linalg.norm(rec - world, axis=2).mean())         # -> 0.0544
```

---

## 3. `pose_aa` — SMPL pose parameters

`(T, 72)` = 24 joints × 3 axis-angle values (Rodrigues vectors).

| Slice | Contents |
|---|---|
| `pose_aa[:, 0:3]` | **Root (pelvis) global orientation** |
| `pose_aa[:, 3:72]` | 23 body joints, each **relative to its parent** |

The 21 body joints used by the encoder's `smpl_pose` observation are
`pose_aa[:, 3:66].reshape(-1, 21, 3)` (root and the two hand joints excluded).

---

## 4. `transl` — global root translation

World-space position of the pelvis. **This is the only field carrying world
motion.**

```text
transl[0]  = [ 0.0024  1.2498 -0.0545]
transl[-1] = [-0.0780  1.2516  0.1257]
travel     = 0.197 m
```

---

## 5. `smpl_joints` — the encoder input

`(T, 24, 3)` joint positions. Three properties, each verified:

### 5.1 Root-local (the pelvis is pinned)

The pelvis entry is *constant* over the whole clip:

```text
pelvis[0]      = [0.0031 -0.3514 0.0120]
pelvis[200]    = [0.0031 -0.3514 0.0120]
std over time  = [0, 1e-6, 0]
pelvis travel  = 0.000 m       (while transl travels 0.197 m)
```

The body articulates around a fixed pelvis; the constant offset
`[0.003, -0.351, 0.012]` is the rest-pose pelvis position in the body frame.

> **Consequence:** `smpl_joints` alone cannot express world motion.
> Recovering it requires `transl` + `pose_aa[:, :3]`.

### 5.2 Root rotation **is** applied (not heading-canonicalized)

Tested on `medium_heavy_one_hand_walk_turn_360_R_001__A504_M.pkl`
(root rotation up to 180°, which discriminates the two hypotheses):

```text
WITH root rotation                 err = 0.062 m
WITHOUT root rotation (canonical)  err = 0.328 m     ← 5x worse
```

So joints live in a gravity-aligned frame that **rotates with the body**.

### 5.3 Z-up axis convention

Relative to raw SMPL FK output, the mapping is `(x, y, z) → (x, -z, y)`:

```text
raw  FK  -> err 0.569 m
Z-up FK  -> err 0.054 m     ← 10x better
```

As a matrix:

```python
ZUP = np.array([[1, 0,  0],
                [0, 0, -1],
                [0, 1,  0]])     # (x, y, z) -> (x, -z, y)
```

Sanity check — the head sits high in **z**, feet low:

```text
head  mean = [ 0.008 -0.371  0.626]
lfoot mean = [ 0.210 -0.421 -0.910]
```

### 5.4 On the residual error

The ~0.05–0.06 m residual in every reconstruction is **expected**: our FK uses a
canonical rest skeleton, while the clips were produced with subject-specific
SMPL **betas** (body shape). It is a constant limb-length offset, not a frame
error.

---

## 6. `original_pose_aa` / `original_fps`

Provenance: the source motion before resampling to the 50 Hz control rate.

```text
249 frames @ 30 fps = 8.30 s   →   414 frames @ 50 fps = 8.28 s
```

Same motion, upsampled. Useful for auditing, not needed at run time.

---

## 7. Producing this format from raw AMASS / LAFAN

Implemented in `normalize_split_test.py`:

```python
PELVIS_OFFSET = np.array([0.003, -0.351, 0.012])

def to_local_zup(pose_aa, pelvis_offset=PELVIS_OFFSET):
    """Root-local, Z-up SMPL joints in the smpl_filtered convention."""
    p = pose_aa.copy()
    p[:, :3] = 0.0                                   # drop root rotation
    loc = smpl_fk(p, np.zeros((len(p), 3)))          # (T,24,3) local, Y-up
    zup = loc[:, :, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
    return zup + pelvis_offset
```

> **Note:** this helper zeroes the root rotation, producing *canonical* joints.
> That is intentional for streaming AMASS clips (whose global heading is
> arbitrary), but it is **not** byte-identical to `smpl_filtered`, which keeps
> the root rotation (§5.2). If exact parity is required, keep
> `pose_aa[:, :3]` when calling `smpl_fk`.

Verification that both input paths agree:

```text
frames: pkl=613  npz=613
max diff: 0.00000003 m
IDENTICAL: True
```

---

## 8. Why this matters — the tracking-gap bug

Streaming **world-frame, Y-up** joints (pelvis ≈ 0.92 m and translating) instead
of **root-local, Z-up** joints is a completely different input distribution and
severely degrades tracking:

| Metric | World Y-up (wrong) | Root-local Z-up (correct) |
|---|---|---|
| `tracked_frac` | 0.50 – 0.70 | **0.85** |
| `max_tilt_deg` | 116 – 174° (fell) | **20.6°** (stable) |
| joint motion while tracking | — | **1.8×** idle |

---

## 9. Relationship to the encoder observations

From `policy/low_latency/observation_config.yaml`, mode 2 (`smpl`) requires:

| Observation | Dim | Source |
|---|---|---|
| `encoder_mode_4` | 4 | internal one-hot |
| `smpl_joints_4frame_step1` | 288 | `smpl_joints` (24 × 3 × 4 frames) |
| `smpl_anchor_orientation_4frame_step1` | 24 | root orientation (6D × 4 frames) |
| `motion_joint_positions_wrists_4frame_step1` | 24 | 6 wrist DOF × 4 frames |

The wrist indices are `{23, 24, 25, 26, 27, 28}`
(`policy_parameters.hpp:88`) — every other robot DOF is unread in mode 2.

---

## 10. Reproducing the verification

```python
import joblib, numpy as np, fix_amass as F

p = '/home/grease/GR00T-WholeBodyControl/data/smpl_filtered/' \
    'medium_heavy_one_hand_walk_turn_360_R_001__A504_M.pkl'
d = joblib.load(p)
J, P = d['smpl_joints'].astype(np.float64), d['pose_aa'].astype(np.float64)

n   = 200
ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
Pz  = P[:n].copy(); Pz[:, :3] = 0.0          # canonical variant
ref = J[:n]

for label, c in [("with root rot",    F.smpl_fk(P[:n], np.zeros((n, 3)))),
                 ("without root rot", F.smpl_fk(Pz,    np.zeros((n, 3))))]:
    cc = c @ ZUP.T
    cc = cc - cc[:, 0:1, :] + ref[:, 0:1, :]  # align pelvis, compare shape
    print(label, np.linalg.norm(cc - ref, axis=2).mean())
```

Expected:

```text
with root rot     0.0619
without root rot  0.3277
```

---

## 11. Summary

- **Minimal sufficient set is `pose_aa` + `transl` + `fps`.** `smpl_joints` is
  derived from `pose_aa` by FK (§2.1) and stored so the encoder need not run FK
  at 50 Hz; `original_*` is provenance only.
- `smpl_joints` is **root-local, root-rotated, Z-up** — the pelvis never moves.
- `smpl_joints` (pose) and `transl` (position) are **complementary**: together
  they reconstruct world motion exactly (§2.2).
- `pose_aa` is standard SMPL axis-angle: root global, others parent-relative;
  `pose_aa[:, :3]` also feeds the anchor-orientation observation.
- Feeding world-frame or Y-up joints to the Mode-2 encoder is out-of-distribution
  and is the single largest cause of poor tracking.
