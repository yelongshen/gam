# Raw AMASS Data Format

Reference for the **raw** AMASS `.npz` files as downloaded/extracted (i.e.
*before* `normalize_split_test.py` converts them into the `smpl_filtered`
representation documented in `SMPL_FILTERED_DATA_FORMAT.md`).

- **Location:** `/home/grease/egodata/downloads/amass/extracted/<DATASET>/...`
- **Loader:** `np.load(path, allow_pickle=True)`
- **Consumed by:** `normalize_split_test.load_clip()`,
  `classify_motions.load_joints()`

Every statement below was verified empirically against real files; the checks
are reproduced inline so they can be re-run.

---

## 1. There are TWO distinct file schemas in this dataset

AMASS was reprocessed over time (older MoSh output vs. the newer MoSh++/SOMA
"stageii" pipeline, which also switched body models from SMPL-H to SMPL-X).
**Both schemas are present side-by-side** in the extracted archive, and code
that assumes only one of them will silently mishandle the other.

### 1.1 Schema A — older SMPL-H clips (no `_stageii` suffix)

Example: `ACCAD/Male2Walking_c3d/B5 -  Walk backwards_poses.npz`

```text
KEYS: ['trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses']
trans             (975, 3)    float64
gender            = male
mocap_framerate   = 120.0
betas             (16,)       float64
dmpls             (975, 8)    float64
poses             (975, 156)  float64
```

### 1.2 Schema B — newer SMPL-X clips (`_stageii` suffix)

Example: `CMU/CMU/123/123_07_stageii.npz`

```text
KEYS: ['gender', 'surface_model_type', 'mocap_frame_rate', 'mocap_time_length',
       'markers_latent', 'latent_labels', 'markers_latent_vids',
       'trans', 'poses', 'betas', 'num_betas',
       'root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye']
gender               = neutral
surface_model_type   = smplx
mocap_frame_rate     = 120.0
mocap_time_length    = 13.0
trans                (1560, 3)   float64
poses                (1560, 165) float64
betas                (16,)       float64
num_betas            = 16
root_orient          (1560, 3)
pose_body            (1560, 63)
pose_hand            (1560, 90)
pose_jaw             (1560, 3)
pose_eye             (1560, 6)
```

### 1.3 Scope in our current test split

```text
AMASS clips total in split_test.csv : 1516
  '_stageii' (schema B)             :  978  (64.5%)
  other      (schema A)             :  538  (35.5%)
```

So **the majority of AMASS clips use schema B**, not schema A.

---

## 2. 🐛 BUG #1 — framerate key mismatch silently corrupts playback speed

Both `normalize_split_test.load_clip()` and `classify_motions.load_joints()`
read the source framerate like this:

```python
src = float(d['mocap_framerate']) if 'mocap_framerate' in d.files else 60.0
```

This key name (**no underscore**) only exists in **schema A**. Schema B
(`_stageii`, 64.5% of our AMASS clips) stores the framerate under
**`mocap_frame_rate`** (**with an underscore**) instead:

```text
'mocap_framerate'  in schema B files? -> False
'mocap_frame_rate' in schema B files? -> True   (= 120.0 for the example above)
```

Since the lookup key doesn't match, the code silently falls through to the
hardcoded default of **60.0 fps** for every schema-B clip — regardless of
the file's true rate.

### Verified impact

```text
mocap_frame_rate (true)      : 120.0
mocap_time_length            : 13.0 s
poses.shape[0]                : 1560
implied fps (frames / time)   : 1560 / 13.0 = 120.0   (confirms 120.0 is correct)

Code's assumed fps (bug)      : 60.0   (hardcoded fallback)
Correct downsample stride     : round(120/50) = 2   (keep every 2nd frame)
Bug's downsample stride       : round(60 /50) = 1   (keep every frame!)
```

**Consequence:** every schema-B clip (978 of 1516 AMASS test clips, 64.5%)
is resampled with the wrong stride and is streamed at **2.4× the intended
speed** for this example (ratio = true_fps / assumed_fps = 120/60 = 2×,
compounding with the `round(120/50)` vs `round(60/50)` stride difference).
The motion is not corrupted in *content*, only in *timing* — but that
timing error means the policy is tracking a stream running much faster
than the human actually performed it.

### The fix

Check both key spellings, preferring whichever is present:

```python
if 'mocap_frame_rate' in d.files:
    src = float(d['mocap_frame_rate'])
elif 'mocap_framerate' in d.files:
    src = float(d['mocap_framerate'])
else:
    src = 60.0  # last-resort fallback only
```

---

## 3. The `poses` array layout depends on the schema

`poses` is the concatenation of the model's per-joint axis-angle parameters,
but **the concatenation order differs by schema** because schema B
(SMPL-X) has extra jaw/eye parameters that schema A (SMPL-H) doesn't.

### 3.1 Schema A (156 = 3 + 63 + 90)

| Slice | Field | Dim |
|---|---|---|
| `poses[:, 0:3]` | root orientation | 3 |
| `poses[:, 3:66]` | body pose (21 joints) | 63 |
| `poses[:, 66:156]` | hand pose (both hands) | 90 |

(No jaw/eye — SMPL-H has no face model.)

### 3.2 Schema B (165 = 3 + 63 + 3 + 6 + 90)

Verified by exact array comparison against the named fields:

```python
poses[:, 0:3]    == root_orient   -> True
poses[:, 3:66]   == pose_body     -> True
poses[:, 66:69]  == pose_jaw      -> True   (NOT hand!)
poses[:, 69:75]  == pose_eye      -> True   (NOT hand!)
poses[:, 75:165] == pose_hand     -> True
```

| Slice | Field | Dim |
|---|---|---|
| `poses[:, 0:3]` | root orientation | 3 |
| `poses[:, 3:66]` | body pose (21 joints) | 63 |
| `poses[:, 66:69]` | jaw pose | 3 |
| `poses[:, 69:75]` | eye pose (L+R) | 6 |
| `poses[:, 75:165]` | hand pose (both hands) | 90 |

### 3.3 🐛 BUG #2 — `poses[:, :72]` means different things in each schema

Our pipeline's 24-joint SMPL FK (`classify_motions.smpl_fk`,
`fix_amass.smpl_fk`) consumes exactly **72 values** = 24 joints × 3
(root + 21 body joints + 2 more "joints" worth of data at indices 66:72,
representing the reduced-24-joint model's L/R hand root).

`normalize_split_test.load_clip()` currently does:

```python
return poses[::step, :72].astype(np.float64), ...
```

This slice's semantic content **depends entirely on the schema**:

| Schema | `poses[:, 66:72]` actually contains |
|---|---|
| A (156-dim) | first 2 joints of **pose_hand** — correct data for FK joints 22/23 |
| B (165-dim, `_stageii`) | **pose_jaw + first half of pose_eye** — face params, NOT hand data |

For schema-B clips (the majority), the FK's last two "joints" (indices 22,
23 — L/R hand in the reduced 24-joint model) are silently being fed **jaw
and eye rotation values** instead of hand rotation. Since these are leaf
joints far out on the kinematic chain from a tracking-fidelity standpoint,
the positional error this introduces is small, but it is nonetheless
feeding the encoder physically meaningless data for those two joints on
978 of our 1516 AMASS test clips.

### The fix

Slice the fields by name rather than a fixed byte offset, e.g.:

```python
if 'pose_hand' in d.files:      # schema B
    pose72 = np.concatenate([d['root_orient'], d['pose_body'],
                              d['pose_hand'][:, :6]], axis=1)
else:                            # schema A
    pose72 = d['poses'][:, :72]
```

---

## 4. `trans` — global root translation (consistent across schemas)

`trans` is `(T, 3)` float64 in both schemas, and is the world-space pelvis
translation — same role as `smpl_filtered`'s `transl` field (see
`SMPL_FILTERED_DATA_FORMAT.md` §4).

---

## 5. `betas` — body shape parameters

`(16,)` float64 in both schemas (16 SMPL/SMPL-X shape coefficients).
`num_betas` (schema B only) simply records `len(betas)`.

---

## 6. Summary of bugs found here

| # | Bug | Scope | Symptom |
|---|---|---|---|
| 1 | `mocap_framerate` vs `mocap_frame_rate` key mismatch | 64.5% of AMASS test clips (schema B) | motion streamed at wrong (faster) speed |
| 2 | `poses[:, :72]` slice assumes schema A layout | 64.5% of AMASS test clips (schema B) | FK joints 22/23 (hands) fed jaw/eye data instead |

Both bugs are in **timing/secondary-joint data only** — they do NOT affect
the root-rotation canonicalization bug documented separately (large root
rotation causing instant falls), which was already fixed in
`normalize_split_test.canonicalize_root_rotation()`. These two are
independent issues affecting AMASS ingestion specifically. **Both are now
fixed** in `convert_amass_to_smpl_filtered.py` (see §8 below) — `load_clip()`
in `normalize_split_test.py` still has the original (unfixed) code path for
the legacy AMASS route and should eventually be reconciled with the new
converter.

---

## 7. Reproducing the verification

```python
import numpy as np

# Schema A
dA = np.load('/home/grease/egodata/downloads/amass/extracted/ACCAD/'
             'Male2Walking_c3d/B5 -  Walk backwards_poses.npz', allow_pickle=True)
print(list(dA.files))                       # no 'mocap_frame_rate', no 'pose_hand' etc.

# Schema B
dB = np.load('/home/grease/egodata/downloads/amass/extracted/CMU/CMU/123/'
             '123_07_stageii.npz', allow_pickle=True)
print(list(dB.files))
print(dB['mocap_frame_rate'], dB['poses'].shape[0] / dB['mocap_time_length'])  # both 120.0

poses = dB['poses']
print(np.allclose(poses[:, 66:69], dB['pose_jaw']))    # True  <- NOT hand data
print(np.allclose(poses[:, 75:165], dB['pose_hand']))  # True  <- hand is here instead
```

---

## 8. Conversion pipeline (`.npz` -> `smpl_filtered`-style `.pkl`)

Implemented in `data_process/convert_amass_to_smpl_filtered.py`, run against
**all 17,892 raw AMASS `.npz` files** (16,758 after filtering out
shape/stagei-only files with no motion data):

1. **Schema-aware parse -> `pose72` (T, 72).** Detect schema via
   `'pose_hand' in d.files`; build the 72-dim root+body slice correctly for
   each schema (fixes Bug #2 above):
   ```python
   if schema == 'B':
       pose72 = np.concatenate([d['root_orient'], d['pose_body'],
                                 d['pose_hand'][:, :6]], axis=1)
   else:
       pose72 = d['poses'][:, :72]
   ```
2. **Schema-aware framerate detection** (fixes Bug #1 above): check
   `mocap_frame_rate` (schema B) then `mocap_framerate` (schema A), only
   falling back to a hardcoded `60.0` if neither key is present.
3. **Resample source fps -> 50 fps** via linear interpolation of `pose72`
   and `trans` (upsamples OR downsamples correctly, unlike a stride-only
   `[::step]` decimation which cannot upsample).
4. **Canonicalize frame-0 root rotation** — but with a **simpler** method
   than LAFAN/BVH: since AMASS `pose_aa` is *already* native SMPL axis-angle
   (no BVH-vs-SMPL rest-frame convention mismatch to correct for), we
   directly zero the root joint's own rotation:
   ```python
   R0 = axis_angle_to_matrix(pose_aa[0, :3])
   D = R0.T                      # D @ R0 = Identity
   ```
   This is applied uniformly to `pose_aa[:, :3]` (via left-multiplication by
   `D`, all frames) and to `trans` (`trans @ D.T`).

   **Why not reuse the LAFAN skeleton-geometry approach (up = Neck-Pelvis,
   right = L_hip-R_hip)?** Verified empirically to be *wrong* for AMASS:
   the neck-pelvis vector passes through the spine joint chain (`spine1`,
   `spine2`, `spine3`), each of which has its own `pose_aa` rotation that
   varies per clip/frame (e.g. leaning while typing). Using it as a proxy
   for "root orientation" gets contaminated by that spine posture, giving
   inconsistent, clip-dependent leftover rotation at frame 0 (measured
   0.03 rad on a walking clip vs. 0.54 rad on a desk/typing clip) instead of
   an exact zero. The direct `R0 = axis_angle_to_matrix(pose_aa[0,:3])`
   approach gives **exactly** `pose_aa[0,:3] = [0,0,0]` on every clip,
   verified on a 300-clip random sample of the full converted set.
5. **Derive `smpl_joints` via FK, KEEPING root rotation** (per
   `SMPL_FILTERED_DATA_FORMAT.md` §5.2 — real `smpl_filtered` does **not**
   heading-canonicalize the whole body, only pins the pelvis translation):
   ```python
   joints = smpl_fk(pose_aa, translation=0) @ ZUP.T          # keep root rot, Z-up remap
   smpl_joints = joints - joints[:, 0:1, :] + PELVIS_OFFSET  # pin the pelvis
   ```
6. **Save** `{pose_aa, transl, smpl_joints, fps, original_pose_aa,
   original_fps}` — the full field set from `SMPL_FILTERED_DATA_FORMAT.md`
   §1, including provenance fields.

### Scale + collision handling

Basenames collide across sub-datasets: **1,904 of 17,892** raw `.npz` files
share a basename with at least one other file elsewhere in the tree (e.g.
multiple subjects each with their own `walk_poses.npz`). The converter
builds a collision-free output name from the path *relative to* `--input`
(`os.sep` -> `__`), not `os.path.basename()` alone.

Parallelized with `multiprocessing.Pool` (8 workers by default).

### Verified result

```text
Converted : 16,755 / 16,758  (3 skipped: too few frames to resample to 50 fps)
Failed    : 0
Runtime   : ~280s (~60 clips/s, 8 workers)

Sanity check (random sample, n=300):
  max joint-to-pelvis distance <= 1.6 m  : 300/300 pass
  pose_aa[0, :3] exactly [0,0,0]         : 300/300 pass
```
