"""Stage 3 with the robot's footsteps at real scale and no foot sliding.

GMR scales the human about the world origin: the root by human_scale_table["pelvis"] and every
other target's offset from the root by its own entry (0.9 x the height ratio for the legs, 0.8
for the arms; 0.83 effective on 20261020_060016). The robot then climbs 0.83 x the real stairs.
Changing only the root entry moves the root along a longer path with the same joint angles, so
planted feet slide (measured: 9.4 vs 5.8 cm/s). A planted foot stays planted only if the root
and the foot get the SAME scale, so here:

  pelvis, hips, knees, feet   x U about the origin, U = --scale (e.g. 0.970: real footsteps,
                              undoing PICO's ~3%), so footsteps and step heights are real
  every target but the feet   lowered by one constant per clip, c = (U - s0) x the median height
                              of the pelvis above the lower foot (s0 = the stock scale), so the
                              pelvis sits as high above the feet as in the stock retarget -- the
                              robot's legs are shorter than the human's at scale U
  torso and arms              the stock offsets from the root (GMR's own table)

Writes the same raw GMR .pkl as scripts/retarget_warm_start.py (yaw_normalize + settle 30, as in
build_picoset_*.sh); GMR itself is not modified.

  (on grease, GMR conda env; replaces stage 3 of build_picoset_*.sh)
  python data_process/gmr_retarget_real_scale.py --src <smplx_npz_dir> --tgt <out_dir> \\
      --scale 0.970 --ref-scale 0.830
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, "/home/grease/GMR")
sys.path.insert(0, "/home/grease/GMR/scripts")
import general_motion_retargeting.motion_retarget as mr
import retarget_warm_start as rws

LEGS = ("pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_foot", "right_foot")
FEET = ("left_foot", "right_foot")
STATE = {"U": None, "c": 0.0}
_stock = mr.GeneralMotionRetargeting.scale_human_data


def scale_human_data(self, human_data, human_root_name, human_scale_table):
    out = _stock(self, human_data, human_root_name, human_scale_table)    # stock: torso, arms
    s0 = human_scale_table[human_root_name]
    U = STATE["U"]
    root0 = np.asarray(human_data[human_root_name][0])
    shift = (U - s0) * root0                       # stock root -> real-scale root
    res = {}
    for name, (pos, quat) in out.items():
        pos = np.asarray(pos, dtype=float)
        if name in LEGS:
            pos = U * np.asarray(human_data[name][0], dtype=float)
        else:
            pos = pos + shift                      # torso/arms ride with the real-scale root
        if name not in FEET:
            pos = pos - np.array([0.0, 0.0, STATE["c"]])
        res[name] = (pos, quat)
    return res


mr.GeneralMotionRetargeting.scale_human_data = scale_human_data
_load = rws.get_smplx_data_offline_fast


def load_and_set_offset(*args, **kwargs):
    frames, fps = _load(*args, **kwargs)
    pel = np.array([f["pelvis"][0][2] for f in frames])
    foot = np.array([min(f["left_foot"][0][2], f["right_foot"][0][2]) for f in frames])
    STATE["h"] = float(np.median(pel - foot))
    return frames, fps


rws.get_smplx_data_offline_fast = load_and_set_offset
_GMR = rws.GMR


def make_gmr(*args, **kwargs):
    g = _GMR(*args, **kwargs)
    s0 = g.human_scale_table[g.human_root_name]
    STATE["U"] = STATE["scale_eff"] * s0 / STATE["ref"] if STATE["ref"] else STATE["scale_eff"]
    STATE["c"] = (STATE["U"] - s0) * STATE["h"]
    print(f"  stock scale {s0:.4f} -> {STATE['U']:.4f}, pelvis lowered {STATE['c'] * 100:.1f} cm", flush=True)
    return g


rws.GMR = make_gmr

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--tgt", required=True)
ap.add_argument("--scale", type=float, required=True, help="effective scale of the robot's footsteps")
ap.add_argument("--ref-scale", type=float, default=None,
                help="effective scale the stock config gives on this data (robot root path vs human "
                     "pelvis path); omit if --scale is meant as a raw table value")
ap.add_argument("--clips", nargs="*")
a = ap.parse_args()
STATE["scale_eff"], STATE["ref"] = a.scale, a.ref_scale
os.makedirs(a.tgt, exist_ok=True)
clips = a.clips or sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(a.src, "*.npz")))
for c in clips:
    rws.retarget_file(os.path.join(a.src, c + ".npz"), os.path.join(a.tgt, c + ".pkl"), yaw_normalize=True, settle=30)
    print("done", c, flush=True)
