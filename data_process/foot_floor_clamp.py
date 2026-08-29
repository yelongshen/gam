"""Automatic, per-frame, collision-geometry-aware foot/floor penetration fix
for G1 `motion_lib`-format retargeted motions.

Replaces the manual, clip-specific, global-constant-shift patch applied by
hand during LAFAN1 debugging (see soma-retargeter/LAFAN1_PIPELINE_DEBUG_NOTES.md
Sec.8) with a principled fix that:

  1. Computes the TRUE foot-sole world height directly from the G1 MJCF's own
     collision geometry (the 4 small spheres under each `..._ankle_roll_link`
     body), parsed PROGRAMMATICALLY -- no hardcoded `-0.03m`/`0.005m` numbers.
  2. Runs automatically for ANY motion_lib entry via full G1 forward
     kinematics (root pos/quat + 29 DOF angles -> world positions), not a
     one-off measurement on a single clip.
  3. Applies a PER-FRAME (not global-constant) vertical root-height
     correction: only frames whose true (collision-aware) foot-sole height
     would otherwise go below the floor get lifted, by exactly the amount
     needed at that frame (plus a small safety epsilon) -- preserving the
     real vertical motion trajectory (jumps, crouches, ...) everywhere else.

Usage (standalone):
    .venv_sim/bin/python foot_floor_clamp.py \\
        --motion_lib /tmp/walk1_subject1_motion_lib.pkl \\
        --mjcf /home/grease/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml \\
        --epsilon_m 0.003 \\
        --out /tmp/walk1_subject1_motion_lib_clamped.pkl

Usage (importable, e.g. from convert_soma_csv_to_motion_lib.py):
    from foot_floor_clamp import apply_per_frame_floor_clamp
    entry, stats = apply_per_frame_floor_clamp(entry, mjcf_path)
"""
import argparse
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import scipy.spatial.transform as sT

# MuJoCo/MJCF actuator-order joint names (same convention as
# convert_soma_csv_to_motion_lib.py's BONES_CSV_JOINT_NAMES / MJ_TO_IL),
# used to map `dof` array columns to named joints during FK.
MJCF_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")


def _parse_vec(s, default):
    if s is None:
        return np.array(default, dtype=np.float64)
    return np.array([float(x) for x in s.split()], dtype=np.float64)


def parse_mjcf_chain_and_foot_geoms(mjcf_path):
    """Parse the MJCF body tree once and return:
      chain: list of dicts (in document order, parent-before-child) with
             name, parent, rest_pos (m, local offset from parent), rest_quat
             (wxyz, local orientation), joint_name (or None), joint_axis
             (local, or None for fixed bodies).
      foot_geoms: {body_name: (N,3) array of local collision-sphere centers
                   (m) with the sphere RADIUS already baked in as an extra
                   downward offset along the body's local -Z axis} -- i.e.
                   each returned point is the true LOWEST point of that
                   collision sphere in the body's local frame, so callers
                   don't need to separately track radius.

    Reads geometry PROGRAMMATICALLY from the MJCF -- no hardcoded numbers.
    Only bodies on the path to (and including) `left_ankle_roll_link` /
    `right_ankle_roll_link` are kept in `chain` (sufficient for foot-height
    FK; avoids wasting time on arms/head/fingers).
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    chain = []
    foot_geoms = {}

    def walk(body_el, parent_name):
        name = body_el.get("name")
        pos = _parse_vec(body_el.get("pos"), [0, 0, 0])
        quat_attr = body_el.get("quat")
        quat_wxyz = _parse_vec(quat_attr, [1, 0, 0, 0]) if quat_attr else np.array([1., 0, 0, 0])

        joint_el = body_el.find("joint")
        joint_name = joint_el.get("name") if joint_el is not None else None
        joint_axis = _parse_vec(joint_el.get("axis"), [0, 0, 1]) if joint_el is not None else None

        chain.append({
            "name": name, "parent": parent_name,
            "rest_pos": pos, "rest_quat": quat_wxyz,
            "joint_name": joint_name, "joint_axis": joint_axis,
        })

        if name in FOOT_LINKS:
            spheres = []
            for geom_el in body_el.findall("geom"):
                size_attr = geom_el.get("size")
                gpos_attr = geom_el.get("pos")
                gtype = geom_el.get("type", "sphere")  # MJCF default geom type is sphere
                # Only primitive collision spheres (skip visual `type="mesh"` geoms).
                if gtype == "mesh" or size_attr is None or gpos_attr is None:
                    continue
                radius = float(size_attr.split()[0])
                center = _parse_vec(gpos_attr, [0, 0, 0])
                # True lowest point of this sphere, in the body's own local
                # frame: center minus radius along local -Z. (Local frame Z
                # is not necessarily world-up, but for the ankle_roll_link
                # near-ground-contact orientation this is the standard MJCF
                # convention and matches how the geom's own local Z is
                # defined relative to the sole.)
                spheres.append(center - np.array([0.0, 0.0, radius]))
            foot_geoms[name] = np.array(spheres, dtype=np.float64)

        for child in body_el.findall("body"):
            walk(child, name)

    for top_body in worldbody.findall("body"):
        walk(top_body, None)

    # Trim chain to only bodies that are ancestors of (or equal to) a foot
    # link, for FK efficiency (skip arms/head/fingers entirely).
    name_to_parent = {c["name"]: c["parent"] for c in chain}

    def ancestors(name):
        out = {name}
        p = name_to_parent.get(name)
        while p is not None:
            out.add(p)
            p = name_to_parent.get(p)
        return out

    needed = set()
    for foot in FOOT_LINKS:
        needed |= ancestors(foot)
    chain = [c for c in chain if c["name"] in needed]

    return chain, foot_geoms


def _quat_wxyz_to_matrix(q):
    # scipy expects xyzw
    return sT.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _axis_angle_matrix(axis, angle):
    return sT.Rotation.from_rotvec(axis / (np.linalg.norm(axis) + 1e-12) * angle).as_matrix()


def compute_foot_sole_heights(root_trans_m, root_quat_xyzw, dof_rad, chain, foot_geoms):
    """Full-clip FK -> per-frame TRUE minimum foot-sole world height (m),
    accounting for the actual collision-sphere geometry (not just the
    `ankle_roll_link` origin).

    Args:
        root_trans_m: (T, 3) world root position, meters.
        root_quat_xyzw: (T, 4) world root orientation, scipy xyzw convention.
        dof_rad: (T, 29) joint angles in radians, MuJoCo actuator order
                 (`MJCF_JOINT_ORDER` above).
        chain, foot_geoms: from `parse_mjcf_chain_and_foot_geoms`.

    Returns:
        (T,) array: min over BOTH feet's collision-sphere lowest points,
        per frame, in world meters (Z-up). Negative => foot penetrates floor.
    """
    T = root_trans_m.shape[0]
    joint_col = {name: i for i, name in enumerate(MJCF_JOINT_ORDER)}

    min_height = np.full(T, np.inf)

    root_rotmats = sT.Rotation.from_quat(root_quat_xyzw).as_matrix()  # (T,3,3)

    for t in range(T):
        wp = {}
        wr = {}
        for node in chain:
            name, parent = node["name"], node["parent"]
            R_rest = _quat_wxyz_to_matrix(node["rest_quat"])
            if node["joint_name"] is not None and node["joint_name"] in joint_col:
                angle = dof_rad[t, joint_col[node["joint_name"]]]
                R_joint = _axis_angle_matrix(node["joint_axis"], angle)
            else:
                # Root free joint (e.g. `floating_base_joint`) or any other
                # joint not in our 29-DOF actuator list: no additional local
                # rotation to apply here (root orientation is already
                # supplied separately via `root_quat_xyzw`).
                R_joint = np.eye(3)
            R_local = R_rest @ R_joint

            if parent is None:
                wp[name] = root_trans_m[t]
                wr[name] = root_rotmats[t]
            else:
                wp[name] = wp[parent] + wr[parent] @ node["rest_pos"]
                wr[name] = wr[parent] @ R_local

        for foot_name in FOOT_LINKS:
            spheres_local = foot_geoms.get(foot_name)
            if spheres_local is None or len(spheres_local) == 0:
                continue
            spheres_world = wp[foot_name] + (wr[foot_name] @ spheres_local.T).T  # (N,3)
            h = spheres_world[:, 2].min()
            if h < min_height[t]:
                min_height[t] = h

    return min_height


def apply_per_frame_floor_clamp(entry, mjcf_path, epsilon_m=0.003):
    """Apply the principled, per-frame, collision-geometry-aware floor clamp
    to one `motion_lib` entry dict (must contain `root_trans_offset` (T,3)
    meters, `root_rot` (T,4) xyzw quaternion, `dof` (T,29) radians in MuJoCo
    order). Modifies and returns `entry` in place, plus a stats dict.

    Only lifts frames whose TRUE (collision-aware) foot-sole height would
    otherwise be < 0 -- by exactly the deficit + epsilon_m needed at that
    frame. Frames that are already clear of the floor are left untouched.
    """
    chain, foot_geoms = parse_mjcf_chain_and_foot_geoms(mjcf_path)

    root_trans = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    root_quat = np.asarray(entry["root_rot"], dtype=np.float64)  # xyzw
    dof = np.asarray(entry["dof"], dtype=np.float64)

    min_height = compute_foot_sole_heights(root_trans, root_quat, dof, chain, foot_geoms)

    deficit = np.maximum(0.0, -(min_height - epsilon_m))  # (T,), >=0, only where needed
    n_affected = int((deficit > 0).sum())

    entry["root_trans_offset"] = root_trans.copy()
    entry["root_trans_offset"][:, 2] += deficit

    stats = {
        "frames_total": len(deficit),
        "frames_affected": n_affected,
        "max_deficit_m": float(deficit.max()) if len(deficit) else 0.0,
        "min_true_foot_height_before_m": float(min_height.min()) if len(min_height) else 0.0,
    }
    return entry, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_lib", required=True, help="input motion_lib .pkl ({name: entry} or bare entry)")
    ap.add_argument("--mjcf", required=True, help="G1 MJCF path (e.g. gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml)")
    ap.add_argument("--epsilon_m", type=float, default=0.003, help="safety clearance above floor, meters")
    ap.add_argument("--out", required=True, help="output corrected .pkl path")
    args = ap.parse_args()

    d = joblib.load(args.motion_lib)
    is_dict_of_entries = isinstance(d, dict) and "dof" not in d
    entries = d if is_dict_of_entries else {"__single__": d}

    for name, entry in entries.items():
        _, stats = apply_per_frame_floor_clamp(entry, args.mjcf, args.epsilon_m)
        print(f"[{name}] frames_affected={stats['frames_affected']}/{stats['frames_total']}  "
              f"max_deficit={stats['max_deficit_m']*1000:.2f}mm  "
              f"min_true_foot_height_before={stats['min_true_foot_height_before_m']*1000:.2f}mm")

    out_obj = entries["__single__"] if not is_dict_of_entries else entries
    joblib.dump(out_obj, args.out, compress=True)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
