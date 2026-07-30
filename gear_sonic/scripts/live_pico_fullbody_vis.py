#!/usr/bin/env python3
"""
live_pico_fullbody_vis.py
===========================
Real-time full-body (24-joint SMPL-style skeleton) visualization driven by
live PICO VR body-tracking data via XRoboToolkit.

Unlike --vr3pt_realtime (which only shows 3 points: L-Wrist, R-Wrist, Head),
this renders the full 24-joint skeleton with bones, using the same
VR3PtPoseVisualizer.update_smpl_joints() used for mock data.

Usage
-----
    conda run -n xr python live_pico_fullbody_vis.py --hz 15 --duration 120
"""

import argparse
import time

import numpy as np

import xrobotoolkit_sdk as xrt

from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import VR3PtPoseVisualizer


JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]


def pico_body_to_local_joints(body_poses: np.ndarray) -> np.ndarray:
    """
    Convert raw PICO body_poses (24,7)=[x,y,z,qx,qy,qz,qw] into (24,3) joint
    positions relative to the Pelvis (root), matching the format expected by
    VR3PtPoseVisualizer.update_smpl_joints().

    PICO joint order already matches our JOINT_NAMES / 24-joint SMPL-like scheme.
    """
    positions = body_poses[:, :3]  # (24, 3)
    root = positions[0].copy()
    local = positions - root  # make relative to pelvis
    return local


def main():
    ap = argparse.ArgumentParser(description="Live full-body SMPL visualization from PICO")
    ap.add_argument("--hz", type=float, default=15.0, help="Update rate (Hz)")
    ap.add_argument("--duration", type=float, default=120.0, help="Total run duration (s)")
    ap.add_argument("--wait_timeout", type=float, default=20.0)
    args = ap.parse_args()

    print("=" * 60)
    print("Live PICO Full-Body (24-joint) SMPL Visualization")
    print("=" * 60)

    xrt.init()
    print(f"Waiting for body tracking data (timeout={args.wait_timeout}s)...")
    start_wait = time.time()
    while not xrt.is_body_data_available():
        if time.time() - start_wait > args.wait_timeout:
            raise SystemExit("ERROR: body tracking data never became available.")
        time.sleep(0.5)
    print("Body data available! Launching visualizer...\n")

    visualizer = VR3PtPoseVisualizer(
        with_g1_robot=False,
        enable_smpl_vis=True,
        smpl_root_position=np.array([0.0, 0.0, 0.0]),
    )
    visualizer.create_realtime_plotter(interactive=True, with_reference_frames=True)

    identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dt = 1.0 / args.hz
    start_t = time.time()
    n = 0
    try:
        while visualizer.is_open and time.time() - start_t < args.duration:
            body_poses = np.array(xrt.get_body_joints_pose(), dtype=np.float64)  # (24,7)

            joints_local = pico_body_to_local_joints(body_poses)  # (24,3)

            # 3-point pose for the VR markers (L-Wrist=idx20, R-Wrist=idx21, Head=idx15)
            vr_3pt_pose = np.vstack([
                np.concatenate([joints_local[20], identity_quat]),
                np.concatenate([joints_local[21], identity_quat]),
                np.concatenate([joints_local[15], identity_quat]),
            ])

            visualizer.update_smpl_joints(joints_local)
            visualizer.update_from_vr_pose(vr_3pt_pose)
            visualizer.render()

            n += 1
            if n % int(max(1, args.hz) * 3) == 0:
                elapsed = time.time() - start_t
                print(f"  [{elapsed:6.1f}s] frame {n} | "
                      f"pelvis(world)={body_poses[0,:3].round(3).tolist()} | "
                      f"head(local)={joints_local[15].round(3).tolist()}")

            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        visualizer.close()
        if hasattr(xrt, "close"):
            xrt.close()
        print(f"\nDone. Rendered {n} frames.")


if __name__ == "__main__":
    main()
