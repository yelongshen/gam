#!/usr/bin/env python3
"""
print_pico_smpl.py
====================
Print raw SMPL/body-tracking frames streamed from a PICO VR headset via
XRoboToolkit directly to the terminal (no file output, no visualization).

Usage
-----
    conda run -n xr python print_pico_smpl.py --seconds 15
"""

import argparse
import time

import numpy as np

import xrobotoolkit_sdk as xrt


JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]


def main():
    ap = argparse.ArgumentParser(description="Print PICO SMPL/body-tracking frames live")
    ap.add_argument("--seconds", type=float, default=15.0, help="Duration (s)")
    ap.add_argument("--fps", type=float, default=2.0, help="Print rate (Hz)")
    ap.add_argument("--wait_timeout", type=float, default=20.0)
    ap.add_argument("--joints", type=str, default="Pelvis,L_Wrist,R_Wrist,Head",
                     help="Comma-separated joint names to print each frame")
    args = ap.parse_args()

    wanted = [j.strip() for j in args.joints.split(",")]
    wanted_idx = [JOINT_NAMES.index(j) for j in wanted if j in JOINT_NAMES]

    print("=" * 70)
    print("PICO Live SMPL Body Tracking - Terminal Display")
    print("=" * 70)
    xrt.init()

    print(f"Waiting for body tracking data (timeout={args.wait_timeout}s)...")
    start_wait = time.time()
    while not xrt.is_body_data_available():
        if time.time() - start_wait > args.wait_timeout:
            print("\nERROR: body_available never became True.")
            print("  -> Check PICO app Status=WORKING, Full body ticked,")
            print("     motion trackers calibrated (blue Calibrate button).")
            if hasattr(xrt, "close"):
                xrt.close()
            return
        print("  ...waiting")
        time.sleep(1.0)

    print("Body data available! Streaming (Ctrl+C to stop early)\n")
    header = "time(s)  | " + " | ".join(f"{JOINT_NAMES[i]:>10s} (x,y,z)" for i in wanted_idx)
    print(header)
    print("-" * len(header))

    dt = 1.0 / args.fps
    start_t = time.time()
    n = 0
    try:
        while time.time() - start_t < args.seconds:
            body_poses = np.array(xrt.get_body_joints_pose(), dtype=np.float64)  # (24,7)
            elapsed = time.time() - start_t
            row = f"{elapsed:7.2f}  | "
            row += " | ".join(
                f"({body_poses[i,0]:+.3f},{body_poses[i,1]:+.3f},{body_poses[i,2]:+.3f})"
                for i in wanted_idx
            )
            print(row)
            n += 1
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"\nPrinted {n} frames over {time.time()-start_t:.1f}s")
    if hasattr(xrt, "close"):
        xrt.close()


if __name__ == "__main__":
    main()
