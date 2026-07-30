#!/usr/bin/env python3
"""
capture_pico_smpl.py
=====================
Capture raw SMPL/body-tracking frames streamed from a PICO VR headset via
XRoboToolkit, and save them to disk (.npz) for later offline visualization.

This decouples DATA CAPTURE from RENDERING, so we can verify the PICO
streaming pipeline independently of any display/window issues.

Usage
-----
    python capture_pico_smpl.py --seconds 10 --out /tmp/pico_smpl_capture.npz

Output
------
    .npz file containing:
        body_poses   : (N, 24, 7) - [x,y,z,qx,qy,qz,qw] per joint per frame
        timestamps_ns: (N,)       - body data timestamp in nanoseconds
"""

import argparse
import time

import numpy as np

try:
    import xrobotoolkit_sdk as xrt
except ImportError as e:
    raise SystemExit(
        "xrobotoolkit_sdk not available. Run inside the 'xr' conda env:\n"
        "  conda run -n xr python capture_pico_smpl.py ..."
    ) from e


JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
]


def main():
    ap = argparse.ArgumentParser(description="Capture PICO SMPL/body-tracking frames")
    ap.add_argument("--seconds", type=float, default=10.0, help="Capture duration (s)")
    ap.add_argument("--fps", type=float, default=30.0, help="Sampling rate (Hz)")
    ap.add_argument("--out", type=str, default="/tmp/pico_smpl_capture.npz",
                     help="Output .npz path")
    ap.add_argument("--wait_timeout", type=float, default=30.0,
                     help="Max seconds to wait for body data to become available")
    args = ap.parse_args()

    print("=" * 60)
    print("PICO SMPL Capture")
    print("=" * 60)
    print("Initializing XRoboToolkit SDK...")
    xrt.init()

    print(f"Waiting for body tracking data (timeout={args.wait_timeout}s)...")
    start_wait = time.time()
    while not xrt.is_body_data_available():
        if time.time() - start_wait > args.wait_timeout:
            xrt.close() if hasattr(xrt, "close") else None
            raise SystemExit(
                "ERROR: Body tracking data never became available.\n"
                "Checklist:\n"
                "  1. PICO Motion Trackers strapped on and PAIRED\n"
                "  2. Calibration sequence completed (blue Calibrate button,\n"
                "     stand still -> look at feet)\n"
                "  3. XRoboToolkit app shows Status: WORKING\n"
                "  4. 'Full body' ticked under Tracking in the app"
            )
        print("  ...waiting for body data")
        time.sleep(1.0)

    print("Body tracking data available! Capturing frames...")
    print(f"Duration: {args.seconds}s at {args.fps} Hz")

    body_poses_list = []
    timestamps_list = []

    dt = 1.0 / args.fps
    start_t = time.time()
    n_frames = 0
    last_report = start_t

    while time.time() - start_t < args.seconds:
        frame_start = time.time()

        body_poses = np.array(xrt.get_body_joints_pose(), dtype=np.float64)  # (24, 7)
        ts_ns = xrt.get_body_timestamp_ns()

        body_poses_list.append(body_poses)
        timestamps_list.append(ts_ns)
        n_frames += 1

        if time.time() - last_report > 2.0:
            elapsed = time.time() - start_t
            print(f"  [{elapsed:5.1f}s] captured {n_frames} frames "
                  f"(pelvis pos={body_poses[0, :3].round(3).tolist()})")
            last_report = time.time()

        # Maintain sampling rate
        elapsed_frame = time.time() - frame_start
        if elapsed_frame < dt:
            time.sleep(dt - elapsed_frame)

    body_poses_arr = np.stack(body_poses_list, axis=0)   # (N, 24, 7)
    timestamps_arr = np.array(timestamps_list, dtype=np.int64)  # (N,)

    print(f"\nCaptured {n_frames} frames.")
    print(f"body_poses shape: {body_poses_arr.shape}")

    # Quick sanity check: did the pelvis position actually change across frames?
    pelvis_positions = body_poses_arr[:, 0, :3]
    pos_std = pelvis_positions.std(axis=0)
    print(f"Pelvis position std over capture (x,y,z): {pos_std.round(4).tolist()}")
    if pos_std.max() < 1e-5:
        print("WARNING: Pelvis position did not change at all across the capture.")
        print("         This may indicate the headset was stationary, or that")
        print("         data is frozen/not actually updating.")

    np.savez(
        args.out,
        body_poses=body_poses_arr,
        timestamps_ns=timestamps_arr,
        joint_names=np.array(JOINT_NAMES),
    )
    print(f"\nSaved capture to: {args.out}")

    if hasattr(xrt, "close"):
        xrt.close()


if __name__ == "__main__":
    main()
