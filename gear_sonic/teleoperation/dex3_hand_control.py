#!/usr/bin/env python3
"""
dex3_hand_control.py
====================
Low-level preset-pose control for Unitree Dex3-1 hands.

Sends a named pose directly to rt/dex3/{left,right}/cmd via DDS.

Motor order (hardware): [thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]

URDF joint limits:
  Right: thumb_0 [-1.05,+1.05]  thumb_1 [-1.05,+0.742]  thumb_2 [-1.75, 0.0]
         middle/index_0 [0,+1.57]   middle/index_1 [0,+1.75]
  Left:  thumb_0 [-1.05,+1.05]  thumb_1 [-0.724,+1.05]  thumb_2 [0.0,+1.75]
         middle/index_0 [-1.57, 0]  middle/index_1 [-1.75, 0]

Poses
-----
  open   — thumb spread, all fingers extended
  fist   — thumb tucked, all fingers fully flexed
  pinch  — thumb meets index tip
  zero   — URDF q=0 rest pose
  stop   — release all torque (kp=kd=0, motors go limp)

Usage
-----
  # Send once then exit:
  python dex3_hand_control.py --net eth0 --pose open
  python dex3_hand_control.py --net eth0 --pose fist --side left

  # Interactive: type pose names in a loop:
  python dex3_hand_control.py --net eth0 --interactive

  # Dry-run (no DDS, just print joint targets):
  python dex3_hand_control.py --print-only --pose pinch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "external_dependencies" / "unitree_sdk2_python"))

# ── Motor metadata ───────────────────────────────────────────────────────────
MOTOR_NUM   = 7
MOTOR_NAMES = [
    "thumb_abd", "thumb_mcp", "thumb_ip",
    "mid_mcp",   "mid_pip",
    "idx_mcp",   "idx_pip",
]

# PD gains: [thumb_0, thumb_1, thumb_2, mid_0, mid_1, idx_0, idx_1]
KP = [2.0, 1.5, 1.5, 1.2, 1.2, 1.2, 1.2]
KD = [0.1, 0.08, 0.08, 0.05, 0.05, 0.05, 0.05]

# ── Preset poses ─────────────────────────────────────────────────────────────
# Format: { name: (q_right[7], q_left[7]) }
# Motor order: [thumb_abd, thumb_mcp, thumb_ip, mid_mcp, mid_pip, idx_mcp, idx_pip]
#
# Sign convention (from dex3_hands.hpp URDF limits):
#   Right: fingers flex in + direction; thumb_ip flexes in - direction
#   Left:  fingers flex in - direction; thumb_ip flexes in + direction
#   thumb_abd: negative = spread (for both hands)
POSES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "zero": (
        np.array([ 0.0,    0.0,    0.0,    0.0,    0.0,    0.0,   0.0  ]),  # right
        np.array([ 0.0,    0.0,    0.0,    0.0,    0.0,    0.0,   0.0  ]),  # left
    ),
    "open": (
        np.array([-1.047, -0.920,  0.0,    0.0,    0.0,    0.0,   0.0  ]),  # right
        np.array([-1.047, +0.920,  0.0,    0.0,    0.0,    0.0,   0.0  ]),  # left
    ),
    "fist": (
        np.array([-1.047, -0.920, -1.745,  1.571,  1.745,  1.571, 1.745]),  # right
        np.array([-1.047, +0.920, +1.745, -1.571, -1.745, -1.571,-1.745]),  # left
    ),
    "pinch": (
        np.array([-1.000,  0.400, -1.745,  0.0,    0.0,    1.571, 0.800]),  # right — index meets thumb
        np.array([-1.000, -0.400, +1.745,  0.0,    0.0,   -1.571,-0.800]),  # left  — mirror
    ),
}

POSE_NAMES = list(POSES.keys()) + ["stop"]


# ── DDS helpers ──────────────────────────────────────────────────────────────

def _make_mode(motor_id: int, status: int = 0x01, timeout: int = 0) -> int:
    """Build the mode byte matching Unitree Dex3 protocol."""
    return (motor_id & 0x0F) | ((status & 0x07) << 4) | ((timeout & 0x01) << 7)


def _build_cmd(q: np.ndarray, stop: bool = False):
    """Build a HandCmd_ from the unitree_sdk2py IDL."""
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

    cmd = unitree_hg_msg_dds__HandCmd_()
    for i in range(MOTOR_NUM):
        m = cmd.motor_cmd[i]
        if stop:
            m.mode    = _make_mode(i, status=0x01, timeout=0x01)
            m.q       = 0.0
            m.dq      = 0.0
            m.kp      = 0.0
            m.kd      = 0.0
            m.tau     = 0.0
        else:
            m.mode    = _make_mode(i)
            m.q       = float(q[i])
            m.dq      = 0.0
            m.kp      = KP[i]
            m.kd      = KD[i]
            m.tau     = 0.0
    return cmd


def _print_pose(name: str, q_right: np.ndarray, q_left: np.ndarray, sides: list[str]) -> None:
    print(f"\n  Pose: {name}")
    for side in sides:
        q = q_right if side == "right" else q_left
        vals = "  ".join(f"{MOTOR_NAMES[i]}={q[i]:+.3f}" for i in range(MOTOR_NUM))
        print(f"    {side.upper():5s}: {vals}")


# ── Commander ────────────────────────────────────────────────────────────────

class Dex3Commander:
    """Thin wrapper around unitree_sdk2py publishers for both hands."""

    def __init__(self, network_interface: str) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_

        ChannelFactoryInitialize(0, network_interface)
        self._left_pub  = ChannelPublisher("rt/dex3/left/cmd",  HandCmd_)
        self._right_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self._left_pub.Init()
        self._right_pub.Init()
        print(f"[Dex3] DDS publishers initialised on {network_interface}")

    def send(self, pose_name: str, sides: list[str]) -> None:
        is_stop = (pose_name == "stop")
        if is_stop:
            q_right = q_left = np.zeros(MOTOR_NUM)
        else:
            q_right, q_left = POSES[pose_name]

        if "right" in sides:
            self._right_pub.Write(_build_cmd(q_right, stop=is_stop))
        if "left" in sides:
            self._left_pub.Write(_build_cmd(q_left,  stop=is_stop))

        side_str = "+".join(s.upper() for s in sides)
        print(f"[Dex3] → {pose_name:8s}  ({side_str})")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unitree Dex3-1 low-level preset-pose control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available poses: {', '.join(POSE_NAMES)}",
    )
    ap.add_argument("--net",         default="enp36s0f1",
                    help="Network interface connected to G1 (default: enp36s0f1)")
    ap.add_argument("--pose",        choices=POSE_NAMES, default=None,
                    help="Pose to send once, then exit")
    ap.add_argument("--side",        choices=["left", "right", "both"], default="both",
                    help="Which hand(s) to command (default: both)")
    ap.add_argument("--interactive", action="store_true",
                    help="Interactive loop: type pose names until Ctrl-C")
    ap.add_argument("--print-only",  action="store_true",
                    help="Print joint targets only, do not connect to G1")
    args = ap.parse_args()

    sides = ["left", "right"] if args.side == "both" else [args.side]

    if args.print_only:
        poses_to_show = POSE_NAMES if args.pose is None else [args.pose]
        for name in poses_to_show:
            if name == "stop":
                print("\n  Pose: stop  (kp=kd=0, motors go limp)")
                continue
            q_r, q_l = POSES[name]
            _print_pose(name, q_r, q_l, sides)
        return

    if not args.interactive and args.pose is None:
        ap.error("Specify --pose <name> or --interactive (or --print-only to dry-run)")

    commander = Dex3Commander(network_interface=args.net)

    if args.pose:
        # Single-shot: send the pose a few times to ensure it lands, then exit
        for _ in range(5):
            commander.send(args.pose, sides)
            time.sleep(0.02)
        return

    # Interactive loop
    print(f"\nAvailable poses: {', '.join(POSE_NAMES)}")
    print("Type a pose name and press Enter. Ctrl-C to quit.\n")
    try:
        while True:
            try:
                raw = input("pose> ").strip().lower()
            except EOFError:
                break
            if not raw:
                continue
            if raw not in POSE_NAMES:
                print(f"  Unknown pose '{raw}'. Choose from: {', '.join(POSE_NAMES)}")
                continue
            # Send 5× at 50 Hz for a smooth transition
            for _ in range(5):
                commander.send(raw, sides)
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[Dex3] Sending stop before exit...")
        for _ in range(5):
            commander.send("stop", sides)
            time.sleep(0.02)
        print("[Dex3] Done.")


if __name__ == "__main__":
    main()
