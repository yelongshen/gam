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
    # thumb_wave: sweep through abd→mcp→ip to verify each thumb joint independently
    # Fingers stay open; only thumb moves.
    "thumb_open": (
        np.array([-1.047, -0.920,  0.0,   0.0, 0.0, 0.0, 0.0]),   # right thumb fully spread
        np.array([-1.047, +0.920,  0.0,   0.0, 0.0, 0.0, 0.0]),   # left  thumb fully spread
    ),
    "thumb_close": (
        np.array([ 0.0,   -0.920, -1.745,  0.0, 0.0, 0.0, 0.0]),   # right thumb folded in
        np.array([ 0.0,   +0.920, +1.745,  0.0, 0.0, 0.0, 0.0]),   # left  thumb folded in
    ),
}

POSE_NAMES = list(POSES.keys()) + ["stop", "recalibrate", "thumb_wave"]

# How long to hold zero-torque (limp) before re-engaging during recalibration
RECALIB_LIMP_S  = 2.0   # seconds motors stay limp — user straightens fingers
RECALIB_RAMP_S  = 5.0   # seconds to ramp to open after re-engagement


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

CTRL_HZ = 50
CTRL_DT = 1.0 / CTRL_HZ


class Dex3Commander:
    """Publishes interpolated pose transitions for both Dex3 hands."""

    def __init__(self, network_interface: str) -> None:
        import threading
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__HandState_ as HandState_default)

        ChannelFactoryInitialize(0, network_interface)

        self._left_pub  = ChannelPublisher("rt/dex3/left/cmd",  HandCmd_)
        self._right_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self._left_pub.Init()
        self._right_pub.Init()

        self._lock = threading.Lock()
        self._q_state_left    = None
        self._q_state_right   = None
        self._tau_state_left  = None   # actual motor torque
        self._tau_state_right = None
        self._q_cmd_left    = np.zeros(MOTOR_NUM)
        self._q_cmd_right   = np.zeros(MOTOR_NUM)

        self._left_sub = ChannelSubscriber("rt/dex3/left/state",  HandState_)
        self._right_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)
        self._left_sub.Init(
            lambda msg: self._on_state(msg, is_left=True),  1)
        self._right_sub.Init(
            lambda msg: self._on_state(msg, is_left=False), 1)

        print(f"[Dex3] DDS initialised on {network_interface}")
        # Brief wait for first state messages
        time.sleep(0.3)

    def _on_state(self, msg, is_left: bool) -> None:
        q   = np.array([msg.motor_state[i].q   for i in range(MOTOR_NUM)])
        tau = np.array([msg.motor_state[i].tau  for i in range(MOTOR_NUM)])
        with self._lock:
            if is_left:
                self._q_state_left   = q
                self._tau_state_left = tau
            else:
                self._q_state_right   = q
                self._tau_state_right = tau

    def _current_cmd(self, is_left: bool) -> np.ndarray:
        with self._lock:
            return (self._q_cmd_left if is_left else self._q_cmd_right).copy()

    def _current_q(self, is_left: bool) -> np.ndarray:
        with self._lock:
            q = self._q_state_left if is_left else self._q_state_right
        return q.copy() if q is not None else np.zeros(MOTOR_NUM)

    def _current_tau(self, is_left: bool) -> np.ndarray:
        with self._lock:
            t = self._tau_state_left if is_left else self._tau_state_right
        return t.copy() if t is not None else np.zeros(MOTOR_NUM)

    def send(self, pose_name: str, sides: list[str],
             duration: float = 2.0) -> None:
        """Linearly interpolate from current state to target pose over `duration` seconds."""
        is_stop = (pose_name == "stop")
        if is_stop:
            q_right = q_left = np.zeros(MOTOR_NUM)
        else:
            q_right, q_left = POSES[pose_name]

        steps    = max(1, int(round(duration * CTRL_HZ)))
        side_str = "+".join(s.upper() for s in sides)
        print(f"[Dex3] → {pose_name:8s}  ({side_str})  {duration:.1f}s  ({steps} steps)")

        q_start = {
            "right": self._current_q(is_left=False),
            "left":  self._current_q(is_left=True),
        }
        q_target = {"right": q_right, "left": q_left}

        for step in range(steps + 1):
            alpha = step / steps
            for side in sides:
                q = (1.0 - alpha) * q_start[side] + alpha * q_target[side]
                pub = self._left_pub if side == "left" else self._right_pub
                pub.Write(_build_cmd(q, stop=is_stop))
                with self._lock:
                    if side == "left":
                        self._q_cmd_left = q.copy()
                    else:
                        self._q_cmd_right = q.copy()
            time.sleep(CTRL_DT)

    def recalibrate(self, sides: list[str]) -> None:
        """Reinitialize the Dex3 hand.

        Step 1 — LIMP: send zero-torque (timeout=0x01, kp=kd=0) for
                        RECALIB_LIMP_S seconds so joints go compliant.
                        Manually straighten all fingers to rest position now.
        Step 2 — RAMP: slowly re-engage position control and ramp to 'open'
                        over RECALIB_RAMP_S seconds from wherever joints land.
        """
        side_str = "+".join(s.upper() for s in sides)

        # ── Step 1: go limp ────────────────────────────────────────────────
        limp_steps = int(round(RECALIB_LIMP_S * CTRL_HZ))
        print(f"[Dex3] recalibrate ({side_str}) — step 1/{limp_steps*CTRL_DT:.0f}s: "
              f"motors LIMP for {RECALIB_LIMP_S:.0f}s — straighten fingers now...")
        for _ in range(limp_steps):
            for side in sides:
                pub = self._left_pub if side == "left" else self._right_pub
                pub.Write(_build_cmd(np.zeros(MOTOR_NUM), stop=True))
            time.sleep(CTRL_DT)

        # ── Step 2: wait for state to settle, then ramp to open ────────────
        time.sleep(0.1)   # let state subscriber update from new joint positions
        q_open_r, q_open_l = POSES["open"]
        q_start = {
            "right": self._current_q(is_left=False),
            "left":  self._current_q(is_left=True),
        }
        q_target = {"right": q_open_r, "left": q_open_l}

        ramp_steps = max(1, int(round(RECALIB_RAMP_S * CTRL_HZ)))
        print(f"[Dex3] recalibrate ({side_str}) — step 2: ramping to OPEN "
              f"over {RECALIB_RAMP_S:.0f}s ({ramp_steps} steps)...")
        for step in range(ramp_steps + 1):
            alpha = step / ramp_steps
            for side in sides:
                q = (1.0 - alpha) * q_start[side] + alpha * q_target[side]
                pub = self._left_pub if side == "left" else self._right_pub
                pub.Write(_build_cmd(q, stop=False))
            time.sleep(CTRL_DT)

        print(f"[Dex3] recalibrate ({side_str}) — done.")

    def thumb_wave(self, sides: list[str], duration: float = 3.0) -> None:
        """Sweep each thumb joint in sequence to verify motion on hardware.

        Sequence (fingers stay open throughout):
          1. thumb_open  — spread thumb out
          2. thumb_close — fold thumb in (abd→neutral, ip full flex)
          3. thumb_open  — return to spread
        """
        side_str = "+".join(s.upper() for s in sides)
        print(f"[Dex3] thumb_wave ({side_str}) — open → close → open, {duration:.1f}s each")
        for pose in ("thumb_open", "thumb_close", "thumb_open"):
            self.send(pose, sides, duration=duration)


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
                    help="Pose to send once, then exit. Use 'recalibrate' to reinit the hand.")
    ap.add_argument("--side",        choices=["left", "right", "both"], default="both",
                    help="Which hand(s) to command (default: both)")
    ap.add_argument("--duration",    type=float, default=3.0,
                    help="Transition duration in seconds (default: 3.0)")
    ap.add_argument("--joint",       choices=MOTOR_NAMES, default=None,
                    help="Test a single joint by name (use with --angle)")
    ap.add_argument("--angle",       type=float, default=None,
                    help="Target angle in radians for --joint test")
    ap.add_argument("--kp-scale",    type=float, default=1.0,
                    help="Scale all KP gains by this factor for --joint test (default: 1.0)")
    ap.add_argument("--interactive", action="store_true",
                    help="Interactive loop: type pose names until Ctrl-C")
    ap.add_argument("--print-only",  action="store_true",
                    help="Print joint targets only, do not connect to G1")
    ap.add_argument("--monitor",     action="store_true",
                    help="Print live joint state (no commands sent). Ctrl-C to stop.")
    args = ap.parse_args()

    sides = ["left", "right"] if args.side == "both" else [args.side]

    if args.print_only:
        poses_to_show = POSE_NAMES if args.pose is None else [args.pose]
        for name in poses_to_show:
            if name == "stop":
                print("\n  Pose: stop  (kp=kd=0, motors go limp)")
                continue
            if name in ("recalibrate", "thumb_wave"):
                print(f"\n  Pose: {name}  (special command, no joint values)")
                continue
            q_r, q_l = POSES[name]
            _print_pose(name, q_r, q_l, sides)
        return

    if not args.interactive and args.pose is None and args.joint is None \
            and not args.monitor:
        ap.error("Specify --pose, --joint --angle, --interactive, or --monitor")

    commander = Dex3Commander(network_interface=args.net)

    # ── Monitor mode: print live state vs commanded ─────────────────────
    if args.monitor:
        print(f"\n[Monitor] live state  (Ctrl-C to stop)")
        print(f"  ACT=actual position  TAU=actual torque")
        print(f"  If TAU≈0 after command → motor fault or competing publisher\n")
        hdr = f"{'joint':>10s}  " + "  ".join(
            f"{'ACT':>7s} {'TAU':>6s}" for _ in sides)
        div = "-" * (12 + 16 * len(sides))
        slbl = "  ".join(f"{'--- '+s.upper()+' ---':>14s}" for s in sides)
        try:
            while True:
                rows = [hdr, f"{'':>10s}  {slbl}", div]
                for i, name in enumerate(MOTOR_NAMES):
                    row = f"{name:>10s}  "
                    for side in sides:
                        q_act = commander._current_q(  is_left=(side=="left"))
                        tau   = commander._current_tau(is_left=(side=="left"))
                        flag  = "!" if abs(tau[i]) < 0.005 and i == 0 else " "
                        row  += f"{q_act[i]:+7.3f} {tau[i]:+6.3f}{flag} "
                    rows.append(row)
                print("\033[2K\r" + ("\033[A\033[2K\r" * (len(rows)-1)), end="")
                print("\n".join(rows), flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[Monitor] stopped.")
        return

    # Single-joint test
    if args.joint is not None:
        if args.angle is None:
            ap.error("--joint requires --angle <radians>")
        j_idx   = MOTOR_NAMES.index(args.joint)
        kp_orig = KP[j_idx]
        KP[j_idx] = kp_orig * args.kp_scale
        if args.kp_scale != 1.0:
            print(f"[Dex3] KP[{j_idx}] scaled {kp_orig:.2f} → {KP[j_idx]:.2f}")
        for side in sides:
            q_start = commander._current_q(is_left=(side == "left")).copy()
            q_target = q_start.copy()
            q_target[j_idx] = args.angle
            steps = max(1, int(round(args.duration * CTRL_HZ)))
            print(f"[Dex3] joint test: {side.upper()} {args.joint}[{j_idx}] "
                  f"→ {args.angle:+.3f} rad  over {args.duration:.1f}s  "
                  f"KP={KP[j_idx]:.2f}")
            pub = commander._left_pub if side == "left" else commander._right_pub
            for step in range(steps + 1):
                alpha = step / steps
                q = (1.0 - alpha) * q_start + alpha * q_target
                pub.Write(_build_cmd(q))
                with commander._lock:
                    if side == "left":
                        commander._q_cmd_left = q.copy()
                    else:
                        commander._q_cmd_right = q.copy()
                time.sleep(CTRL_DT)
        KP[j_idx] = kp_orig  # restore
        return

    if args.pose:
        if args.pose == "recalibrate":
            commander.recalibrate(sides)
        elif args.pose == "thumb_wave":
            commander.thumb_wave(sides, duration=args.duration)
        else:
            commander.send(args.pose, sides, duration=args.duration)
        return

    # Interactive loop
    print(f"\nAvailable poses: {', '.join(POSE_NAMES)}")
    print(f"Duration: {args.duration}s   (change with --duration)")
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
            if raw == "recalibrate":
                commander.recalibrate(sides)
            elif raw == "thumb_wave":
                commander.thumb_wave(sides, duration=args.duration)
            else:
                commander.send(raw, sides, duration=args.duration)
    except KeyboardInterrupt:
        print("\n[Dex3] Sending stop before exit...")
        commander.send("stop", sides, duration=0.5)
        print("[Dex3] Done.")


if __name__ == "__main__":
    main()
