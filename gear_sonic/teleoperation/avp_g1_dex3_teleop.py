#!/usr/bin/env python3
"""
avp_g1_dex3_teleop.py
=====================
Apple Vision Pro → Unitree G1-EDU Dex3 (tri-finger) hand teleoperation.

Receives hand joint positions streamed over UDP from the AVPHandStreamer
visionOS app, retargets to G1 Dex3 7-DOF motor commands, and sends via DDS.

Pipeline
--------
  AVP (AVPHandStreamer visionOS app)
      │  WiFi UDP port 9870
      ▼
  UDP receiver  →  25 joint positions per hand  (x,y,z meters, world frame)
      │  retarget()
      ▼
  7 motor targets per hand  (radians)
      │  DDS  (unitree_sdk2py)
      ▼
  G1 Dex3  rt/dex3/{left,right}/cmd

Hand joint order — WebXR XRHand spec (25 joints), same as TeleVuer/vuer:
  Joint indices match the official unitreerobotics/xr_teleoperate exactly.
   0  wrist
   1  thumb-metacarpal
   2  thumb-phalanx-proximal
   3  thumb-phalanx-distal
   4  thumb-tip
   5  index-finger-metacarpal
   6  index-finger-phalanx-proximal
   7  index-finger-phalanx-intermediate
   8  index-finger-phalanx-distal
   9  index-finger-tip
  10  middle-finger-metacarpal
  11  middle-finger-phalanx-proximal
  12  middle-finger-phalanx-intermediate
  13  middle-finger-phalanx-distal
  14  middle-finger-tip
  15  ring-finger-metacarpal
  16  ring-finger-phalanx-proximal
  17  ring-finger-phalanx-intermediate
  18  ring-finger-phalanx-distal
  19  ring-finger-tip
  20  pinky-finger-metacarpal
  21  pinky-finger-phalanx-proximal
  22  pinky-finger-phalanx-intermediate
  23  pinky-finger-phalanx-distal
  24  pinky-finger-tip

  Note: AVP HandSkeleton.allCases has 27 joints (adds forearmWrist=25,
  forearmArm=26). We use only the first 25 to match the official pipeline.

Dex3 motor map (7 motors per hand, hardware order):
   0  Thumb abduction (thumb_0)
   1  Thumb MCP flex  (thumb_1)
   2  Thumb IP  flex  (thumb_2)
   3  Middle MCP flex (middle_0)
   4  Middle PIP flex (middle_1)
   5  Index MCP flex  (index_0)
   6  Index PIP flex  (index_1)

Usage
-----
  python avp_g1_dex3_teleop.py --net enp36s0f1 [--port 9870]

Requirements
------------
  pip install numpy
  unitree_sdk2_python installed (external_dependencies/)
"""

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "external_dependencies" / "unitree_sdk2_python"))

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
import threading as _threading

class RecurrentThread:
    """Cross-platform replacement for unitree_sdk2py RecurrentThread (uses threading.Timer)."""
    def __init__(self, interval, target, name="RecurrentThread"):
        self._interval = interval
        self._target   = target
        self._name     = name
        self._stop_evt = _threading.Event()
        self._thread   = None

    def _run(self):
        while not self._stop_evt.wait(self._interval):
            self._target()

    def Start(self):
        self._stop_evt.clear()
        self._thread = _threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def Stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2.0)

# ── Constants ─────────────────────────────────────────────────────────────────
UDP_PORT   = 9870
MOTOR_NUM  = 7
CTRL_HZ    = 50
CTRL_DT    = 1.0 / CTRL_HZ
TIMEOUT_S  = 0.5      # stop sending if no AVP packet for this long

# PD gains
KP = [2.0, 1.5, 1.5, 1.2, 1.2, 1.2, 1.2]
KD = [0.1, 0.08, 0.08, 0.05, 0.05, 0.05, 0.05]

# JOINT_MIN/MAX per official NLopt bounds for DexPilot (both hands):
# Order: [thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]
# DexPilot sign: index/middle positive=flex; thumb_2 negative=flex
JOINT_MIN = np.array([-1.05, -0.92, -1.75, 0.0,  0.0,  0.0,  0.0 ])
JOINT_MAX = np.array([ 1.05,  0.73,  0.0,  1.57, 1.75, 1.57, 1.75])

# AVP joint indices
J_WRIST        = 0
J_THUMB_KN     = 1   # thumbKnuckle  (CMC)
J_THUMB_IB     = 2   # thumbIntermediateBase (MCP)
J_THUMB_IT     = 3   # thumbIntermediateTip  (IP)
J_THUMB_TIP    = 4
J_IDX_META     = 5
J_IDX_KN       = 6   # indexFingerKnuckle    (MCP)
J_IDX_IB       = 7   # indexFingerIntermediateBase (PIP)
J_IDX_IT       = 8   # indexFingerIntermediateTip  (DIP)
J_IDX_TIP      = 9
J_MID_META     = 10
J_MID_KN       = 11  # middleFingerKnuckle
J_MID_IB       = 12
J_MID_IT       = 13
J_MID_TIP      = 14


# ── DexPilot retargeting (official Unitree algorithm) ────────────────────────
# Uses dex-retargeting library with pinocchio IK.
# DexPilot takes 6 fingertip-pair direction vectors and solves the 7-DOF Dex3
# joint configuration via nonlinear optimization.

import yaml as _yaml
from pathlib import Path as _Path
from dex_retargeting.retargeting_config import RetargetingConfig as _RetargetingConfig

_ASSETS = _Path(__file__).parent / "assets"
_RetargetingConfig.set_default_urdf_dir(str(_ASSETS))
_cfg_yaml = _yaml.safe_load((_ASSETS / "unitree_hand/unitree_dex3.yml").read_text())

# Official approach: pass YAML dict directly, matching hand_retargeting.py exactly:
#   RetargetingConfig.from_dict(self.cfg['left'])
# from_dict() handles np.array conversion for target_link_human_indices internally.
_left_retarget  = _RetargetingConfig.from_dict(_cfg_yaml["left"]).build()
_right_retarget = _RetargetingConfig.from_dict(_cfg_yaml["right"]).build()

# Hardware joint order from official Unitree hand_retargeting.py (both hands):
#   dex3_api_joint_names = [thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]
_LEFT_HW_JOINTS  = ["left_hand_thumb_0_joint",  "left_hand_thumb_1_joint",  "left_hand_thumb_2_joint",
                    "left_hand_middle_0_joint",  "left_hand_middle_1_joint",
                    "left_hand_index_0_joint",   "left_hand_index_1_joint"]
_RIGHT_HW_JOINTS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
                    "right_hand_index_0_joint",  "right_hand_index_1_joint"]

_left_to_hw  = [_left_retarget.joint_names.index(n)  for n in _LEFT_HW_JOINTS]
_right_to_hw = [_right_retarget.joint_names.index(n) for n in _RIGHT_HW_JOINTS]
_left_indices  = _left_retarget.optimizer.target_link_human_indices   # shape (2,6) for DexPilot
_right_indices = _right_retarget.optimizer.target_link_human_indices

# Coordinate transform: AVP visionOS (OpenXR) world frame → Unitree URDF hand frame.
# Coordinate transform: wrist-local geometric frame → Unitree URDF hand frame.
# Derived as R_old @ R_wrist_local_to_world where:
#   R_old = [[0,1,0],[0,0,1],[1,0,0]] was the OpenXR-world → URDF rotation, and
#   the geometric wrist-local frame axes in OpenXR world for canonical right hand are:
#     x_local = OpenXR +X (ulnar)   y_local = OpenXR -Z (distal)   z_local = OpenXR +Y (dorsal)
# Mapping:  x_local → URDF +Z   |  y_local (distal) → URDF -Y   |  z_local (dorsal) → URDF +X
# Same matrix for both hands (left hand x flipped in _pts_to_wrist_frame).
_R_WRIST_TO_URDF = np.array([[0., 0., 1.],
                              [0.,-1., 0.],
                              [1., 0., 0.]])


def _pts_to_wrist_frame(pts: np.ndarray, is_right: bool) -> np.ndarray:
    """Wrist-local frame from joint geometry; replicates tv_wrapper fast_mat_inv.

      y-axis : wrist(0) → middle_meta(10)  — distal
      x-axis : ulnar (pinky side); left hand negated so z=x×y is dorsal
      z-axis : x × y  — dorsal
    """
    origin = pts[0]
    y = pts[10] - origin
    yn = np.linalg.norm(y)
    if yn < 1e-9:
        return pts - origin
    y /= yn
    x = pts[20] - pts[5]     # pinky_meta - index_meta
    if not is_right:
        x = -x               # left hand: pinky/index swap laterally
    x -= np.dot(x, y) * y
    xn = np.linalg.norm(x)
    if xn < 1e-9:
        return pts - origin
    x /= xn
    z = np.cross(x, y)
    x = np.cross(y, z)
    return (pts - origin) @ np.stack([x, y, z], axis=1)


def retarget(joints: list, is_right: bool) -> np.ndarray:
    """
    Pure DexPilot retargeting: 25 hand joint positions -> Dex3 7-DOF motor targets.

    Uses only the first 25 joints (WebXR XRHand layout, same as TeleVuer/vuer).
    AVP HandSkeleton sends 27; joints 25-26 (forearmWrist, forearmArm) are ignored.

    Output hardware order (both hands): [thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]
    """
    if len(joints) < 25:
        return np.zeros(MOTOR_NUM)

    pts        = np.array(joints[:25], dtype=np.float64)
    retargeter = _right_retarget if is_right else _left_retarget
    indices    = _right_indices  if is_right else _left_indices
    to_hw      = _right_to_hw   if is_right else _left_to_hw

    # Step 1: wrist-local frame (makes retargeting invariant to wrist rotation).
    pts_local = _pts_to_wrist_frame(pts, is_right)
    # Step 2: 6 DexPilot inter-finger vectors in wrist-local frame.
    raw_ref = pts_local[indices[1]] - pts_local[indices[0]]  # (6, 3)
    # Step 3: rotate wrist-local → Unitree URDF frame.
    ref = raw_ref @ _R_WRIST_TO_URDF.T                      # (6, 3)
    return retargeter.retarget(ref)[to_hw]


# ── UDP receiver thread ───────────────────────────────────────────────────────

class AVPReceiver:
    """Listens for UDP hand packets from the AVPHandStreamer visionOS app."""

    PACKET_FORMAT = {
        # {"hand":"left","joints":[[x,y,z],...27...],"t":timestamp}
    }

    def __init__(self, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(0.1)
        self._sock.bind(("0.0.0.0", port))

        self._left_joints:  list | None = None
        self._right_joints: list | None = None
        self._left_ts  = 0.0
        self._right_ts = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        print(f"[AVP] Listening for hand data on UDP port {port}")

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._sock.close()

    def _recv_loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(65536)
                pkt = json.loads(data.decode("utf-8"))
                hand = pkt.get("hand", "")
                joints = pkt.get("joints", [])
                ts = pkt.get("t", time.time())
                with self._lock:
                    if hand == "left":
                        self._left_joints = joints
                        self._left_ts = ts
                    elif hand == "right":
                        self._right_joints = joints
                        self._right_ts = ts
            except socket.timeout:
                pass
            except Exception as e:
                print(f"[AVP] recv error: {e}")

    def get(self):
        """Returns (left_joints, right_joints, left_active, right_active)."""
        now = time.time()
        with self._lock:
            l_active = (self._left_joints is not None and
                        now - self._left_ts < TIMEOUT_S)
            r_active = (self._right_joints is not None and
                        now - self._right_ts < TIMEOUT_S)
            return (self._left_joints or [],
                    self._right_joints or [],
                    l_active, r_active)


# ── G1 Dex3 commander ────────────────────────────────────────────────────────

class Dex3Commander:
    RAMP_DURATION = 2.0   # seconds to ramp from current pose to first AVP target

    def __init__(self, network_interface: str, receiver: AVPReceiver):
        ChannelFactoryInitialize(0, network_interface)

        self._left_pub  = ChannelPublisher("rt/dex3/left/cmd",  HandCmd_)
        self._right_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self._left_pub.Init()
        self._right_pub.Init()

        self._receiver = receiver
        self._q_left  = np.zeros(MOTOR_NUM)
        self._q_right = np.zeros(MOTOR_NUM)
        self._thread: RecurrentThread | None = None
        self._running = False

        # Ramp-in state: interpolate from current pose to first target over RAMP_DURATION
        self._ramp_steps     = max(1, int(round(self.RAMP_DURATION * CTRL_HZ)))
        self._ramp_step      = 0          # counts up to _ramp_steps
        self._ramp_start_l   = np.zeros(MOTOR_NUM)
        self._ramp_start_r   = np.zeros(MOTOR_NUM)
        self._ramp_target_l  = None       # set on first AVP packet
        self._ramp_target_r  = None

    def _build_cmd(self, q: np.ndarray) -> HandCmd_:
        cmd = unitree_hg_msg_dds__HandCmd_()
        for i in range(MOTOR_NUM):
            m = cmd.motor_cmd[i]
            m.q   = float(q[i])
            m.dq  = 0.0
            m.tau = 0.0
            m.kp  = KP[i]
            m.kd  = KD[i]
            m.mode = (i & 0x0F) | (0x01 << 4) | (0x01 << 7)
        # HandCmd_ does not use CRC (unlike LowCmd_) — per official Unitree examples
        return cmd

    def _control_loop(self):
        if not self._running:
            return
        l_joints, r_joints, l_active, r_active = self._receiver.get()

        q_target_l = retarget(l_joints, is_right=False) if l_active else self._q_left
        q_target_r = retarget(r_joints, is_right=True)  if r_active else self._q_right

        # Ramp-in: on first packet, record start pose and interpolate over RAMP_DURATION
        if self._ramp_target_l is None and (l_active or r_active):
            self._ramp_target_l = q_target_l.copy()
            self._ramp_target_r = q_target_r.copy()
            print(f"[Dex3] Ramping to first AVP target over {self.RAMP_DURATION:.1f}s ...")

        if self._ramp_step < self._ramp_steps and self._ramp_target_l is not None:
            alpha = self._ramp_step / self._ramp_steps
            self._q_left  = (1.0 - alpha) * self._ramp_start_l + alpha * self._ramp_target_l
            self._q_right = (1.0 - alpha) * self._ramp_start_r + alpha * self._ramp_target_r
            self._ramp_step += 1
        else:
            if l_active:
                self._q_left  = q_target_l
            if r_active:
                self._q_right = q_target_r

        self._left_pub.Write(self._build_cmd(self._q_left))
        self._right_pub.Write(self._build_cmd(self._q_right))

    def start(self):
        self._running = True
        self._thread = RecurrentThread(
            interval=CTRL_DT, target=self._control_loop,
            name="AVP_Dex3_Loop")
        self._thread.Start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.Stop()

    @property
    def q_left(self):  return self._q_left
    @property
    def q_right(self): return self._q_right


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Apple Vision Pro → G1 Dex3 hand teleoperation (UDP)")
    ap.add_argument("--net",  default="enp36s0f1",
                    help="Network interface connected to G1 (default: enp36s0f1)")
    ap.add_argument("--port", type=int, default=UDP_PORT,
                    help=f"UDP port to receive AVP hand data (default: {UDP_PORT})")
    ap.add_argument("--print-only", action="store_true",
                    help="Print raw joint angles only; do not connect to G1")
    args = ap.parse_args()

    # Detect local IP (cross-platform: connect a UDP socket to get the outbound IP)
    try:
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        _local_ip = _s.getsockname()[0]
        _s.close()
    except Exception:
        _local_ip = socket.gethostbyname(socket.gethostname())
    print(f"[Info] PC IP: {_local_ip}")
    print(f"[Info] Waiting for AVPHandStreamer app on UDP port {args.port} ...")
    print(f"[Info] Enter this IP and port {args.port} in the AVP app.\n")

    receiver = AVPReceiver(port=args.port)
    receiver.start()

    # Wait for first packet
    while True:
        _, _, l, r = receiver.get()
        if l or r:
            break
        time.sleep(0.1)
    print("[AVP] Hand data received! Starting G1 control.\n")

    if args.print_only:
        print("[Info] --print-only mode: printing raw joint angles, no G1 connection.\n")
        try:
            while True:
                l_joints, r_joints, la, ra = receiver.get()
                ql = retarget(l_joints, is_right=False) if la else np.zeros(MOTOR_NUM)
                qr = retarget(r_joints, is_right=True)  if ra else np.zeros(MOTOR_NUM)
                print(
                    f"  L({'ON ' if la else 'OFF'}): "
                    f"abd={ql[0]:+.2f} tmcp={ql[1]:+.2f} tip={ql[2]:+.2f} "
                    f"mmcp={ql[3]:+.2f} mpip={ql[4]:+.2f} "
                    f"imcp={ql[5]:+.2f} ipip={ql[6]:+.2f}"
                )
                print(
                    f"  R({'ON ' if ra else 'OFF'}): "
                    f"abd={qr[0]:+.2f} tmcp={qr[1]:+.2f} tip={qr[2]:+.2f} "
                    f"mmcp={qr[3]:+.2f} mpip={qr[4]:+.2f} "
                    f"imcp={qr[5]:+.2f} ipip={qr[6]:+.2f}"
                )
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[Info] Shutting down...")
        finally:
            receiver.stop()
        return

    commander = Dex3Commander(network_interface=args.net, receiver=receiver)
    commander.start()

    try:
        while True:
            ql = commander.q_left
            qr = commander.q_right
            _, _, la, ra = receiver.get()
            print(
                f"  L({'ON ' if la else 'OFF'}): "
                f"abd={ql[0]:+.2f} tmcp={ql[1]:+.2f} tip={ql[2]:+.2f} "
                f"imcp={ql[3]:+.2f} ipip={ql[4]:+.2f} "
                f"mmcp={ql[5]:+.2f} mpip={ql[6]:+.2f}"
            )
            print(
                f"  R({'ON ' if ra else 'OFF'}): "
                f"abd={qr[0]:+.2f} tmcp={qr[1]:+.2f} tip={qr[2]:+.2f} "
                f"imcp={qr[3]:+.2f} ipip={qr[4]:+.2f} "
                f"mmcp={qr[5]:+.2f} mpip={qr[6]:+.2f}"
            )
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Info] Shutting down...")
    finally:
        commander.stop()
        receiver.stop()


if __name__ == "__main__":
    main()
