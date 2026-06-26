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
  UDP receiver  →  27 joint positions per hand  (x,y,z meters, world frame)
      │  retarget()
      ▼
  7 motor targets per hand  (radians)
      │  DDS  (unitree_sdk2py)
      ▼
  G1 Dex3  rt/dex3/{left,right}/cmd

AVP HandSkeleton joint order (27 joints, matches HandSkeleton.JointName.allCases):
   0  wrist
   1  thumbKnuckle
   2  thumbIntermediateBase
   3  thumbIntermediateTip
   4  thumbTip
   5  indexFingerMetacarpal
   6  indexFingerKnuckle
   7  indexFingerIntermediateBase
   8  indexFingerIntermediateTip
   9  indexFingerTip
  10  middleFingerMetacarpal
  11  middleFingerKnuckle
  12  middleFingerIntermediateBase
  13  middleFingerIntermediateTip
  14  middleFingerTip
  15  ringFingerMetacarpal
  16  ringFingerKnuckle
  17  ringFingerIntermediateBase
  18  ringFingerIntermediateTip
  19  ringFingerTip
  20  littleFingerMetacarpal
  21  littleFingerKnuckle
  22  littleFingerIntermediateBase
  23  littleFingerIntermediateTip
  24  littleFingerTip
  25  forearmWrist
  26  forearmArm

Dex3 motor map (7 motors per hand):
   0  Thumb abduction
   1  Thumb MCP flex
   2  Thumb IP  flex
   3  Index MCP flex
   4  Index PIP flex
   5  Middle MCP flex
   6  Middle PIP flex

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
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

# ── Constants ─────────────────────────────────────────────────────────────────
UDP_PORT   = 9870
MOTOR_NUM  = 7
CTRL_HZ    = 50
CTRL_DT    = 1.0 / CTRL_HZ
TIMEOUT_S  = 0.5      # stop sending if no AVP packet for this long

# PD gains
KP = [2.0, 1.5, 1.5, 1.2, 1.2, 1.2, 1.2]
KD = [0.1, 0.08, 0.08, 0.05, 0.05, 0.05, 0.05]

# Dex3 joint limits [min, max] radians
JOINT_MIN = np.array([-0.5, -1.50, 0.0,  -1.57, -1.75, -1.57, -1.75])
JOINT_MAX = np.array([ 0.5,  0.0,  1.70,  0.0,   0.0,   0.0,   0.0 ])

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


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _p(joints: list, idx: int) -> np.ndarray:
    return np.array(joints[idx], dtype=np.float64)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at joint B in radians."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _lateral(wrist: np.ndarray, thumb_kn: np.ndarray,
             idx_kn: np.ndarray) -> float:
    """Thumb abduction angle relative to palm plane."""
    palm = idx_kn - wrist
    thumb = thumb_kn - wrist
    np_ = np.linalg.norm(palm)
    nt  = np.linalg.norm(thumb)
    if np_ < 1e-6 or nt < 1e-6:
        return 0.0
    palm /= np_
    cos_a = np.clip(np.dot(palm, thumb / nt), -1.0, 1.0)
    return float(np.arccos(cos_a)) - np.pi / 2.0


# ── Retargeting ───────────────────────────────────────────────────────────────

def retarget(joints: list, is_right: bool) -> np.ndarray:
    """
    Map AVP 27-joint list → Dex3 7-DOF motor targets (radians).

    joints : list of 27 entries, each [x, y, z]  (meters, world frame)
    """
    if len(joints) < 15:
        return np.zeros(MOTOR_NUM)

    wrist     = _p(joints, J_WRIST)
    thumb_kn  = _p(joints, J_THUMB_KN)
    thumb_ib  = _p(joints, J_THUMB_IB)
    thumb_it  = _p(joints, J_THUMB_IT)
    thumb_tip = _p(joints, J_THUMB_TIP)
    idx_kn    = _p(joints, J_IDX_KN)
    idx_ib    = _p(joints, J_IDX_IB)
    idx_tip   = _p(joints, J_IDX_TIP)
    mid_kn    = _p(joints, J_MID_KN)
    mid_ib    = _p(joints, J_MID_IB)
    mid_tip   = _p(joints, J_MID_TIP)

    sign = 1.0 if is_right else -1.0

    # Motor 0: thumb abduction
    q0 = np.clip(sign * _lateral(wrist, thumb_kn, idx_kn) * 0.6,
                 JOINT_MIN[0], JOINT_MAX[0])

    # Motor 1: thumb MCP flex
    q1 = np.clip((np.pi - _angle(wrist, thumb_ib, thumb_it)) * 0.8 * -1,
                 JOINT_MIN[1], JOINT_MAX[1])

    # Motor 2: thumb IP flex
    q2 = np.clip((np.pi - _angle(thumb_ib, thumb_it, thumb_tip)) * 0.9,
                 JOINT_MIN[2], JOINT_MAX[2])

    # Motor 3: index MCP flex
    q3 = np.clip(-(np.pi - _angle(wrist, idx_kn, idx_ib)) * 0.9,
                 JOINT_MIN[3], JOINT_MAX[3])

    # Motor 4: index PIP flex
    q4 = np.clip(-(np.pi - _angle(idx_kn, idx_ib, idx_tip)) * 0.9,
                 JOINT_MIN[4], JOINT_MAX[4])

    # Motor 5: middle MCP flex
    q5 = np.clip(-(np.pi - _angle(wrist, mid_kn, mid_ib)) * 0.9,
                 JOINT_MIN[5], JOINT_MAX[5])

    # Motor 6: middle PIP flex
    q6 = np.clip(-(np.pi - _angle(mid_kn, mid_ib, mid_tip)) * 0.9,
                 JOINT_MIN[6], JOINT_MAX[6])

    return np.array([q0, q1, q2, q3, q4, q5, q6], dtype=np.float64)


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
    def __init__(self, network_interface: str, receiver: AVPReceiver):
        ChannelFactoryInitialize(0, network_interface)

        self._left_pub  = ChannelPublisher("rt/dex3/left/cmd",  HandCmd_)
        self._right_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self._left_pub.Init()
        self._right_pub.Init()

        self._crc = CRC()
        self._receiver = receiver
        self._q_left  = np.zeros(MOTOR_NUM)
        self._q_right = np.zeros(MOTOR_NUM)
        self._thread: RecurrentThread | None = None
        self._running = False

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
        cmd.crc = self._crc.Crc(cmd)
        return cmd

    def _control_loop(self):
        if not self._running:
            return
        l_joints, r_joints, l_active, r_active = self._receiver.get()
        if l_active:
            self._q_left  = retarget(l_joints, is_right=False)
        if r_active:
            self._q_right = retarget(r_joints, is_right=True)
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
    args = ap.parse_args()

    print(f"[Info] PC IP: ", end="")
    import subprocess
    print(subprocess.check_output("hostname -I | awk '{print $1}'",
                                  shell=True).decode().strip())
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
