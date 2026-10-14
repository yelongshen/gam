#!/usr/bin/env python3
"""
inspire_read_params.py  --  READ-ONLY: print the speed / force / angle registers of an Inspire hand.

Run it on the machine the hand's USB-serial adapter is plugged into (the robot PC), with the inspire_g1 service STOPPED
(two programs on one serial port garble each other).

Uses the same RS485 frame as Unitree's DFX_inspire_service (inspire.h):
    request : EB 90 <id> 04 11 <addr_lo> <addr_hi> <nbytes> <checksum>      (0x11 = read)
    reply   : EB 90 <id> <len> 11 <addr_lo> <addr_hi> <data ...> <checksum>
    checksum = sum of bytes [2 .. last-1] mod 256
Registers (6 x uint16 little endian, DOF order pinky, ring, middle, index, thumb_bend, thumb_rotation):
    0x05F2 SPEED_SET (0..1000)    0x05DA FORCE_SET (0..1000)    0x060A ANGLE_ACT (0..1000)

    python3 gear_sonic/teleoperation/inspire_read_params.py --port /dev/ttyUSB1          # right hand in inspire_g1.cpp
    python3 gear_sonic/teleoperation/inspire_read_params.py --port /dev/ttyUSB2
"""
import argparse
import os
import select
import sys
import termios
import time

REGS = {"SPEED_SET": 0x05F2, "FORCE_SET": 0x05DA, "ANGLE_ACT": 0x060A}
DOF = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]


def checksum(frame):
    return sum(frame[2:]) & 0xFF


def read_frame(hand_id, addr, nbytes=12):
    f = [0xEB, 0x90, hand_id, 0x04, 0x11, addr & 0xFF, (addr >> 8) & 0xFF, nbytes]
    return bytes(f + [checksum(f + [0])])


def open_port(path, baud=termios.B115200):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
    attr = termios.tcgetattr(fd)
    attr[0] = 0                                  # iflag
    attr[1] = 0                                  # oflag
    attr[2] = termios.CS8 | termios.CREAD | termios.CLOCAL   # 8N1
    attr[3] = 0                                  # lflag: raw
    attr[4] = attr[5] = baud
    attr[6][termios.VMIN], attr[6][termios.VTIME] = 0, 0
    termios.tcsetattr(fd, termios.TCSANOW, attr)
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


def query(fd, hand_id, addr, tries=3):
    for _ in range(tries):
        termios.tcflush(fd, termios.TCIOFLUSH)
        os.write(fd, read_frame(hand_id, addr))
        buf, t0 = b"", time.time()
        while len(buf) < 20 and time.time() - t0 < 0.3:
            if select.select([fd], [], [], 0.05)[0]:
                buf += os.read(fd, 64)
        if len(buf) >= 20 and buf[0] == 0xEB and buf[1] == 0x90 and buf[19] == checksum(list(buf[:20])):
            return [buf[7 + 2 * i] | (buf[8 + 2 * i] << 8) for i in range(6)]
    return None


def main():
    ap = argparse.ArgumentParser(description="read-only: print Inspire hand speed/force/angle registers")
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", type=int, default=1, help="hand slave id (the service uses 1)")
    a = ap.parse_args()
    fd = open_port(a.port)
    print(f"port {a.port}  hand id {a.id}  (read-only)")
    print(f"{'register':10s} " + " ".join(f"{d:>10s}" for d in DOF))
    ok = False
    for name, addr in REGS.items():
        v = query(fd, a.id, addr)
        ok |= v is not None
        print(f"{name:10s} " + (" ".join(f"{x:10d}" for x in v) if v else "   no valid reply (service still running? wrong port or id?)"))
    os.close(fd)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
