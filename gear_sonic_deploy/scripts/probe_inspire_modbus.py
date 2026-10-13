#!/usr/bin/env python3
"""Read-only Modbus-TCP probe for Inspire hands (FTP-style, Ethernet).

Asks each candidate IP directly -- no DDS, no hand service, no inspire_sdkpy
needed (just `pymodbus`). For every IP that accepts a TCP connection on the
Modbus port it reads two things, using ONLY read-holding-register requests:

  register 1000  (1 reg)  HAND_ID
  register 1546  (6 regs) ANGLE_ACT  -> current finger angles, 0..1000
                                        (order: pinky, ring, middle, index,
                                         thumb_bend, thumb_rot)

Register numbers are from NaCl-1374/inspire_hand_ws (`HAND_ID` in
init_set_inspire_hand.py, `angle_act` in inspire_sdk.py's states_structure).
Nothing is ever written, so this cannot move the fingers or change settings.

It deliberately does NOT scan serial ports: sending Modbus frames to an
unknown /dev/ttyUSB* device could disturb whatever is really attached there.

Usage:
    python3 gear_sonic_deploy/scripts/probe_inspire_modbus.py
    python3 gear_sonic_deploy/scripts/probe_inspire_modbus.py --ips 192.168.123.210 192.168.123.211

Note: the hand must be reachable from this machine's interface. A hand still
at its factory IP (192.168.11.210) is NOT on eth0's 192.168.123.x subnet.
"""
import argparse
import socket
import sys

DEFAULT_IPS = ["192.168.123.210", "192.168.123.211", "192.168.11.210"]
JOINTS = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]


def tcp_open(ip, port, timeout):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def to_signed16(v):
    return v - 65536 if v >= 32768 else v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ips", nargs="+", default=DEFAULT_IPS)
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--device-id", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1.5)
    args = ap.parse_args()

    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError:
        print("pymodbus is not installed (or is an old 2.x release). Try:  pip3 install --user 'pymodbus>=3'")
        return 2

    found = 0
    for ip in args.ips:
        if not tcp_open(ip, args.port, args.timeout):
            print("%-16s tcp/%d  closed or unreachable" % (ip, args.port))
            continue

        print("%-16s tcp/%d  OPEN" % (ip, args.port))
        client = ModbusTcpClient(ip, port=args.port, timeout=args.timeout)
        try:
            if not client.connect():
                print("    modbus connect failed")
                continue
            r = client.read_holding_registers(1000, 1, args.device_id)
            if r.isError():
                print("    HAND_ID read (reg 1000, device_id=%d): error %s" % (args.device_id, r))
                continue
            print("    HAND_ID (reg 1000) = %d" % r.registers[0])
            found += 1

            a = client.read_holding_registers(1546, 6, args.device_id)
            if a.isError():
                print("    ANGLE_ACT (reg 1546): error %s" % (a,))
            else:
                vals = [to_signed16(x) for x in a.registers]
                print("    ANGLE_ACT = " + " ".join("%s=%d" % (n, v) for n, v in zip(JOINTS, vals)))
        except Exception as e:
            print("    probe error: %r" % (e,))
        finally:
            client.close()

    print("\n%d hand(s) answered Modbus on port %d." % (found, args.port))
    return 0


if __name__ == "__main__":
    sys.exit(main())
