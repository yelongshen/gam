import socket
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_


MOTOR_NUM_HAND = 7

IGNORED_INTERFACE_PREFIXES = (
    "lo",
    "docker",
    "dummy",
    "l4tbr",
    "rndis",
    "usb",
    "br-",
    "veth",
)


def is_ignored_interface(name):
    return any(name.startswith(prefix) for prefix in IGNORED_INTERFACE_PREFIXES)


def interface_operstate(name):
    path = f"/sys/class/net/{name}/operstate"
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        return "unknown"


def print_available_interfaces():
    print("Available network interfaces:")
    for _, name in socket.if_nameindex():
        state = interface_operstate(name)
        ignored = " ignored" if is_ignored_interface(name) else ""
        print(f"  - {name}: state={state}{ignored}")


def pick_default_interface():
    candidates = []
    fallback_candidates = []

    for _, name in socket.if_nameindex():
        if is_ignored_interface(name):
            continue

        fallback_candidates.append(name)
        if interface_operstate(name) == "up":
            candidates.append(name)

    if candidates:
        return candidates[0]

    if fallback_candidates:
        return fallback_candidates[0]

    return None


def print_hand_state(label, msg):
    print(f"\n{label} hand state:")
    if msg is None:
        print("  no state received")
        return

    motor_count = len(msg.motor_state)
    print(f"  motors: {motor_count}")
    for index in range(min(motor_count, MOTOR_NUM_HAND)):
        motor = msg.motor_state[index]
        q_deg = np.rad2deg(motor.q)
        print(
            f"  {index}: q={motor.q:+.4f} rad ({q_deg:+.1f} deg), "
            f"dq={motor.dq:+.4f}, tau_est={motor.tau_est:+.4f}"
        )


def main():
    if len(sys.argv) > 2:
        print(f"Usage: python3 {sys.argv[0]} [networkInterface]")
        print("Example: python3 monitor_g1_dex3_hand_state.py eth0")
        sys.exit(1)

    if len(sys.argv) == 2:
        network_interface = sys.argv[1]
        print(f"Using provided network interface: {network_interface}")
        ChannelFactoryInitialize(0, network_interface)
    else:
        print_available_interfaces()
        network_interface = pick_default_interface()
        if network_interface is None:
            print("No usable network interface found.")
            sys.exit(1)
        print(f"No network interface provided. First usable interface is: {network_interface}")
        print("Using CycloneDDS autodetect mode for initialization.")
        ChannelFactoryInitialize(0)

    state = {"left": None, "right": None}

    left_subscriber = ChannelSubscriber("rt/dex3/left/state", HandState_)
    right_subscriber = ChannelSubscriber("rt/dex3/right/state", HandState_)
    left_subscriber.Init(lambda msg: state.__setitem__("left", msg), 10)
    right_subscriber.Init(lambda msg: state.__setitem__("right", msg), 10)

    print("Monitoring Dex3 hand states every 5 seconds. Press Ctrl+C to stop.")
    print("Open pose is approximately q=0 for each motor; compare left and right signs/magnitudes.")
    while True:
        time.sleep(5.0)
        print("\n=== Dex3 snapshot ===")
        print("time:", time.strftime("%Y-%m-%d %H:%M:%S"))
        print_hand_state("Left", state["left"])
        print_hand_state("Right", state["right"])


if __name__ == "__main__":
    main()