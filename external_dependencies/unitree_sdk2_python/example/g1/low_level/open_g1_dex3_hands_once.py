import socket
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_


MOTOR_NUM_HAND = 7
CONTROL_DT = 0.02
RAMP_DURATION = 5.0
HOLD_DURATION = 1.0
KP = [0.6] * MOTOR_NUM_HAND
KD = [0.08] * MOTOR_NUM_HAND
KP[0] = 1.0

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


def make_hand_mode(motor_index, timeout=0x00):
    status = 0x01
    mode = motor_index & 0x0F
    mode |= status << 4
    mode |= timeout << 7
    return mode


def current_q(msg):
    return np.array([msg.motor_state[index].q for index in range(MOTOR_NUM_HAND)], dtype=float)


def make_command(q_des, gains_enabled=True):
    cmd = unitree_hg_msg_dds__HandCmd_()
    for index in range(MOTOR_NUM_HAND):
        motor = cmd.motor_cmd[index]
        motor.mode = make_hand_mode(index, timeout=0x00 if gains_enabled else 0x01)
        motor.q = float(q_des[index])
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = KP[index] if gains_enabled else 0.0
        motor.kd = KD[index] if gains_enabled else 0.0
    return cmd


def wait_for_states(state, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state["left"] is not None and state["right"] is not None:
            return True
        time.sleep(0.05)
    return False


def print_start_state(label, q):
    print(f"{label} start q rad:", [round(value, 4) for value in q])
    print(f"{label} start q deg:", [round(float(np.rad2deg(value)), 1) for value in q])


def main():
    if len(sys.argv) > 2:
        print(f"Usage: python3 {sys.argv[0]} [networkInterface]")
        print("Example: python3 open_g1_dex3_hands_once.py eth0")
        sys.exit(1)

    print("WARNING: This publishes Dex3 hand commands to open both hands toward q=0.")
    print("Make sure the hands are clear and no other hand-control program is running.")
    input("Press Enter to continue, or Ctrl+C to cancel...")

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

    left_state_subscriber = ChannelSubscriber("rt/dex3/left/state", HandState_)
    right_state_subscriber = ChannelSubscriber("rt/dex3/right/state", HandState_)
    left_state_subscriber.Init(lambda msg: state.__setitem__("left", msg), 10)
    right_state_subscriber.Init(lambda msg: state.__setitem__("right", msg), 10)

    left_publisher = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
    right_publisher = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
    left_publisher.Init()
    right_publisher.Init()

    print("Waiting for left/right Dex3 hand state...")
    if not wait_for_states(state):
        print("Timed out waiting for Dex3 hand states. No commands were sent.")
        sys.exit(1)

    left_start = current_q(state["left"])
    right_start = current_q(state["right"])
    target = np.zeros(MOTOR_NUM_HAND)

    print_start_state("Left", left_start)
    print_start_state("Right", right_start)
    print(f"Ramping both hands to q=0 over {RAMP_DURATION:.1f} seconds.")

    start_time = time.monotonic()
    while True:
        elapsed = time.monotonic() - start_time
        ratio = min(elapsed / RAMP_DURATION, 1.0)
        left_q = (1.0 - ratio) * left_start + ratio * target
        right_q = (1.0 - ratio) * right_start + ratio * target

        left_publisher.Write(make_command(left_q))
        right_publisher.Write(make_command(right_q))

        if ratio >= 1.0:
            break
        time.sleep(CONTROL_DT)

    hold_until = time.monotonic() + HOLD_DURATION
    open_command = make_command(target)
    while time.monotonic() < hold_until:
        left_publisher.Write(open_command)
        right_publisher.Write(open_command)
        time.sleep(CONTROL_DT)

    relaxed_command = make_command(target, gains_enabled=False)
    left_publisher.Write(relaxed_command)
    right_publisher.Write(relaxed_command)
    print("Done. Sent final relaxed q=0 command and exiting.")


if __name__ == "__main__":
    main()