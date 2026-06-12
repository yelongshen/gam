import socket
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_


MOTOR_NUM_HAND = 7
PUBLISH_DT = 0.02
PUBLISH_DURATION = 1.0

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


def make_hand_mode(motor_index):
    status = 0x01
    timeout = 0x01
    mode = motor_index & 0x0F
    mode |= status << 4
    mode |= timeout << 7
    return mode


def current_q(msg):
    return [msg.motor_state[index].q for index in range(MOTOR_NUM_HAND)]


def make_zero_torque_command(q):
    cmd = unitree_hg_msg_dds__HandCmd_()
    for index in range(MOTOR_NUM_HAND):
        motor = cmd.motor_cmd[index]
        motor.mode = make_hand_mode(index)
        motor.q = float(q[index])
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = 0.0
        motor.kd = 0.0
    return cmd


def wait_for_states(state, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state["left"] is not None and state["right"] is not None:
            return True
        time.sleep(0.05)
    return False


def initialize_channel():
    if len(sys.argv) > 2:
        print(f"Usage: python3 {sys.argv[0]} [networkInterface]")
        print("Example: python3 zero_torque_g1_dex3_hands.py eth0")
        sys.exit(1)

    if len(sys.argv) == 2:
        network_interface = sys.argv[1]
        print(f"Using provided network interface: {network_interface}")
        ChannelFactoryInitialize(0, network_interface)
        return

    print_available_interfaces()
    network_interface = pick_default_interface()
    if network_interface is None:
        print("No usable network interface found.")
        sys.exit(1)
    print(f"No network interface provided. First usable interface is: {network_interface}")
    print("Using CycloneDDS autodetect mode for initialization.")
    ChannelFactoryInitialize(0)


def main():
    print("WARNING: This publishes zero-gain Dex3 hand commands for both hands.")
    print("The hands may relax or move under gravity/contact. Keep fingers and objects clear.")
    print("Make sure no other hand-control program is running.")
    input("Press Enter to continue, or Ctrl+C to cancel...")

    initialize_channel()

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

    left_command = make_zero_torque_command(current_q(state["left"]))
    right_command = make_zero_torque_command(current_q(state["right"]))

    print(f"Publishing zero-torque hand commands for {PUBLISH_DURATION:.1f} seconds.")
    stop_time = time.monotonic() + PUBLISH_DURATION
    while time.monotonic() < stop_time:
        left_publisher.Write(left_command)
        right_publisher.Write(right_command)
        time.sleep(PUBLISH_DT)

    print("Done. Dex3 zero-torque command sent and script exited.")


if __name__ == "__main__":
    main()