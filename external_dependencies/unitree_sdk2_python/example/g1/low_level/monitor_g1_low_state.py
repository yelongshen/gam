import socket
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


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


def print_low_state(msg):
    print("\n=== LowState snapshot ===")
    print("time:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("mode_pr:", msg.mode_pr)
    print("mode_machine:", msg.mode_machine)
    print("tick:", msg.tick)
    print("imu rpy:", list(msg.imu_state.rpy))
    print("imu gyroscope:", list(msg.imu_state.gyroscope))
    print("imu accelerometer:", list(msg.imu_state.accelerometer))

    print(f"all {len(msg.motor_state)} motors [index: q, dq, tau_est]:")
    for index, motor in enumerate(msg.motor_state):
        print(f"  {index:02d}: q={motor.q:.4f}, dq={motor.dq:.4f}, tau_est={motor.tau_est:.4f}")


def main():
    if len(sys.argv) > 2:
        print(f"Usage: python3 {sys.argv[0]} [networkInterface]")
        print("Example: python3 monitor_g1_low_state.py eth0")
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

    state = {"latest": None}

    def callback(msg):
        state["latest"] = msg

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(callback, 10)

    print("Monitoring rt/lowstate every 10 seconds. Press Ctrl+C to stop.")
    while True:
        time.sleep(10.0)
        if state["latest"] is None:
            print("No rt/lowstate received yet. Check interface, robot power, and DDS network.")
        else:
            print_low_state(state["latest"])


if __name__ == "__main__":
    main()