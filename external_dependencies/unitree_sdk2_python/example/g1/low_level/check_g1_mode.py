import socket
import sys
import time

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_api import (
    ROBOT_API_ID_LOCO_GET_BALANCE_MODE,
    ROBOT_API_ID_LOCO_GET_FSM_ID,
    ROBOT_API_ID_LOCO_GET_FSM_MODE,
)
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


FSM_ID_NAMES = {
    0: "zero torque",
    1: "damp / damping",
    3: "sit",
    200: "normal/start locomotion",
    702: "Lie2StandUp",
    706: "squat/stand transition",
}

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


def print_motion_switcher_state():
    print("=== Motion switcher ===")
    client = MotionSwitcherClient()
    client.SetTimeout(3.0)
    client.Init()

    code, result = client.CheckMode()
    print("CheckMode code:", code)
    print("CheckMode result:", result)


def call_loco_api(client, name, api_id):
    code, data = client._Call(api_id, "{}")
    print(f"{name}: code={code}, data={data}")

    if name == "fsm_id" and code == 0:
        try:
            fsm_id = int(data)
        except (TypeError, ValueError):
            return

        if fsm_id in FSM_ID_NAMES:
            print(f"fsm_id meaning: {FSM_ID_NAMES[fsm_id]}")


def print_loco_state():
    print("\n=== Loco state ===")
    client = LocoClient()
    client.SetTimeout(3.0)
    client.Init()

    call_loco_api(client, "fsm_id", ROBOT_API_ID_LOCO_GET_FSM_ID)
    call_loco_api(client, "fsm_mode", ROBOT_API_ID_LOCO_GET_FSM_MODE)
    call_loco_api(client, "balance_mode", ROBOT_API_ID_LOCO_GET_BALANCE_MODE)


def print_low_state(timeout=3.0):
    print("\n=== LowState ===")
    state = {"received": False}

    def callback(msg):
        if state["received"]:
            return

        state["received"] = True
        print("mode_pr:", msg.mode_pr)
        print("mode_machine:", msg.mode_machine)
        print("imu rpy:", list(msg.imu_state.rpy))

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(callback, 10)

    deadline = time.time() + timeout
    while not state["received"] and time.time() < deadline:
        time.sleep(0.1)

    if not state["received"]:
        print("No rt/lowstate received. Check interface, robot power, and DDS network.")


def main():
    if len(sys.argv) > 2:
        print(f"Usage: python3 {sys.argv[0]} [networkInterface]")
        print("Example: python3 check_g1_mode.py eth0")
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

    print_motion_switcher_state()
    print_loco_state()
    print_low_state()


if __name__ == "__main__":
    main()