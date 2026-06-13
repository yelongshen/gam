import argparse
import socket
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


CONTROL_DT = 0.02
G1_NUM_MOTOR = 29

ARM_KP = 25.0
ARM_KD = 1.2
WAIST_KP = 20.0
WAIST_KD = 1.0

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


class G1JointIndex:
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    ArmSdkEnable = 29


LEFT_ARM_JOINTS = [
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
    G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
]

RIGHT_ARM_JOINTS = [
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]

WAIST_JOINTS = [
    G1JointIndex.WaistYaw,
    G1JointIndex.WaistRoll,
    G1JointIndex.WaistPitch,
]

ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + WAIST_JOINTS

LIFT_BOTH_ARMS_TARGET = {
    G1JointIndex.LeftShoulderPitch: 0.0,
    G1JointIndex.LeftShoulderRoll: np.pi / 2.0,
    G1JointIndex.LeftShoulderYaw: 0.0,
    G1JointIndex.LeftElbow: np.pi / 2.0,
    G1JointIndex.LeftWristRoll: 0.0,
    G1JointIndex.LeftWristPitch: 0.0,
    G1JointIndex.LeftWristYaw: 0.0,
    G1JointIndex.RightShoulderPitch: 0.0,
    G1JointIndex.RightShoulderRoll: -np.pi / 2.0,
    G1JointIndex.RightShoulderYaw: 0.0,
    G1JointIndex.RightElbow: np.pi / 2.0,
    G1JointIndex.RightWristRoll: 0.0,
    G1JointIndex.RightWristPitch: 0.0,
    G1JointIndex.RightWristYaw: 0.0,
    G1JointIndex.WaistYaw: 0.0,
    G1JointIndex.WaistRoll: 0.0,
    G1JointIndex.WaistPitch: 0.0,
}

HANDSHAKE_RIGHT_ARM_TARGET = {
    G1JointIndex.LeftShoulderPitch: 0.0,
    G1JointIndex.LeftShoulderRoll: 0.25,
    G1JointIndex.LeftShoulderYaw: 0.0,
    G1JointIndex.LeftElbow: 0.6,
    G1JointIndex.LeftWristRoll: 0.0,
    G1JointIndex.LeftWristPitch: 0.0,
    G1JointIndex.LeftWristYaw: 0.0,
    G1JointIndex.RightShoulderPitch: 0.20,
    G1JointIndex.RightShoulderRoll: -0.75,
    G1JointIndex.RightShoulderYaw: 0.0,
    G1JointIndex.RightElbow: 1.25,
    G1JointIndex.RightWristRoll: 0.0,
    G1JointIndex.RightWristPitch: 0.0,
    G1JointIndex.RightWristYaw: 0.0,
    G1JointIndex.WaistYaw: 0.0,
    G1JointIndex.WaistRoll: 0.0,
    G1JointIndex.WaistPitch: 0.0,
}


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run conservative G1 arm actions over rt/arm_sdk.")
    parser.add_argument("action", choices=("lift", "handshake", "lift_handshake"), help="Arm action to run.")
    parser.add_argument("network_interface", nargs="?", help="Optional DDS network interface, for example eth0.")
    parser.add_argument("--hold", type=float, default=1.0, help="Seconds to hold the target pose before return.")
    parser.add_argument("--no-return", action="store_true", help="Do not return to the starting arm pose after the action.")
    return parser.parse_args()


def initialize_channel(network_interface):
    if network_interface:
        print(f"Using provided network interface: {network_interface}")
        ChannelFactoryInitialize(0, network_interface)
        return

    print_available_interfaces()
    default_interface = pick_default_interface()
    if default_interface is None:
        print("No usable network interface found.")
        sys.exit(1)
    print(f"No network interface provided. First usable interface is: {default_interface}")
    print("Using CycloneDDS autodetect mode for initialization.")
    ChannelFactoryInitialize(0)


def wait_for_low_state(state, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state["low_state"] is not None:
            return True
        time.sleep(0.05)
    return False


def measured_arm_pose(low_state):
    return {joint: low_state.motor_state[joint].q for joint in ARM_JOINTS}


def blend_pose(start_pose, target_pose, ratio):
    smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
    return {
        joint: (1.0 - smooth_ratio) * start_pose[joint] + smooth_ratio * target_pose[joint]
        for joint in ARM_JOINTS
    }


def build_arm_sdk_command(pose, enable_weight, crc):
    command = unitree_hg_msg_dds__LowCmd_()
    command.motor_cmd[G1JointIndex.ArmSdkEnable].q = float(enable_weight)

    for joint in ARM_JOINTS:
        motor = command.motor_cmd[joint]
        motor.tau = 0.0
        motor.q = float(pose[joint])
        motor.dq = 0.0
        motor.kp = WAIST_KP if joint in WAIST_JOINTS else ARM_KP
        motor.kd = WAIST_KD if joint in WAIST_JOINTS else ARM_KD

    command.crc = crc.Crc(command)
    return command


def publish_pose_for_duration(publisher, pose, duration, crc, enable_weight=1.0):
    command = build_arm_sdk_command(pose, enable_weight, crc)
    end_time = time.monotonic() + duration
    while time.monotonic() < end_time:
        publisher.Write(command)
        time.sleep(CONTROL_DT)


def ramp_between_poses(publisher, start_pose, target_pose, duration, crc):
    start_time = time.monotonic()
    while True:
        elapsed = time.monotonic() - start_time
        ratio = min(elapsed / duration, 1.0)
        pose = blend_pose(start_pose, target_pose, ratio)
        publisher.Write(build_arm_sdk_command(pose, 1.0, crc))
        if ratio >= 1.0:
            break
        time.sleep(CONTROL_DT)


def run_handshake_wave(publisher, base_pose, duration, crc):
    start_time = time.monotonic()
    amplitude = 0.12
    frequency = 1.0
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= duration:
            break
        pose = dict(base_pose)
        offset = amplitude * np.sin(2.0 * np.pi * frequency * elapsed)
        pose[G1JointIndex.RightElbow] = base_pose[G1JointIndex.RightElbow] + offset
        pose[G1JointIndex.RightWristRoll] = base_pose[G1JointIndex.RightWristRoll] + 0.5 * offset
        publisher.Write(build_arm_sdk_command(pose, 1.0, crc))
        time.sleep(CONTROL_DT)


def release_arm_sdk(publisher, pose, crc):
    for ratio in np.linspace(1.0, 0.0, 50):
        publisher.Write(build_arm_sdk_command(pose, ratio, crc))
        time.sleep(CONTROL_DT)


def run_action(publisher, action, start_pose, hold_duration, crc):
    if action == "lift":
        ramp_between_poses(publisher, start_pose, LIFT_BOTH_ARMS_TARGET, 4.0, crc)
        publish_pose_for_duration(publisher, LIFT_BOTH_ARMS_TARGET, hold_duration, crc)
        return LIFT_BOTH_ARMS_TARGET

    if action == "handshake":
        ramp_between_poses(publisher, start_pose, HANDSHAKE_RIGHT_ARM_TARGET, 4.0, crc)
        run_handshake_wave(publisher, HANDSHAKE_RIGHT_ARM_TARGET, 3.0, crc)
        return HANDSHAKE_RIGHT_ARM_TARGET

    ramp_between_poses(publisher, start_pose, LIFT_BOTH_ARMS_TARGET, 4.0, crc)
    publish_pose_for_duration(publisher, LIFT_BOTH_ARMS_TARGET, hold_duration, crc)
    ramp_between_poses(publisher, LIFT_BOTH_ARMS_TARGET, HANDSHAKE_RIGHT_ARM_TARGET, 3.0, crc)
    run_handshake_wave(publisher, HANDSHAKE_RIGHT_ARM_TARGET, 3.0, crc)
    return HANDSHAKE_RIGHT_ARM_TARGET


def main():
    args = parse_args()

    print("WARNING: This publishes G1 arm commands on rt/arm_sdk.")
    print("Run only with the robot standing stable, arms clear, and no other arm/SONIC publisher active.")
    print("Do not use this for contact with a person; the handshake action is a no-contact arm wave.")
    input("Press Enter to continue, or Ctrl+C to cancel...")

    initialize_channel(args.network_interface)

    state = {"low_state": None}
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(lambda msg: state.__setitem__("low_state", msg), 10)

    publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    publisher.Init()

    print("Waiting for rt/lowstate...")
    if not wait_for_low_state(state):
        print("Timed out waiting for rt/lowstate. No arm command was sent.")
        sys.exit(1)

    crc = CRC()
    start_pose = measured_arm_pose(state["low_state"])

    print(f"Running action: {args.action}")
    final_pose = run_action(publisher, args.action, start_pose, args.hold, crc)

    if args.no_return:
        print("Holding final pose because --no-return was set. Press Ctrl+C or stop this script when done.")
        while True:
            publish_pose_for_duration(publisher, final_pose, 1.0, crc)

    print("Returning to the starting arm pose.")
    ramp_between_poses(publisher, final_pose, start_pose, 4.0, crc)
    publish_pose_for_duration(publisher, start_pose, 0.5, crc)
    release_arm_sdk(publisher, start_pose, crc)
    print("Done. Released arm_sdk and exited.")


if __name__ == "__main__":
    main()