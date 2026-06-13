import argparse
import socket
import sys
import time

import numpy as np

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_, unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


G1_NUM_MOTOR = 29
DEX3_MOTOR_NUM = 7
CONTROL_DT = 0.01

LEG_KP = [60, 60, 60, 100, 40, 40, 60, 60, 60, 100, 40, 40]
LEG_KD = [1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1]
WAIST_KP = [60, 0, 0]
WAIST_KD = [1, 0, 0]
ARM_KP = [30, 30, 30, 30, 20, 15, 15, 30, 30, 30, 30, 20, 15, 15]
ARM_KD = [1.2] * 14
HAND_KP = [0.6] * DEX3_MOTOR_NUM
HAND_KD = [0.08] * DEX3_MOTOR_NUM
HAND_KP[0] = 1.0

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
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
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


class Mode:
    PR = 0


LEG_JOINTS = list(range(12))
WAIST_JOINTS = [G1JointIndex.WaistYaw, G1JointIndex.WaistRoll, G1JointIndex.WaistPitch]
ARM_JOINTS = list(range(G1JointIndex.LeftShoulderPitch, G1JointIndex.RightWristYaw + 1))

LIFT_ARMS_TARGET = {
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
}

HANDSHAKE_TARGET = {
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
}

HAND_OPEN_TARGET = np.zeros(DEX3_MOTOR_NUM)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run conservative low-level G1 arm + Dex3 hand actions.")
    parser.add_argument("action", choices=("lift", "handshake", "lift_handshake"))
    parser.add_argument("network_interface", nargs="?", help="Optional DDS network interface, for example eth0.")
    parser.add_argument("--release-mode", action="store_true", help="Release active MotionSwitcher mode before low-level control.")
    parser.add_argument("--hold", type=float, default=1.0, help="Seconds to hold the lift pose before the next stage.")
    return parser.parse_args()


def maybe_release_motion_mode(enabled):
    if not enabled:
        print("Not releasing MotionSwitcher mode. If low-level commands are ignored, rerun with --release-mode.")
        return

    client = MotionSwitcherClient()
    client.SetTimeout(5.0)
    client.Init()
    status, result = client.CheckMode()
    print(f"MotionSwitcher before release: status={status}, result={result}")
    while result and result.get("name"):
        client.ReleaseMode()
        time.sleep(1.0)
        status, result = client.CheckMode()
        print(f"MotionSwitcher after release attempt: status={status}, result={result}")


def wait_for_state(state, key, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state[key] is not None:
            return True
        time.sleep(0.05)
    return False


def measured_body_pose(low_state):
    return np.array([low_state.motor_state[index].q for index in range(G1_NUM_MOTOR)], dtype=float)


def measured_hand_pose(hand_state):
    return np.array([hand_state.motor_state[index].q for index in range(DEX3_MOTOR_NUM)], dtype=float)


def smoothstep(ratio):
    return ratio * ratio * (3.0 - 2.0 * ratio)


def blend_vector(start, target, ratio):
    smooth_ratio = smoothstep(ratio)
    return (1.0 - smooth_ratio) * start + smooth_ratio * target


def body_target_from_arm_pose(base_body, arm_target):
    target = base_body.copy()
    for joint, q in arm_target.items():
        target[joint] = q
    return target


def make_hand_mode(motor_index):
    status = 0x01
    timeout = 0x00
    mode = motor_index & 0x0F
    mode |= status << 4
    mode |= timeout << 7
    return mode


def make_hand_cmd(q_target):
    cmd = unitree_hg_msg_dds__HandCmd_()
    for index in range(DEX3_MOTOR_NUM):
        q = float(q_target[index])
        motor = cmd.motor_cmd[index]
        motor.mode = make_hand_mode(index)
        motor.q = q
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = HAND_KP[index]
        motor.kd = HAND_KD[index]
    return cmd


def make_low_cmd(q_target, mode_machine, crc):
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = Mode.PR
    cmd.mode_machine = mode_machine

    for index in range(G1_NUM_MOTOR):
        motor = cmd.motor_cmd[index]
        motor.mode = 1
        motor.tau = 0.0
        motor.q = float(q_target[index])
        motor.dq = 0.0
        if index in LEG_JOINTS:
            motor.kp = LEG_KP[index]
            motor.kd = LEG_KD[index]
        elif index in WAIST_JOINTS:
            waist_index = index - G1JointIndex.WaistYaw
            motor.kp = WAIST_KP[waist_index]
            motor.kd = WAIST_KD[waist_index]
        else:
            arm_index = index - G1JointIndex.LeftShoulderPitch
            motor.kp = ARM_KP[arm_index]
            motor.kd = ARM_KD[arm_index]

    cmd.crc = crc.Crc(cmd)
    return cmd


def publish_targets(publishers, body_q, left_hand_q, right_hand_q, mode_machine, crc):
    lowcmd_pub, left_hand_pub, right_hand_pub = publishers
    lowcmd_pub.Write(make_low_cmd(body_q, mode_machine, crc))
    left_hand_pub.Write(make_hand_cmd(left_hand_q))
    right_hand_pub.Write(make_hand_cmd(right_hand_q))


def ramp_targets(publishers, start, target, duration, mode_machine, crc):
    start_body, start_left_hand, start_right_hand = start
    target_body, target_left_hand, target_right_hand = target

    start_time = time.monotonic()
    while True:
        ratio = min((time.monotonic() - start_time) / duration, 1.0)
        body_q = blend_vector(start_body, target_body, ratio)
        left_hand_q = blend_vector(start_left_hand, target_left_hand, ratio)
        right_hand_q = blend_vector(start_right_hand, target_right_hand, ratio)
        publish_targets(publishers, body_q, left_hand_q, right_hand_q, mode_machine, crc)
        if ratio >= 1.0:
            break
        time.sleep(CONTROL_DT)


def hold_targets(publishers, target, duration, mode_machine, crc):
    body_q, left_hand_q, right_hand_q = target
    stop_time = time.monotonic() + duration
    while time.monotonic() < stop_time:
        publish_targets(publishers, body_q, left_hand_q, right_hand_q, mode_machine, crc)
        time.sleep(CONTROL_DT)


def run_handshake_wave(publishers, body_q, left_hand_q, right_hand_q, duration, mode_machine, crc):
    start_time = time.monotonic()
    amplitude = 0.10
    frequency = 1.0
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= duration:
            break
        wave_body = body_q.copy()
        offset = amplitude * np.sin(2.0 * np.pi * frequency * elapsed)
        wave_body[G1JointIndex.RightElbow] = body_q[G1JointIndex.RightElbow] + offset
        wave_body[G1JointIndex.RightWristRoll] = body_q[G1JointIndex.RightWristRoll] + 0.5 * offset
        publish_targets(publishers, wave_body, left_hand_q, right_hand_q, mode_machine, crc)
        time.sleep(CONTROL_DT)


def main():
    args = parse_args()

    print("WARNING: This uses low-level rt/lowcmd for the G1 body plus Dex3 hand cmd topics.")
    print("It commands all 29 body joints, holding legs at the measured start pose while moving arms.")
    print("Run only with the robot standing stable, arms/hands clear, and no other lowcmd or hand publisher active.")
    print("The handshake motion is a no-contact demo. Do not use it for touching a person.")
    input("Press Enter to continue, or Ctrl+C to cancel...")

    initialize_channel(args.network_interface)
    maybe_release_motion_mode(args.release_mode)

    state = {"low_state": None, "left_hand": None, "right_hand": None}
    lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
    left_hand_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
    right_hand_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)
    lowstate_sub.Init(lambda msg: state.__setitem__("low_state", msg), 10)
    left_hand_sub.Init(lambda msg: state.__setitem__("left_hand", msg), 10)
    right_hand_sub.Init(lambda msg: state.__setitem__("right_hand", msg), 10)

    lowcmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    left_hand_pub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
    right_hand_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
    lowcmd_pub.Init()
    left_hand_pub.Init()
    right_hand_pub.Init()

    print("Waiting for rt/lowstate and Dex3 hand states...")
    if not wait_for_state(state, "low_state"):
        print("Timed out waiting for rt/lowstate. No command was sent.")
        sys.exit(1)
    if not wait_for_state(state, "left_hand") or not wait_for_state(state, "right_hand"):
        print("Timed out waiting for Dex3 hand states. No command was sent.")
        sys.exit(1)

    crc = CRC()
    mode_machine = state["low_state"].mode_machine
    start_body = measured_body_pose(state["low_state"])
    start_left_hand = measured_hand_pose(state["left_hand"])
    start_right_hand = measured_hand_pose(state["right_hand"])
    start = (start_body, start_left_hand, start_right_hand)

    lift_body = body_target_from_arm_pose(start_body, LIFT_ARMS_TARGET)
    handshake_body = body_target_from_arm_pose(start_body, HANDSHAKE_TARGET)
    lift_target = (lift_body, HAND_OPEN_TARGET, HAND_OPEN_TARGET)
    handshake_target = (handshake_body, HAND_OPEN_TARGET, HAND_OPEN_TARGET)

    publishers = (lowcmd_pub, left_hand_pub, right_hand_pub)

    print(f"Running low-level action: {args.action}")
    if args.action == "lift":
        ramp_targets(publishers, start, lift_target, 4.0, mode_machine, crc)
        hold_targets(publishers, lift_target, args.hold, mode_machine, crc)
        final_target = lift_target
    elif args.action == "handshake":
        ramp_targets(publishers, start, handshake_target, 4.0, mode_machine, crc)
        run_handshake_wave(publishers, handshake_body, HAND_OPEN_TARGET, HAND_OPEN_TARGET, 3.0, mode_machine, crc)
        final_target = handshake_target
    else:
        ramp_targets(publishers, start, lift_target, 4.0, mode_machine, crc)
        hold_targets(publishers, lift_target, args.hold, mode_machine, crc)
        ramp_targets(publishers, lift_target, handshake_target, 3.0, mode_machine, crc)
        run_handshake_wave(publishers, handshake_body, HAND_OPEN_TARGET, HAND_OPEN_TARGET, 3.0, mode_machine, crc)
        final_target = handshake_target

    print("Returning to measured starting body and hand state.")
    ramp_targets(publishers, final_target, start, 4.0, mode_machine, crc)
    hold_targets(publishers, start, 0.5, mode_machine, crc)
    print("Done. Low-level action completed and script exited.")


if __name__ == "__main__":
    main()