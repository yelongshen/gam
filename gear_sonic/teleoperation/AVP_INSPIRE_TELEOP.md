# AVP -> Inspire hand teleoperation (retarget on the PC, publish from the robot PC)

Status (2026-10-12): the software path was tested on one PC with a dry-run relay (fake hands). **Not yet run on the real
hands.** The thumb mapping has not been confirmed against a real hand.

## 1. Architecture

```
Apple Vision Pro            PC (192.168.8.192)                               ROBOT PC (unitree@ubuntu)
AVPHandStreamer app         avp_g1_inspire_teleop.py --relay 192.168.8.227   eth0  192.168.123.164  (G1 DDS network)
   |                         - UDP receiver (hand joints)                     wlan0 192.168.8.227   (shared Wi-Fi)
   |--- Wi-Fi UDP :9870 ---> - wrist frame + DexPilot retargeting             inspire_relay.py (no torch)
                             - settle / ON-OFF ramps                           - UDP :9871 -> DDS rt/inspire/cmd (eth0)
                             - real hand pose comes back  <--- UDP :9871 ----  - rt/inspire/state -> UDP back
                                                                               inspire_g1 (DFX_inspire_service)
                                                                                 q x 1000 -> RS485 -> Inspire hands
                                                                                 (right /dev/ttyUSB1, left /dev/ttyUSB2)
```

Why this split: the G1's DDS lives on `eth0` (192.168.123.x), which the PC (192.168.8.x only) is not on. The retargeting
needs `torch`, `pinocchio`, `nlopt`, `dex_retargeting`, which are installed on the PC but not on the robot PC. The relay
needs only the Python standard library plus `unitree_sdk2py`.

Hand API: 6 values per hand, order `[pinky, ring, middle, index, thumb_bend, thumb_rotation]`, each 0..1 with 1 = open.
DDS motor ids: right hand 0-5, left hand 6-11. The service writes `q x 1000` (clipped 0..1000) to register 0x05CE (1486).

## 2. Retargeting

`DexPilot` from the `dex_retargeting` library (pip 0.5.0), config `assets/inspire_hand/inspire_hand.yml` with the Inspire
URDFs (from `unitreerobotics/xr_teleoperate`, Apache-2.0):
1. 25 AVP joints -> wrist-local frame built from the joint geometry -> rotated into the URDF hand frame.
2. 15 target vectors (10 fingertip pairs + 5 wrist-to-tip), solved for the 6 actuated joints (SLSQP, pinocchio FK, joint
   limits, DexPilot pinch handling at 3 cm / 5 cm, low-pass 0.2, scaling 1.2). Mimic joints are filled in afterwards.
3. radians -> 0..1 with `(max - q) / (max - min)`; ranges: fingers 0..1.7, thumb bend 0..0.5, thumb rotation -0.1..1.3.

Differences from Unitree's `xr_teleoperate`: the wrist frame comes from joint geometry (the app streams positions only, not
the headset's wrist orientation), and the pip `dex_retargeting` replaces Unitree's fork (solver identical; the yml keys are
renamed when loading).

## 3. One-time setup

| Machine | What |
|---|---|
| Robot PC | `inspire_g1` service running (Unitree `DFX_inspire_service`; check with `python3 gear_sonic_deploy/scripts/read_inspire_state.py --iface eth0 --variant dfx --seconds 5`). `git pull` in `~/gam` for `inspire_relay.py`. |
| PC | `.venv_teleop` has `dex_retargeting`, `pin`, `nlopt`, `torch`, `unitree_sdk2py`. |
| Headset | AVPHandStreamer app installed (Xcode build, see `AVPHandStreamer/README.md`). |

## 4. Run order

| # | Where | Command / action | Wait for |
|---|---|---|---|
| 0 | Robot PC | `inspire_g1` service already running | |
| 1 | Robot PC | `python3 gear_sonic/teleoperation/inspire_relay.py --iface eth0 --allow 192.168.8.192` (leave running) | `[relay] UDP :9871 -> DDS rt/inspire/cmd on eth0` |
| 2 | PC | `.venv_teleop/bin/python gear_sonic/teleoperation/avp_g1_inspire_teleop.py --relay 192.168.8.227 --check` | the real hand pose printed; `NO state from the relay` = stop and fix |
| 3 | PC | optional dry check: `... avp_g1_inspire_teleop.py --print-only --viz`, start streaming (step 5), look at the window, then close it | colored hand follows the gray hand |
| 4 | PC | `... avp_g1_inspire_teleop.py --relay 192.168.8.227 --settle 3 --lost open --off-ramp 2 --lost-grace 1 --viz` | `hands now: right [...] left [...] -> settle 3.0s` |
| 5 | Headset | open AVPHandStreamer, enter **192.168.8.192**, port **9870**, tap **Start Streaming**, hold the hands in view | `right hand ON -> settling over 3.0s` |
| 6 | | operate | |

Order rationale: the relay first, because the PC script refuses to start without the real hand pose; the check before
anything can move; the headset last, because the hands start moving when a tracked hand first appears (that starts the
settle from the real pose). Steps 3 and 4 both listen on UDP 9870: close one before starting the other.

### Stopping
1. Headset: stop streaming or lower the hands; after `--lost-grace` s without tracking, that hand opens over `--off-ramp` s.
2. PC: Ctrl+C. Both hands ramp open over `--off-ramp` s; a second Ctrl+C opens them immediately.
3. Robot PC: Ctrl+C on the relay last. The hands stay where they are.
If the PC script dies, the relay stops publishing after 0.5 s and the hands hold their last position.

## 5. Options (`avp_g1_inspire_teleop.py`)

| Option | Default | Meaning |
|---|---|---|
| `--relay HOST[:PORT]` | off | send over UDP to `inspire_relay.py` (port 9871) instead of publishing DDS from this machine |
| `--net` | `enp4s0` | NIC for DDS (only without `--relay`) |
| `--port` | 9870 | UDP port the headset streams to |
| `--settle` (`--ramp`) | 3.0 s | move from the hand's **actual** pose to the tracked pose when a hand first appears (smoothstep; the target keeps tracking) |
| `--lost hold\|open` | `open` | what a hand does once OFF |
| `--lost-grace` | 1.0 s | no tracking for this long = OFF; shorter dropouts hold the last command, no new settle |
| `--off-ramp` | 2.0 s | OFF ramp to open, also used on exit; 0 = immediately |
| `--assume-open` | off | start without a hand pose, assuming open (otherwise it refuses) |
| `--print-only` / `--viz` / `--sim` | | retarget and print only / simulation window / fake hands |
| `--check` | | read the hand pose (via DDS or the relay) and exit |
| `--type vector` | dexpilot | vector retargeting instead of DexPilot |

Relay (`inspire_relay.py`): `--iface`, `--port`, `--rate`, `--state-hz`, `--timeout` (0.5 s), `--allow IP...`, `--dry-run`.

## 6. Relay protocol
UDP JSON datagrams. PC -> relay: `{"type":"hello"}` and `{"type":"cmd","seq":ms,"q":[12]}`. Relay -> PC:
`{"type":"state","t":unix,"q":[12]}`. `q` order is the DDS ids (right 0-5, left 6-11). The relay clamps to 0..1, drops
malformed / non-finite / out-of-order packets, restricts sources with `--allow`, and stops publishing when commands stop.

## 7. Safety
- Keep fingers clear of the hands; try one hand first.
- The service applies **both** hands from every message and treats a missing value as 0 (closed). The script therefore
  never leaves a hand unspecified: an untracked hand is re-sent at its own current pose.
- The UDP link is unauthenticated; `--allow` is a source-IP filter only. Use a trusted network.
- The relay's watchdog holds the hands (it does not open them); `--lost open` opens only on the PC's command.

## 8. Troubleshooting
| Symptom | Check |
|---|---|
| `NO state from the relay` | relay running? `--allow` has the PC's IP? UDP 9871 reachable (Wi-Fi isolation / firewall)? |
| relay shows no peer | PC script not started, or wrong IP (use the robot PC's **wlan0** 192.168.8.227) |
| script exits "no hand state" | `inspire_g1` not running, or the relay is not on the DDS network of the service (`--iface eth0`) |
| window shows "(no data)" | headset not streaming to 192.168.8.192:9870; port taken by another viewer; `sudo ufw allow 9870/udp` |
| left and right swapped | swap `serial1` and `serial2` in `inspire_g1.cpp` and rebuild |
| `ModuleNotFoundError: dex_retargeting` on the robot PC | expected: do not run the teleop script there; use the relay |

## 9. Not verified yet
- Real hands (service, serial ports, command direction, speed/force limits).
- UDP 9871 between the PC and the robot PC.
- Thumb behaviour: DexPilot responds weakly on the two-joint Inspire thumb (in a round-trip test on Inspire's own geometry,
  thumb rotation 1.3 rad came back as about 0.37); a possible fix is to drive the two thumb joints directly from the human
  thumb angles.

## 10. Files
`gear_sonic/teleoperation/avp_g1_inspire_teleop.py`, `inspire_relay.py`, `visualize_avp_stream.py` (raw viewer),
`assets/inspire_hand/`, `AVPHandStreamer/` (headset app). The Dex3 equivalent is `avp_g1_dex3_teleop.py`
(`rt/dex3/{left,right}/cmd`, PD gains, no relay).

## 11. Added after the first version (2026-10-14 / 16)

- `avp_g1_inspire_teleop.py`: `--alpha` (retargeting low-pass, 1 = off), `--project-dist` / `--escape-dist` (DexPilot pinch
  detection; a thumb closer than 3 cm to a fingertip makes DexPilot curl that finger, which can leave a finger bent on an open
  hand; `--type vector` has no pinch logic), `--record FILE.npz` (raw joints, retargeted values and pinch flags, for offline
  debugging), and a status line with the actual hand pose and a warning when a finger does not follow its command.
- `inspire_read_params.py`: read-only dump of the hand's speed / force / angle registers (stop `inspire_g1` first).
- `pico_trigger_inspire.py`: PICO controller triggers open/close the Inspire hands over the same relay (arming, speed limit,
  close limit, open ramp when controller data stops). Tested only against a simulated headset and fake hands.
- `xrt_reader_test.py`: starts several XRoboToolkit SDK readers in parallel (two readers start and poll without errors; data
  sharing is unverified until the headset is sending).
