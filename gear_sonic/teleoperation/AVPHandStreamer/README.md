# AVPHandStreamer — visionOS App

Streams Apple Vision Pro hand joint positions to the Ubuntu PC via UDP
for G1 Dex3 robot hand teleoperation.

## Requirements

- Xcode 15+ (visionOS SDK 1.0+)
- Apple Vision Pro (physical device) — hand tracking does not work in simulator
- Apple Developer account (free tier works for direct device install)

## Build & Install

1. **Open the project in Xcode on a Mac:**
   ```
   Open AVPHandStreamer/AVPHandStreamer.xcodeproj
   ```
   (You need to create the Xcode project — see "Create Xcode Project" below)

2. **Set your Team** in Signing & Capabilities → Team

3. **Connect AVP to Mac** via USB-C or use wireless pairing

4. **Select Apple Vision Pro** as the run destination

5. **Run** (⌘R) — trust the developer on AVP if prompted

## Create Xcode Project

Since this repo contains only the Swift source files, create the Xcode project:

1. Open Xcode → **File → New → Project**
2. Choose **visionOS → App**
3. Product Name: `AVPHandStreamer`
4. Bundle ID: `com.yourname.AVPHandStreamer`
5. Language: **Swift**, Interface: **SwiftUI**
6. **Replace** the generated files with the ones in this folder:
   - `AVPHandStreamerApp.swift`
   - `ContentView.swift`
   - `HandStreamer.swift`
   - `Info.plist` (merge — add the `NSHandsTrackingUsageDescription` key)

7. In **Signing & Capabilities**, add the **Hand Tracking** entitlement:
   - Click **+ Capability**
   - Search for and add **Hand Tracking**

## Usage

1. Start the Ubuntu PC receiver first:
   ```bash
   python gear_sonic/teleoperation/avp_g1_dex3_teleop.py --net enp36s0f1
   ```
   Note the PC IP printed at startup.

2. On AVP, open **AVPHandStreamer**

3. Enter the **Ubuntu PC IP address** and port `9870`

4. Tap **Start Streaming**

5. Green dots = hands tracked and streaming

## Packet Format

Each UDP packet is a JSON object:
```json
{
  "hand":   "left" | "right",
  "joints": [[x,y,z], ...],   // 27 joints, meters, world frame
  "t":      1234567890.123    // Unix timestamp (seconds)
}
```

Joint order matches `HandSkeleton.JointName.allCases`:
```
 0  wrist
 1  thumbKnuckle
 2  thumbIntermediateBase
 3  thumbIntermediateTip
 4  thumbTip
 5  indexFingerMetacarpal
 6  indexFingerKnuckle
 7  indexFingerIntermediateBase
 8  indexFingerIntermediateTip
 9  indexFingerTip
10  middleFingerMetacarpal
11  middleFingerKnuckle
...
26  forearmArm
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Hand tracking permission denied | Settings → Privacy → Hand Tracking → enable app |
| UDP packets not received on PC | Check firewall: `sudo ufw allow 9870/udp` |
| High latency | Make sure AVP and PC are on same WiFi network (5GHz preferred) |
| "ARKit error" | Ensure Hand Tracking entitlement is added in Xcode |
