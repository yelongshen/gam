import ARKit
import Foundation
import Network
import RealityKit

// Joint order matches HandSkeleton.JointName.allCases on visionOS
// Index → name:
//  0  wrist
//  1  thumbKnuckle
//  2  thumbIntermediateBase
//  3  thumbIntermediateTip
//  4  thumbTip
//  5  indexFingerMetacarpal
//  6  indexFingerKnuckle
//  7  indexFingerIntermediateBase
//  8  indexFingerIntermediateTip
//  9  indexFingerTip
// 10  middleFingerMetacarpal
// 11  middleFingerKnuckle
// 12  middleFingerIntermediateBase
// 13  middleFingerIntermediateTip
// 14  middleFingerTip
// 15  ringFingerMetacarpal
// 16  ringFingerKnuckle
// 17  ringFingerIntermediateBase
// 18  ringFingerIntermediateTip
// 19  ringFingerTip
// 20  littleFingerMetacarpal
// 21  littleFingerKnuckle
// 22  littleFingerIntermediateBase
// 23  littleFingerIntermediateTip
// 24  littleFingerTip
// 25  forearmWrist
// 26  forearmArm

@MainActor
class HandStreamer: ObservableObject {
    // ── UI state ─────────────────────────────────────────────────────────────
    @Published var streaming    = false
    @Published var leftActive   = false
    @Published var rightActive  = false
    @Published var hostIP       = "192.168.1.100"
    @Published var portStr      = "9870"
    @Published var errorMessage: String? = nil
    @Published var hz: Double   = 0.0

    // ── Internals ─────────────────────────────────────────────────────────────
    private let arkitSession    = ARKitSession()
    private let handTracking    = HandTrackingProvider()
    private var connection: NWConnection?
    private var updateTask: Task<Void, Never>?
    private var hzTask:     Task<Void, Never>?
    private var frameCount  = 0
    private var lastHzTime  = Date()

    // ── Joint name list (ordered) ─────────────────────────────────────────────
    private let jointNames: [HandSkeleton.JointName] = [
        .wrist,
        .thumbKnuckle,
        .thumbIntermediateBase,
        .thumbIntermediateTip,
        .thumbTip,
        .indexFingerMetacarpal,
        .indexFingerKnuckle,
        .indexFingerIntermediateBase,
        .indexFingerIntermediateTip,
        .indexFingerTip,
        .middleFingerMetacarpal,
        .middleFingerKnuckle,
        .middleFingerIntermediateBase,
        .middleFingerIntermediateTip,
        .middleFingerTip,
        .ringFingerMetacarpal,
        .ringFingerKnuckle,
        .ringFingerIntermediateBase,
        .ringFingerIntermediateTip,
        .ringFingerTip,
        .littleFingerMetacarpal,
        .littleFingerKnuckle,
        .littleFingerIntermediateBase,
        .littleFingerIntermediateTip,
        .littleFingerTip,
        .forearmWrist,
        .forearmArm,
    ]

    // ── Start / Stop ──────────────────────────────────────────────────────────

    func start() {
        guard !streaming else { return }
        errorMessage = nil

        guard let port = UInt16(portStr), port > 0 else {
            errorMessage = "Invalid port number"
            return
        }
        guard !hostIP.isEmpty else {
            errorMessage = "Enter the Ubuntu PC IP address"
            return
        }

        // Set up UDP connection
        let endpoint = NWEndpoint.hostPort(
            host: NWEndpoint.Host(hostIP),
            port: NWEndpoint.Port(rawValue: port)!)
        let params = NWParameters.udp
        params.allowLocalEndpointReuse = true
        connection = NWConnection(to: endpoint, using: params)
        connection?.start(queue: .global())

        // Start ARKit hand tracking
        updateTask = Task {
            do {
                try await arkitSession.run([handTracking])
                for await update in handTracking.anchorUpdates {
                    await handleAnchorUpdate(update)
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "ARKit error: \(error.localizedDescription)"
                    self.streaming = false
                }
            }
        }

        // Hz counter
        lastHzTime = Date()
        frameCount = 0
        hzTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                let now = Date()
                let elapsed = now.timeIntervalSince(lastHzTime)
                if elapsed > 0 {
                    await MainActor.run {
                        self.hz = Double(self.frameCount) / elapsed
                        self.frameCount = 0
                        self.lastHzTime = now
                    }
                }
            }
        }

        streaming = true
    }

    func stop() {
        updateTask?.cancel()
        hzTask?.cancel()
        connection?.cancel()
        connection = nil
        streaming   = false
        leftActive  = false
        rightActive = false
        hz = 0
    }

    // ── Handle ARKit update ───────────────────────────────────────────────────

    private func handleAnchorUpdate(_ update: AnchorUpdate<HandAnchor>) {
        let anchor = update.anchor
        guard anchor.isTracked, let skeleton = anchor.handSkeleton else {
            Task { @MainActor in
                if anchor.chirality == .left  { leftActive  = false }
                if anchor.chirality == .right { rightActive = false }
            }
            return
        }

        // Extract world-space positions for all joints
        let originFromAnchor = anchor.originFromAnchorTransform
        var joints: [[Float]] = []
        for name in jointNames {
            let joint = skeleton.joint(name)
            let worldTransform = originFromAnchor * joint.anchorFromJointTransform
            let pos = worldTransform.columns.3
            joints.append([pos.x, pos.y, pos.z])
        }

        let hand = (anchor.chirality == .left) ? "left" : "right"
        let timestamp = Date().timeIntervalSince1970

        // Encode as compact JSON
        var pkt: [String: Any] = [
            "hand":   hand,
            "joints": joints,
            "t":      timestamp,
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: pkt) else { return }

        connection?.send(
            content: data,
            completion: .contentProcessed { _ in })

        Task { @MainActor in
            if anchor.chirality == .left  { leftActive  = true }
            if anchor.chirality == .right { rightActive = true }
            frameCount += 1
        }
    }
}
