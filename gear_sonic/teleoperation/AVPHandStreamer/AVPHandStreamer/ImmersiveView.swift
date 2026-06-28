import ARKit
import RealityKit
import SwiftUI

// ImmersiveView owns the ARKitSession and HandTrackingProvider.
// @State keeps them alive for the lifetime of the ImmersiveSpace.
// Matches Apple's "Animating hand models in visionOS" official sample pattern.
struct ImmersiveView: View {
    // @State — kept alive by SwiftUI, NOT recreated on each view update
    @State private var arkitSession = ARKitSession()
    @State private var handTrackingProvider = HandTrackingProvider()

    // Joint names in order — must match the Python receiver's expected order
    private let jointNames: [HandSkeleton.JointName] = [
        .wrist,
        .thumbKnuckle, .thumbIntermediateBase, .thumbIntermediateTip, .thumbTip,
        .indexFingerMetacarpal, .indexFingerKnuckle, .indexFingerIntermediateBase,
        .indexFingerIntermediateTip, .indexFingerTip,
        .middleFingerMetacarpal, .middleFingerKnuckle, .middleFingerIntermediateBase,
        .middleFingerIntermediateTip, .middleFingerTip,
        .ringFingerMetacarpal, .ringFingerKnuckle, .ringFingerIntermediateBase,
        .ringFingerIntermediateTip, .ringFingerTip,
        .littleFingerMetacarpal, .littleFingerKnuckle, .littleFingerIntermediateBase,
        .littleFingerIntermediateTip, .littleFingerTip,
        .forearmWrist, .forearmArm,
    ]

    var body: some View {
        RealityView { _ in }
            .onAppear {
                Task { await HandStreamer.shared.log("ImmersiveView.onAppear ✅") }
            }
            .onDisappear {
                Task { await HandStreamer.shared.log("ImmersiveView.onDisappear") }
            }
            .task {
                await startHandTracking()
            }
    }

    // Matches Apple "Animating hand models" startHandTracking() method exactly
    private func startHandTracking() async {
        await HandStreamer.shared.log("ImmersiveView: startHandTracking")
        await HandStreamer.shared.sessionStatus = "starting ARKit..."
        do {
            try await arkitSession.run([handTrackingProvider])
        } catch {
            await HandStreamer.shared.log("ARKit run error: \(error)")
            await HandStreamer.shared.sessionStatus = "ARKit error: \(error.localizedDescription)"
            return
        }
        await HandStreamer.shared.log("ARKit running — iterating updates")
        await HandStreamer.shared.sessionStatus = "waiting for hands"
        for await update in handTrackingProvider.anchorUpdates {
            await processUpdate(update)
        }
        await HandStreamer.shared.log("anchorUpdates stream ended")
    }

    @MainActor
    private func processUpdate(_ update: AnchorUpdate<HandAnchor>) {
        let anchor = update.anchor
        let streamer = HandStreamer.shared

        guard anchor.isTracked, let skeleton = anchor.handSkeleton else {
            if anchor.chirality == .left  { streamer.leftActive  = false }
            if anchor.chirality == .right { streamer.rightActive = false }
            return
        }

        streamer.sessionStatus = "tracking"

        let originFromAnchor = anchor.originFromAnchorTransform
        var joints: [[Float]] = []
        for name in jointNames {
            let joint = skeleton.joint(name)
            let worldTransform = originFromAnchor * joint.anchorFromJointTransform
            let pos = worldTransform.columns.3
            joints.append([pos.x, pos.y, pos.z])
        }

        let hand = (anchor.chirality == .left) ? "left" : "right"
        streamer.send(hand: hand, joints: joints)

        if anchor.chirality == .left  { streamer.leftActive  = true }
        if anchor.chirality == .right { streamer.rightActive = true }
        streamer.frameCount += 1
    }
}
