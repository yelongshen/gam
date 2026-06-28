import ARKit
import Foundation
import Network

// Joint order — matches HandSkeleton.JointName order used in ImmersiveView
// (indices 0-26, same order as before)

@MainActor
class HandStreamer: ObservableObject {
    static let shared = HandStreamer()

    // ── UI state ──────────────────────────────────────────────────────────────
    @Published var streaming    = false
    @Published var leftActive   = false
    @Published var rightActive  = false
    @Published var hostIP       = "192.168.1.13"
    @Published var portStr      = "9870"
    @Published var errorMessage: String? = nil
    @Published var hz: Double   = 0.0
    @Published var sessionStatus: String = "idle"
    @Published var debugLines: [String] = []

    // ── Internals ─────────────────────────────────────────────────────────────
    private(set) var connection: NWConnection?
    private var hzTask: Task<Void, Never>?
    var frameCount  = 0
    var lastHzTime  = Date()

    func log(_ msg: String) {
        let ts = String(format: "%.1f", Date().timeIntervalSince1970.truncatingRemainder(dividingBy: 1000))
        let line = "[\(ts)] \(msg)"
        debugLines.append(line)
        if debugLines.count > 14 { debugLines.removeFirst() }
        print(line)
    }

    // ── UDP setup (called after ImmersiveSpace opens) ─────────────────────────
    func setupUDP() {
        log("setupUDP \(hostIP):\(portStr)")
        errorMessage = nil
        guard let port = UInt16(portStr), port > 0 else {
            errorMessage = "Invalid port"; return
        }
        let endpoint = NWEndpoint.hostPort(
            host: NWEndpoint.Host(hostIP),
            port: NWEndpoint.Port(rawValue: port)!)
        let params = NWParameters.udp
        params.allowLocalEndpointReuse = true
        connection = NWConnection(to: endpoint, using: params)
        connection?.start(queue: .global())

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
        log("UDP ready")
    }

    func stop() {
        hzTask?.cancel()
        connection?.cancel()
        connection = nil
        streaming   = false
        leftActive  = false
        rightActive = false
        hz = 0
    }

    // ── Send a hand packet ────────────────────────────────────────────────────
    func send(hand: String, joints: [[Float]]) {
        let pkt: [String: Any] = [
            "hand":   hand,
            "joints": joints,
            "t":      Date().timeIntervalSince1970,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: pkt) else { return }
        connection?.send(content: data, completion: .contentProcessed { _ in })
    }
}
