import SwiftUI

struct ContentView: View {
    @ObservedObject private var streamer = HandStreamer.shared
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @State private var spaceIsOpen = false

    var body: some View {
        VStack(spacing: 16) {
            Text("AVP → G1 Hand Teleop").font(.title)

            HStack(spacing: 40) {
                VStack {
                    Circle().fill(streamer.leftActive ? Color.green : Color.gray)
                        .frame(width: 24, height: 24)
                    Text("Left").font(.caption)
                }
                VStack {
                    Circle().fill(streamer.rightActive ? Color.green : Color.gray)
                        .frame(width: 24, height: 24)
                    Text("Right").font(.caption)
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Label("Host IP", systemImage: "network").font(.headline)
                TextField("192.168.1.13", text: $streamer.hostIP)
                    .textFieldStyle(.roundedBorder).frame(width: 200)
                HStack {
                    Label("Port", systemImage: "number")
                    TextField("9870", text: $streamer.portStr)
                        .textFieldStyle(.roundedBorder).frame(width: 80)
                }
            }

            // Direct openImmersiveSpace — no pre-flight auth call (matches Apple HappyBeam pattern)
            Button(spaceIsOpen ? "Stop Streaming" : "Start Streaming") {
                Task {
                    if spaceIsOpen {
                        streamer.log("Dismissing space")
                        await dismissImmersiveSpace()
                        spaceIsOpen = false
                        streamer.stop()
                    } else {
                        streamer.log("→ openImmersiveSpace()")
                        switch await openImmersiveSpace(id: "HandTracking") {
                        case .opened:
                            streamer.log("✅ space opened")
                            spaceIsOpen = true
                            streamer.setupUDP()
                        case .error:
                            streamer.log("❌ space FAILED")
                        case .userCancelled:
                            streamer.log("space cancelled")
                        @unknown default:
                            streamer.log("space unknown result")
                        }
                    }
                }
            }
            .buttonStyle(.borderedProminent)
            .tint(spaceIsOpen ? .red : .blue)

            Text("Hz: \(streamer.hz, specifier: "%.0f")  |  \(streamer.sessionStatus)")
                .font(.caption).foregroundColor(.secondary)

            Divider()

            VStack(alignment: .leading, spacing: 2) {
                Text("Debug Log:").font(.caption).bold()
                ForEach(streamer.debugLines, id: \.self) { line in
                    Text(line).font(.system(size: 11, design: .monospaced))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if let err = streamer.errorMessage {
                Text(err).foregroundColor(.red).font(.caption)
            }
        }
        .padding(40)
        .frame(minWidth: 520, minHeight: 480)
    }
}
