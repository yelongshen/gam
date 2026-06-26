import SwiftUI

struct ContentView: View {
    @StateObject private var streamer = HandStreamer()

    var body: some View {
        VStack(spacing: 20) {
            Text("AVP → G1 Hand Teleop")
                .font(.title)

            HStack(spacing: 40) {
                VStack {
                    Circle()
                        .fill(streamer.leftActive ? Color.green : Color.gray)
                        .frame(width: 24, height: 24)
                    Text("Left Hand").font(.caption)
                }
                VStack {
                    Circle()
                        .fill(streamer.rightActive ? Color.green : Color.gray)
                        .frame(width: 24, height: 24)
                    Text("Right Hand").font(.caption)
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Label("Host IP", systemImage: "network")
                    .font(.headline)
                TextField("192.168.1.100", text: $streamer.hostIP)
                    .textFieldStyle(.roundedBorder)
                    .keyboardType(.decimalPad)
                    .frame(width: 200)

                HStack {
                    Label("Port", systemImage: "number")
                    TextField("9870", text: $streamer.portStr)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 80)
                }
            }

            Button(streamer.streaming ? "Stop Streaming" : "Start Streaming") {
                if streamer.streaming {
                    streamer.stop()
                } else {
                    streamer.start()
                }
            }
            .buttonStyle(.borderedProminent)
            .tint(streamer.streaming ? .red : .blue)

            if streamer.streaming {
                Text("Streaming at \(streamer.hz, specifier: "%.0f") Hz")
                    .foregroundColor(.secondary)
                    .font(.caption)
            }

            if let err = streamer.errorMessage {
                Text(err)
                    .foregroundColor(.red)
                    .font(.caption)
            }
        }
        .padding(40)
        .frame(minWidth: 380, minHeight: 340)
        .onDisappear { streamer.stop() }
    }
}
