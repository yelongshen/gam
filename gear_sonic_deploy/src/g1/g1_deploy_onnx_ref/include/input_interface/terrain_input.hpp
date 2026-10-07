/**
 * @file terrain_input.hpp
 * @brief Latest `terrain` message from the onboard perception stack, for the height-map
 *        observation (terrain_scan.hpp).
 *
 * The publisher is gam gear_sonic_deploy/perception/terrain_publisher.py (default
 * tcp://127.0.0.1:5559, topic "terrain", 50 Hz, gear_sonic packed format). Fields used:
 *
 *   height_grid       f32  [rows, cols]  terrain_z - torso_z, torso heading frame, NaN = unseen
 *   height_grid_valid bool [rows, cols]
 *   grid_resolution   f32  [1]
 *   grid_origin       f32  [2]          x, y of height_grid[0][0]'s centre
 *   torso_pos         f64  [3]          odometry frame
 *   torso_quat        f64  [4]          x y z w
 *   timestamp         f64  [1]          when the publisher sampled the map (its clock)
 *   map_stamp         f64  [1]          stamp of the map it sampled (same clock)
 *   torso_grid_21       f32  [21, 21] optional: torso_z - surface_z, 10 cm, torso heading frame
 *   torso_grid_21_valid bool [21, 21] optional
 *
 * A message is usable when it arrived less than `max_msg_age_s` ago and its map is less
 * than `max_map_age_s` older than the sample (both default to a few map periods). When it
 * is not, the observation is "no map": every ray invalid and zero, which the policy was
 * trained on, rather than silently holding old heights.
 */

#ifndef TERRAIN_INPUT_HPP
#define TERRAIN_INPUT_HPP

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#include "../terrain_scan.hpp"
#include "zmq_packed_message_subscriber.hpp"

/// Command-line settings for the terrain input (set in main, read when the height-map
/// observation is configured).
struct TerrainInputOptions {
  std::string host = "127.0.0.1";
  int port = 5559;
  std::string topic = "terrain";
  double max_msg_age_s = 0.1;  // 5 samples at 50 Hz
  double max_map_age_s = 0.6;  // 3 periods of the 5 Hz map
};

inline TerrainInputOptions& GlobalTerrainInputOptions() {
  static TerrainInputOptions options;
  return options;
}

class TerrainInput {
 public:
  explicit TerrainInput(const TerrainInputOptions& options)
      : options_(options),
        subscriber_(options.host, options.port, options.topic, /*timeout_ms=*/100, /*verbose=*/false,
                    /*conflate=*/true) {
    subscriber_.SetOnDecodedMessage(
        [this](const std::string&, const ZMQPackedMessageSubscriber::DecodedHeader& header,
               const std::vector<ZMQPackedMessageSubscriber::BufferView>& buffers) { OnMessage(header, buffers); });
  }

  // The receive thread writes into msg_ under mutex_. Members are destroyed in reverse order, so
  // without this the thread could still be running after msg_ and mutex_ are gone (seen as
  // "free(): invalid size" when the runner exits; AddressSanitizer: double-free in the thread).
  ~TerrainInput() { subscriber_.Stop(); }
  TerrainInput(const TerrainInput&) = delete;
  TerrainInput& operator=(const TerrainInput&) = delete;

  /// Connects and starts the background receive thread (errors are logged by the subscriber;
  /// until messages arrive, Latest() reports kNoMessage and the scan is all invalid).
  void Start() {
    std::cout << "[TerrainInput] subscribing to tcp://" << options_.host << ":" << options_.port << " topic '"
              << options_.topic << "'" << std::endl;
    subscriber_.Start();
  }
  void Stop() { subscriber_.Stop(); }

  enum class Status { kOk, kNoMessage, kStaleMessage, kStaleMap };

  /// Copy the latest message if it is usable now.
  Status Latest(terrain_scan::TerrainMessage& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!have_msg_) return Status::kNoMessage;
    const double age = std::chrono::duration<double>(Clock::now() - recv_time_).count();
    if (age > options_.max_msg_age_s) return Status::kStaleMessage;
    if (map_age_s_ > options_.max_map_age_s) return Status::kStaleMap;
    out = msg_;
    return Status::kOk;
  }

  static const char* StatusName(Status s) {
    switch (s) {
      case Status::kOk: return "ok";
      case Status::kNoMessage: return "no terrain message yet";
      case Status::kStaleMessage: return "terrain messages stopped arriving";
      case Status::kStaleMap: return "terrain map is stale";
    }
    return "?";
  }

 private:
  using Clock = std::chrono::steady_clock;

  template <typename T>
  static bool Read(const ZMQPackedMessageSubscriber::FieldInfo& f,
                   const ZMQPackedMessageSubscriber::BufferView& b, const char* dtype, size_t count, T* dst) {
    if (f.dtype != dtype || b.size != count * sizeof(T)) return false;
    std::memcpy(dst, b.data, b.size);
    return true;
  }

  void OnMessage(const ZMQPackedMessageSubscriber::DecodedHeader& header,
                 const std::vector<ZMQPackedMessageSubscriber::BufferView>& buffers) {
    if (header.NeedsByteSwap() || buffers.size() != header.fields.size()) {
      Warn("unexpected byte order or field count");
      return;
    }
    terrain_scan::TerrainMessage m;
    float res = 0.f, origin[2] = {0.f, 0.f};
    double timestamp = 0.0, map_stamp = 0.0;  // no NaN sentinels: built with -ffast-math
    bool got_timestamp = false, got_map_stamp = false;
    bool got_grid = false, got_res = false, got_origin = false, got_pos = false, got_quat = false;
    for (size_t i = 0; i < header.fields.size(); ++i) {
      const auto& f = header.fields[i];
      const auto& b = buffers[i];
      if (f.name == "height_grid" && f.shape.size() == 2) {
        m.rows = static_cast<int>(f.shape[0]);
        m.cols = static_cast<int>(f.shape[1]);
        m.height_grid.resize(f.shape[0] * f.shape[1]);
        got_grid = Read(f, b, "f32", m.height_grid.size(), m.height_grid.data());
      } else if (f.name == "height_grid_valid" && f.shape.size() == 2) {
        m.grid_valid.resize(f.shape[0] * f.shape[1]);
        if (!Read(f, b, "bool", m.grid_valid.size(), m.grid_valid.data())) m.grid_valid.clear();
      } else if (f.name == "grid_resolution") {
        got_res = Read(f, b, "f32", 1, &res);
      } else if (f.name == "grid_origin") {
        got_origin = Read(f, b, "f32", 2, origin);
      } else if (f.name == "torso_pos") {
        got_pos = Read(f, b, "f64", 3, m.torso_pos.data());
      } else if (f.name == "torso_quat") {
        got_quat = Read(f, b, "f64", 4, m.torso_quat_xyzw.data());
      } else if (f.name == "timestamp") {
        got_timestamp = Read(f, b, "f64", 1, &timestamp);
      } else if (f.name == "map_stamp") {
        got_map_stamp = Read(f, b, "f64", 1, &map_stamp);
      } else if (f.name == "torso_grid_21" && f.shape.size() == 2 && f.shape[0] * f.shape[1] == terrain_scan::kTorsoGridCells) {
        m.torso_grid.resize(terrain_scan::kTorsoGridCells);
        if (!Read(f, b, "f32", m.torso_grid.size(), m.torso_grid.data())) m.torso_grid.clear();
      } else if (f.name == "torso_grid_21_valid" && f.shape.size() == 2 && f.shape[0] * f.shape[1] == terrain_scan::kTorsoGridCells) {
        m.torso_grid_valid.resize(terrain_scan::kTorsoGridCells);
        if (!Read(f, b, "bool", m.torso_grid_valid.size(), m.torso_grid_valid.data())) m.torso_grid_valid.clear();
      }
    }
    if (!(got_grid && got_res && got_origin && got_pos && got_quat) ||
        (!m.grid_valid.empty() && m.grid_valid.size() != m.height_grid.size())) {
      Warn("missing or malformed fields");
      return;
    }
    m.resolution = res;
    m.origin_x = origin[0];
    m.origin_y = origin[1];
    std::lock_guard<std::mutex> lock(mutex_);
    msg_ = std::move(m);
    // Unknown map stamp -> treat the map as fresh; the message age still applies.
    map_age_s_ = (got_timestamp && got_map_stamp) ? timestamp - map_stamp : 0.0;
    recv_time_ = Clock::now();
    have_msg_ = true;
  }

  void Warn(const char* what) {
    if (++warnings_ <= 5) std::cerr << "[TerrainInput] dropped a message: " << what << std::endl;
  }

  TerrainInputOptions options_;
  ZMQPackedMessageSubscriber subscriber_;
  mutable std::mutex mutex_;
  terrain_scan::TerrainMessage msg_;
  double map_age_s_ = 0.0;
  Clock::time_point recv_time_{};
  bool have_msg_ = false;
  int warnings_ = 0;
};

#endif  // TERRAIN_INPUT_HPP
