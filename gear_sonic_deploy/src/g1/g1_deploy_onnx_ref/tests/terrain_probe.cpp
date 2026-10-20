// terrain_probe: bring-up check of the height-map observations, no motors involved.
//
// Reads the onboard `terrain` topic (perception/terrain_publisher.py) through the deploy runner's own
// TerrainInput / terrain_scan.hpp code and prints, twice a second, what a policy would observe:
// the input status, the 11x11 pelvis scan (height_map_flat / height_map_valid_flat) and the 21x21
// torso grid (torso_heightmap_21 / torso_heightmap_21_valid); at the end, one ASCII map of each.
// Waist angles are taken as 0 (the probe does not read joint states), so the pelvis sits 5.4 cm
// below the torso. On a gantry, start perception with GANTRY_FILTER=true.
//
//   ./target/release/terrain_probe [seconds=20] [port=5559] [host=127.0.0.1]
//
// From another machine (e.g. the workstation running the policy over Wi-Fi), pass the robot's
// address as host; the robot's perception stack must then publish with TERRAIN_BIND=tcp://0.0.0.0:5559.
//
// Built with -DBUILD_TESTS=ON (needs libzmq). First gantry run (2026-10-08): floor 0.82 m below the
// torso / 0.755 m below the pelvis, 35/121 scan rays and ~200/441 grid cells seen standing still
// (only the area ahead is mapped), a chair seat ~1 m front-left at the right height.
#include "input_interface/terrain_input.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

int main(int argc, char** argv) {
  const double seconds = argc > 1 ? std::atof(argv[1]) : 20.0;
  if (argc > 2) GlobalTerrainInputOptions().port = std::atoi(argv[2]);
  std::FILE* dump = argc > 4 ? std::fopen(argv[4], "w") : nullptr;
  if (argc > 4 && !dump) {
    std::fprintf(stderr, "cannot write %s\n", argv[4]);
    return 1;
  }
  if (argc > 3) GlobalTerrainInputOptions().host = argv[3];
  TerrainInput in(GlobalTerrainInputOptions());
  in.Start();
  terrain_scan::TerrainMessage last;
  terrain_scan::Scan last_scan;
  terrain_scan::TorsoGrid last_grid;
  bool have = false;
  const auto t0 = std::chrono::steady_clock::now();
  auto last_print = t0 - std::chrono::seconds(1);
  terrain_scan::Vec3 last_pos{1e9, 1e9, 1e9};
  size_t n_dumped = 0;
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(dump ? 2 : 500));
    if (dump) {
      terrain_scan::TerrainMessage d;
      const auto st = in.Latest(d);
      if (st == TerrainInput::Status::kOk && d.torso_pos != last_pos) {
        last_pos = d.torso_pos;
        const double now = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const auto pg = terrain_scan::PelvisGridFromTerrainMessage(d, 0.0, 0.0, 0.0);
        const auto sc = terrain_scan::ScanFromTerrainMessage(d, 0.0, 0.0, 0.0);
        std::fprintf(dump, "%.6f,%s,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f", now, TerrainInput::StatusName(st),
                     d.torso_pos[0], d.torso_pos[1], d.torso_pos[2], d.torso_quat_xyzw[0], d.torso_quat_xyzw[1],
                     d.torso_quat_xyzw[2], d.torso_quat_xyzw[3]);
        for (double v : pg.heights) std::fprintf(dump, ",%.6f", v);
        for (double v : pg.valid) std::fprintf(dump, ",%.0f", v);
        for (size_t r = 0; r < sc.valid.size(); ++r) std::fprintf(dump, ",%.6f", sc.points[3 * r + 2]);
        for (double v : sc.valid) std::fprintf(dump, ",%.0f", v);
        std::fprintf(dump, "\n");
        ++n_dumped;
      }
      if (std::chrono::steady_clock::now() - last_print < std::chrono::milliseconds(500)) continue;
      last_print = std::chrono::steady_clock::now();
    }
    terrain_scan::TerrainMessage m;
    const auto st = in.Latest(m);
    if (st != TerrainInput::Status::kOk) {
      std::printf("status: %s\n", TerrainInput::StatusName(st));
      continue;
    }
    const auto scan = terrain_scan::ScanFromTerrainMessage(m, 0.0, 0.0, 0.0);
    const auto grid = terrain_scan::TorsoGridFromTerrainMessage(m);
    std::vector<double> z, g;
    for (size_t r = 0; r < scan.valid.size(); ++r) if (scan.valid[r] > 0.5) z.push_back(scan.points[3 * r + 2]);
    for (size_t i = 0; i < grid.valid.size(); ++i) if (grid.valid[i] > 0.5) g.push_back(grid.heights[i]);
    std::printf("ok: torso z %.3f | scan %3zu/121 valid, median hit z %.3f m below pelvis | torso grid %3zu/441 valid, median %.3f m\n",
                m.torso_pos[2], z.size(), -median(z), g.size(), median(g));
    last = m; last_scan = scan; last_grid = grid; have = true;
  }
  if (have) {
    std::printf("\npelvis scan, hit z relative to pelvis (cm), rows = +x (forward) at top, '  .' = invalid:\n");
    for (int i = 10; i >= 0; --i) {
      for (int j = 10; j >= 0; --j) {  // +y to the left
        const size_t r = static_cast<size_t>(i * 11 + j);
        if (last_scan.valid[r] > 0.5) std::printf("%4.0f", 100.0 * last_scan.points[3 * r + 2]); else std::printf("   .");
      }
      std::printf("\n");
    }
    std::printf("\ntorso grid, torso z - surface z (dm), forward at top, '.' = unseen (filled):\n");
    for (int x = 20; x >= 0; --x) {
      for (int y = 20; y >= 0; --y) {
        const size_t i = static_cast<size_t>(y * 21 + x);  // [y][x]
        if (last_grid.valid[i] > 0.5) std::printf("%3.0f", 10.0 * last_grid.heights[i]); else std::printf("  .");
      }
      std::printf("\n");
    }
  }
  if (dump) {
    std::fclose(dump);
    std::printf("\n%zu messages written to %s\n", n_dumped, argv[4]);
  }
  in.Stop();
  return 0;
}
