// Checks include/terrain_scan.hpp against golden outputs of the Python reference
// (gear_sonic/utils/terrain_scan.py, scan_from_terrain_msg). Regenerate the data with
// gear_sonic/tests/make_terrain_scan_golden.py.
//
// Data layout (unit_tests/golden/terrain_scan_golden.txt), whitespace separated:
//   cases rows cols
//   per case: origin_x origin_y resolution
//             torso_pos(3) torso_quat_xyzw(4) waist_yaw waist_roll waist_pitch
//             height_grid (rows*cols) / grid_valid (rows*cols)
//             expected height_map_flat (n*n*3) / expected height_map_valid_flat (n*n)

#include <gtest/gtest.h>

#include "../include/terrain_scan.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>

namespace {

// Next to this source file (CMake compiles it by absolute path), or $TERRAIN_SCAN_GOLDEN_DIR.
std::string GoldenPath() {
  if (const char* dir = std::getenv("TERRAIN_SCAN_GOLDEN_DIR")) return std::string(dir) + "/terrain_scan_golden.txt";
  std::string here = __FILE__;
  here = here.substr(0, here.find_last_of('/') + 1);
  return here + "golden/terrain_scan_golden.txt";
}

}  // namespace

TEST(TerrainScan, MatchesPythonReference) {
  std::ifstream in(GoldenPath());
  ASSERT_TRUE(in.good()) << "missing " << GoldenPath();
  int cases = 0, rows = 0, cols = 0;
  in >> cases >> rows >> cols;
  ASSERT_GT(cases, 0);
  const int n = terrain_scan::NumRaysPerSide();
  int rays_differing = 0, rays_total = 0, valid_total = 0;
  double worst = 0.0;
  for (int c = 0; c < cases; ++c) {
    terrain_scan::TerrainMessage msg;
    msg.rows = rows;
    msg.cols = cols;
    float ox, oy, res;
    in >> ox >> oy >> res;
    msg.origin_x = ox;  // float32 on the wire, like the Python reference
    msg.origin_y = oy;
    msg.resolution = res;
    double waist[3];
    in >> msg.torso_pos[0] >> msg.torso_pos[1] >> msg.torso_pos[2];
    for (auto& q : msg.torso_quat_xyzw) in >> q;
    in >> waist[0] >> waist[1] >> waist[2];
    msg.height_grid.resize(static_cast<size_t>(rows * cols));
    msg.grid_valid.resize(static_cast<size_t>(rows * cols));
    for (auto& h : msg.height_grid) in >> h;
    for (auto& v : msg.grid_valid) {
      int b;
      in >> b;
      v = static_cast<uint8_t>(b);
    }
    std::vector<double> want_points(static_cast<size_t>(n * n * 3)), want_valid(static_cast<size_t>(n * n));
    for (auto& p : want_points) in >> p;
    for (auto& v : want_valid) in >> v;
    ASSERT_TRUE(in.good()) << "truncated golden data at case " << c;

    const auto got = terrain_scan::ScanFromTerrainMessage(msg, waist[0], waist[1], waist[2]);
    ASSERT_EQ(got.points.size(), want_points.size());
    for (int r = 0; r < n * n; ++r) {
      double err = 0.0;
      for (int k = 0; k < 3; ++k) err = std::max(err, std::abs(got.points[3 * r + k] - want_points[3 * r + k]));
      const bool differs = err > 1e-9 || got.valid[r] != want_valid[r];
      rays_differing += differs ? 1 : 0;
      worst = std::max(worst, err);
      valid_total += want_valid[r] > 0.5 ? 1 : 0;
      ++rays_total;
    }
  }
  std::cout << "terrain_scan: " << rays_total << " rays (" << valid_total << " valid), " << rays_differing
            << " differ from the Python reference, worst " << worst << " m" << std::endl;
  // Identical arithmetic: on x86 every ray matches. Fused multiply-adds (aarch64 default)
  // can move a ray that lands exactly on a cell boundary to the neighbouring cell.
  EXPECT_LE(rays_differing, rays_total / 500);
  EXPECT_GT(valid_total, rays_total / 4);  // the cases exercise valid and invalid rays
  EXPECT_LT(valid_total, rays_total);
}

TEST(TerrainScan, FlatFloorIsAnalytic) {
  terrain_scan::TerrainMessage msg;
  msg.rows = msg.cols = 50;
  msg.origin_x = msg.origin_y = -0.98;
  msg.resolution = 0.04;
  msg.height_grid.assign(2500, -0.95f);  // floor 0.95 m under the torso
  msg.torso_pos = {1.0, 2.0, 0.95};
  msg.torso_quat_xyzw = {0.0, 0.0, std::sin(0.2), std::cos(0.2)};
  const auto s = terrain_scan::ScanFromTerrainMessage(msg, 0.0, 0.0, 0.0);
  const double pelvis_h = 0.95 - 0.054;  // waist chain: 0.035 + 0.019
  for (int i = 0; i < 11; ++i)
    for (int j = 0; j < 11; ++j) {
      const int r = i * 11 + j;
      EXPECT_EQ(s.valid[r], 1.0);
      EXPECT_NEAR(s.points[3 * r + 0], (-0.75 + 0.15 * i) * pelvis_h, 1e-3);
      EXPECT_NEAR(s.points[3 * r + 1], (-0.75 + 0.15 * j) * pelvis_h, 1e-3);
      EXPECT_NEAR(s.points[3 * r + 2], -pelvis_h, 1e-3);
    }
}

TEST(TerrainScan, NoMapMeansEveryRayInvalid) {
  terrain_scan::TerrainMessage msg;
  msg.rows = msg.cols = 50;
  msg.origin_x = msg.origin_y = -0.98;
  msg.resolution = 0.04;
  msg.height_grid.assign(2500, -0.9f);
  msg.grid_valid.assign(2500, 0);  // nothing observed
  const auto s = terrain_scan::ScanFromTerrainMessage(msg, 0.0, 0.0, 0.0);
  for (double v : s.valid) EXPECT_EQ(v, 0.0);
  for (double p : s.points) EXPECT_EQ(p, 0.0);
  const auto e = terrain_scan::EmptyScan();
  EXPECT_EQ(e.points.size(), 363u);
  EXPECT_EQ(e.valid.size(), 121u);
}
