/**
 * @file terrain_scan.hpp
 * @brief The policy's pelvis height-map observation, computed from the onboard terrain map.
 *
 * Port of gear_sonic/utils/terrain_scan.py (scan_from_terrain_msg and what it calls); the
 * unit test checks it against golden outputs of that Python reference.
 *
 * The observation is a fan of n x n rays (n = int(size / resolution) + 1, 11 by default)
 * cast from the pelvis along normalize(x, y, -1), rotated by the pelvis heading, marched
 * against a 2.5D height field. Output per ray: the hit point relative to the pelvis in the
 * pelvis heading frame ([x][y][xyz] order, 363 values), and a validity flag (121 values).
 * A ray is valid when it lands on an observed cell from above, or runs into a taller
 * observed cell from an observed one; invalid rays read zero. See
 * gear_sonic/README_height_map.md.
 *
 * Header-only, C++17, no dependencies beyond the standard library.
 * Quaternions are wxyz unless a name says otherwise.
 */

#ifndef TERRAIN_SCAN_HPP
#define TERRAIN_SCAN_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace terrain_scan {

constexpr double kHeightMapSize = 1.5;
constexpr double kHeightMapResolution = 0.15;
constexpr double kHeightMapMaxDist = 5.0;

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;
using QuatWxyz = std::array<double, 4>;

inline int NumRaysPerSide(double size = kHeightMapSize, double resolution = kHeightMapResolution) {
  return static_cast<int>(size / resolution) + 1;
}

// ---------------------------------------------------------------------------
// Small rotation helpers (row-major 3x3)
// ---------------------------------------------------------------------------

inline Mat3 RotZ(double a) {
  const double c = std::cos(a), s = std::sin(a);
  return {{{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}}};
}
inline Mat3 RotX(double a) {
  const double c = std::cos(a), s = std::sin(a);
  return {{{1.0, 0.0, 0.0}, {0.0, c, -s}, {0.0, s, c}}};
}
inline Mat3 RotY(double a) {
  const double c = std::cos(a), s = std::sin(a);
  return {{{c, 0.0, s}, {0.0, 1.0, 0.0}, {-s, 0.0, c}}};
}

inline Mat3 MatMul(const Mat3& a, const Mat3& b) {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
  return r;
}
inline Mat3 Transpose(const Mat3& a) {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r[i][j] = a[j][i];
  return r;
}
inline Vec3 MatVec(const Mat3& a, const Vec3& v) {
  return {a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
          a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
          a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2]};
}

inline Mat3 QuatToMat(const QuatWxyz& q_in) {
  const double n = std::sqrt(q_in[0] * q_in[0] + q_in[1] * q_in[1] + q_in[2] * q_in[2] + q_in[3] * q_in[3]);
  const double w = q_in[0] / n, x = q_in[1] / n, y = q_in[2] / n, z = q_in[3] / n;
  return {{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
           {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
           {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}}};
}

inline QuatWxyz MatToQuat(const Mat3& m) {
  const double t = m[0][0] + m[1][1] + m[2][2];
  QuatWxyz q{};
  if (t > 0) {
    const double s = 2.0 * std::sqrt(t + 1.0);
    q = {0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s};
  } else {
    int i = 0;
    if (m[1][1] > m[i][i]) i = 1;
    if (m[2][2] > m[i][i]) i = 2;
    const int j = (i + 1) % 3, k = (i + 2) % 3;
    const double s = 2.0 * std::sqrt(1.0 + m[i][i] - m[j][j] - m[k][k]);
    q[0] = (m[k][j] - m[j][k]) / s;
    q[1 + i] = 0.25 * s;
    q[1 + j] = (m[j][i] + m[i][j]) / s;
    q[1 + k] = (m[k][i] + m[i][k]) / s;
  }
  if (q[0] < 0) {
    for (auto& v : q) v = -v;
  }
  return q;
}

/// Yaw of the sim's get_heading_q: the twist about z of a wxyz quaternion.
inline double HeadingYaw(const QuatWxyz& q) { return 2.0 * std::atan2(q[3], q[0]); }

// ---------------------------------------------------------------------------
// Pelvis pose from the torso pose (waist chain of gear_sonic_deploy/g1/g1_29dof.xml)
// ---------------------------------------------------------------------------

struct Pose {
  Vec3 pos{};
  QuatWxyz quat{1.0, 0.0, 0.0, 0.0};
};

inline Pose PelvisPoseFromTorso(const Vec3& torso_pos, const QuatWxyz& torso_quat, double waist_yaw,
                                double waist_roll, double waist_pitch) {
  const Vec3 waist_roll_offset{-0.0039635, 0.0, 0.035};
  const Vec3 torso_offset{0.0, 0.0, 0.019};
  const Mat3 rz = RotZ(waist_yaw), rx = RotX(waist_roll), ry = RotY(waist_pitch);
  const Mat3 r_pt = MatMul(MatMul(rz, rx), ry);
  const Vec3 rx_off = MatVec(rx, torso_offset);
  const Vec3 t_pt = MatVec(rz, {waist_roll_offset[0] + rx_off[0], waist_roll_offset[1] + rx_off[1],
                                waist_roll_offset[2] + rx_off[2]});
  const Mat3 r_wp = MatMul(QuatToMat(torso_quat), Transpose(r_pt));
  const Vec3 off = MatVec(r_wp, t_pt);
  return {{torso_pos[0] - off[0], torso_pos[1] - off[1], torso_pos[2] - off[2]}, MatToQuat(r_wp)};
}

// ---------------------------------------------------------------------------
// Height field and the ray march
// ---------------------------------------------------------------------------

/// Gravity-aligned 2.5D terrain: heights[row * cols + col] = terrain z at the cell centre
/// origin + (col, row) * resolution. NaN = unseen.
struct HeightField {
  std::vector<double> heights;
  int rows = 0, cols = 0;
  double origin_x = 0.0, origin_y = 0.0, resolution = 0.04;

  /// Nearest-cell height (np.rint semantics: round half to even); NaN if unseen or off the grid.
  double Sample(double x, double y) const {
    const long i = static_cast<long>(std::nearbyint((x - origin_x) / resolution));
    const long j = static_cast<long>(std::nearbyint((y - origin_y) / resolution));
    if (i < 0 || i >= cols || j < 0 || j >= rows) return std::numeric_limits<double>::quiet_NaN();
    return heights[static_cast<size_t>(j) * static_cast<size_t>(cols) + static_cast<size_t>(i)];
  }

  /// Rays below this z (two cells under the lowest observed height) can't land from above.
  double FloorStop() const {
    double lo = std::numeric_limits<double>::infinity();
    for (double h : heights)
      if (std::isfinite(h)) lo = std::min(lo, h);
    return std::isfinite(lo) ? lo - 2.0 * resolution : -std::numeric_limits<double>::infinity();
  }
};

struct RayHit {
  Vec3 point{};
  bool hit = false;
  bool valid = false;
};

/// First hit of one ray (see terrain_scan.raycast_heightfield). floor_stop = hf.FloorStop().
inline RayHit RaycastHeightField(const HeightField& hf, const Vec3& start, const Vec3& dir, double max_dist,
                                 double floor_stop, int refine = 8) {
  const double step = 0.5 * hf.resolution;
  const size_t n_t = static_cast<size_t>(std::ceil((max_dist + step) / step));  // np.arange length
  auto t_at = [&](size_t k) { return k + 1 == n_t ? std::min(static_cast<double>(k) * step, max_dist)
                                                  : static_cast<double>(k) * step; };
  auto h_at = [&](double t) { return hf.Sample(start[0] + t * dir[0], start[1] + t * dir[1]); };

  RayHit out;
  size_t k = n_t - 1;
  for (size_t s = 0; s < n_t; ++s) {
    const double t = t_at(s);
    const double z = start[2] + t * dir[2];
    const double h = h_at(t);
    if (z <= h && z >= floor_stop) {  // NaN compares false: unseen cells never stop a ray
      out.hit = true;
      k = s;
      break;
    }
  }
  if (out.hit) {
    const double tk = t_at(k);
    const bool prev_unseen = k > 0 && std::isnan(h_at(t_at(k - 1)));
    const double penetration = h_at(tk) - (start[2] + tk * dir[2]);
    out.valid = !prev_unseen || penetration <= 1.01 * step;
  }

  // Bisection between the last sample above the surface and the first below it.
  double lo = t_at(k > 0 ? k - 1 : 0), hi = t_at(k);
  for (int it = 0; it < refine; ++it) {
    const double mid = 0.5 * (lo + hi);
    const bool inside = (start[2] + mid * dir[2]) <= h_at(mid);
    if (out.hit && inside) hi = mid;
    if (out.hit && !inside) lo = mid;
  }
  const double t_hit = out.hit ? hi : max_dist;
  out.point = {start[0] + t_hit * dir[0], start[1] + t_hit * dir[1], start[2] + t_hit * dir[2]};
  return out;
}

/// Unit ray directions in the root heading frame, [x][y] order, n*n entries.
inline std::vector<Vec3> ScanRayDirs(double size = kHeightMapSize, double resolution = kHeightMapResolution) {
  const int n = NumRaysPerSide(size, resolution);
  std::vector<double> lin(static_cast<size_t>(n));
  const double start = -0.5 * size, stop = 0.5 * size, step = (stop - start) / (n - 1);
  for (int i = 0; i < n; ++i) lin[static_cast<size_t>(i)] = start + i * step;  // np.linspace
  lin[static_cast<size_t>(n - 1)] = stop;
  std::vector<Vec3> dirs;
  dirs.reserve(static_cast<size_t>(n * n));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const double x = lin[static_cast<size_t>(i)], y = lin[static_cast<size_t>(j)];
      const double norm = std::sqrt(x * x + y * y + 1.0);
      dirs.push_back({x / norm, y / norm, -1.0 / norm});
    }
  return dirs;
}

struct Scan {
  std::vector<double> points;  ///< n*n*3, pelvis heading frame, invalid rays zero
  std::vector<double> valid;   ///< n*n, 1.0 / 0.0
};

/// The sim height_map observation against a height field (terrain_scan.terrain_scan), with
/// invalid rays zeroed as the policy sees them. root_pose: pelvis pose in the field's frame.
inline Scan TerrainScan(const HeightField& hf, const Pose& root, double floor_z,
                        double size = kHeightMapSize, double resolution = kHeightMapResolution,
                        double max_dist = kHeightMapMaxDist) {
  const double yaw = HeadingYaw(root.quat);
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double floor_stop = hf.FloorStop();
  const auto dirs_local = ScanRayDirs(size, resolution);
  Scan out;
  out.points.assign(dirs_local.size() * 3, 0.0);
  out.valid.assign(dirs_local.size(), 0.0);
  for (size_t r = 0; r < dirs_local.size(); ++r) {
    const Vec3& d = dirs_local[r];
    const Vec3 dir{c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]};
    RayHit h = RaycastHeightField(hf, root.pos, dir, max_dist, floor_stop);
    if (!h.valid) continue;
    // The sim's floor clamp (a no-op for floor_z = -inf, as on the robot).
    const double sz = root.pos[2] - floor_z;
    const double denom = std::max(root.pos[2] - h.point[2], 1e-8);
    const double scale = std::min(sz / denom, 1.0);
    const double dx = (h.point[0] - root.pos[0]) * scale;
    const double dy = (h.point[1] - root.pos[1]) * scale;
    const double dz = (h.point[2] - root.pos[2]) * scale;
    out.points[3 * r + 0] = c * dx + s * dy;
    out.points[3 * r + 1] = -s * dx + c * dy;
    out.points[3 * r + 2] = dz;
    out.valid[r] = 1.0;
  }
  return out;
}

// ---------------------------------------------------------------------------
// From one `terrain` message (gam perception/terrain_publisher.py)
// ---------------------------------------------------------------------------

/// The fields of a `terrain` message this observation needs.
struct TerrainMessage {
  std::vector<float> height_grid;   ///< rows*cols, terrain_z - torso_z, [y][x]; NaN = unseen
  std::vector<uint8_t> grid_valid;  ///< rows*cols, empty = all valid
  int rows = 0, cols = 0;
  double origin_x = 0.0, origin_y = 0.0, resolution = 0.04;
  Vec3 torso_pos{};
  std::array<double, 4> torso_quat_xyzw{0.0, 0.0, 0.0, 1.0};
};

/// terrain_scan.scan_from_terrain_msg: the policy's height_map_flat / height_map_valid_flat.
inline Scan ScanFromTerrainMessage(const TerrainMessage& msg, double waist_yaw, double waist_roll,
                                   double waist_pitch, double size = kHeightMapSize,
                                   double resolution = kHeightMapResolution,
                                   double max_dist = kHeightMapMaxDist) {
  HeightField hf;
  hf.rows = msg.rows;
  hf.cols = msg.cols;
  hf.origin_x = msg.origin_x;
  hf.origin_y = msg.origin_y;
  hf.resolution = msg.resolution;
  hf.heights.resize(msg.height_grid.size());
  for (size_t i = 0; i < msg.height_grid.size(); ++i) {
    const bool seen = msg.grid_valid.empty() || msg.grid_valid[i] != 0;
    hf.heights[i] = seen ? static_cast<double>(msg.height_grid[i]) : std::numeric_limits<double>::quiet_NaN();
  }
  const auto& qx = msg.torso_quat_xyzw;
  const QuatWxyz torso_q{qx[3], qx[0], qx[1], qx[2]};
  const Pose pelvis = PelvisPoseFromTorso(msg.torso_pos, torso_q, waist_yaw, waist_roll, waist_pitch);
  // Into the grid frame: centred on the torso, rotated by the torso heading, z relative.
  const Mat3 rt_t = Transpose(RotZ(HeadingYaw(torso_q)));
  const Vec3 dp{pelvis.pos[0] - msg.torso_pos[0], pelvis.pos[1] - msg.torso_pos[1],
                pelvis.pos[2] - msg.torso_pos[2]};
  Pose pelvis_g{MatVec(rt_t, dp), MatToQuat(MatMul(rt_t, QuatToMat(pelvis.quat)))};
  return TerrainScan(hf, pelvis_g, -std::numeric_limits<double>::infinity(), size, resolution, max_dist);
}

/// All rays invalid: what the policy gets when there is no usable map.
inline Scan EmptyScan(double size = kHeightMapSize, double resolution = kHeightMapResolution) {
  const auto n = static_cast<size_t>(NumRaysPerSide(size, resolution));
  return {std::vector<double>(n * n * 3, 0.0), std::vector<double>(n * n, 0.0)};
}

}  // namespace terrain_scan

#endif  // TERRAIN_SCAN_HPP
