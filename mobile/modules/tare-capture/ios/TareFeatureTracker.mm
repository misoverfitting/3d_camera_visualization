// OpenCV must be imported before Foundation: its headers use identifiers that
// Objective-C defines as macros (YES/NO).
#ifdef __cplusplus
#import <opencv2/core.hpp>
#import <opencv2/imgproc.hpp>
#import <opencv2/video/tracking.hpp>
#endif

#import "TareFeatureTracker.h"

#include <cmath>
#include <vector>

namespace {

// Downscaled-pixel thresholds assume the default downscale of 2 (960 x 720 from 1920 x 1440).
constexpr int kMaxTracks = 400;
constexpr double kMinCornerDistance = 10;
constexpr double kForwardBackwardPixels = 0.75;
constexpr double kBorderPixels = 8;
constexpr size_t kMaxObservations = 40;
// Depth error grows as depth^2 / (focal * baseline); require a wide angle before
// accepting a point instead of emitting noisy short-baseline triangulations.
constexpr double kMinParallaxDegrees = 6;
constexpr double kMinBaselineMeters = 0.02;
constexpr double kMaxReprojectionPixels = 1.0;
constexpr double kMinDepthMeters = 0.08;
constexpr double kMaxDepthMeters = 2.0;

struct Observation {
  cv::Point2f pixel;
  cv::Matx34d worldToCamera;  // OpenCV camera axes: +X right, +Y down, +Z forward.
  cv::Matx34d projection;
  cv::Vec3d center;
};

struct Track {
  std::vector<Observation> observations;
};

// Linear multi-view (DLT) triangulation followed by cheirality, reprojection and
// parallax gates. ARKit poses are held fixed; nothing here refines them.
bool Triangulate(const Track &track, cv::Vec3d &world) {
  const auto &first = track.observations.front();
  const auto &last = track.observations.back();
  if (cv::norm(last.center - first.center) < kMinBaselineMeters) return false;

  cv::Mat a((int)track.observations.size() * 2, 4, CV_64F);
  for (size_t k = 0; k < track.observations.size(); k++) {
    const auto &o = track.observations[k];
    for (int axis = 0; axis < 2; axis++) {
      double *row = a.ptr<double>((int)(2 * k + axis));
      const double coordinate = axis == 0 ? o.pixel.x : o.pixel.y;
      double length = 0;
      for (int c = 0; c < 4; c++) {
        row[c] = coordinate * o.projection(2, c) - o.projection(axis, c);
        length += row[c] * row[c];
      }
      length = std::sqrt(length);
      if (length > 0) for (int c = 0; c < 4; c++) row[c] /= length;
    }
  }
  cv::Mat solution;
  cv::SVD::solveZ(a, solution);
  const double w = solution.at<double>(3);
  if (std::abs(w) < 1e-12) return false;
  world = cv::Vec3d(solution.at<double>(0) / w, solution.at<double>(1) / w, solution.at<double>(2) / w);
  const cv::Vec4d homogeneous(world[0], world[1], world[2], 1);

  for (const auto &o : track.observations) {
    const cv::Vec3d camera = o.worldToCamera * homogeneous;
    if (camera[2] < kMinDepthMeters || camera[2] > kMaxDepthMeters) return false;
    const cv::Vec3d projected = o.projection * homogeneous;
    const double du = projected[0] / projected[2] - o.pixel.x;
    const double dv = projected[1] / projected[2] - o.pixel.y;
    if (std::hypot(du, dv) > kMaxReprojectionPixels) return false;
  }
  const cv::Vec3d fromFirst = world - first.center, fromLast = world - last.center;
  const double cosine = fromFirst.dot(fromLast) / (cv::norm(fromFirst) * cv::norm(fromLast));
  return std::acos(std::min(1.0, std::max(-1.0, cosine))) * 180 / CV_PI >= kMinParallaxDegrees;
}

}  // namespace

@implementation TareFeatureTracker {
  int _downscale;
  cv::Mat _previous;
  std::vector<Track> _tracks;
  std::vector<cv::Point2f> _points;  // Latest pixel of each track, parallel to _tracks.
}

- (instancetype)initWithDownscale:(int)downscale {
  if ((self = [super init])) _downscale = std::max(1, downscale);
  return self;
}

- (void)reset {
  _previous.release();
  _tracks.clear();
  _points.clear();
  _activeTracks = 0;
}

- (NSData *)processFrame:(CVPixelBufferRef)pixelBuffer
              intrinsics:(simd_float3x3)intrinsics
           cameraToWorld:(simd_float4x4)cameraToWorld {
  NSMutableData *output = [NSMutableData data];
  if (CVPixelBufferGetPlaneCount(pixelBuffer) < 1) return output;

  cv::Mat gray;
  CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  const int width = (int)CVPixelBufferGetWidthOfPlane(pixelBuffer, 0);
  const int height = (int)CVPixelBufferGetHeightOfPlane(pixelBuffer, 0);
  cv::Mat luma(height, width, CV_8UC1, CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
               CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0));
  cv::resize(luma, gray, cv::Size(width / _downscale, height / _downscale), 0, 0, cv::INTER_AREA);
  CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

  // Scale full-resolution intrinsics to downscaled pixels (pixel-center convention).
  const double s = _downscale;
  const cv::Matx33d k(intrinsics.columns[0][0] / s, 0, (intrinsics.columns[2][0] + 0.5) / s - 0.5,
                      0, intrinsics.columns[1][1] / s, (intrinsics.columns[2][1] + 0.5) / s - 0.5,
                      0, 0, 1);
  // ARKit cameras look along -Z with +Y up; flip Y and Z to reach OpenCV's convention.
  const simd_float4x4 inverse = simd_inverse(cameraToWorld);
  cv::Matx34d worldToCamera;
  for (int r = 0; r < 3; r++) {
    const double sign = r == 0 ? 1 : -1;
    for (int c = 0; c < 4; c++) worldToCamera(r, c) = sign * inverse.columns[c][r];
  }
  const Observation current{{}, worldToCamera, k * worldToCamera,
    cv::Vec3d(cameraToWorld.columns[3][0], cameraToWorld.columns[3][1], cameraToWorld.columns[3][2])};

  const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  if (_previous.size() != gray.size()) {
    _tracks.clear();
    _points.clear();
  } else if (!_points.empty()) {
    std::vector<cv::Point2f> next, back;
    std::vector<uchar> status, backStatus;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(_previous, gray, _points, next, status, error, cv::Size(21, 21), 3, criteria);
    cv::calcOpticalFlowPyrLK(gray, _previous, next, back, backStatus, error, cv::Size(21, 21), 3, criteria);
    std::vector<Track> keptTracks;
    std::vector<cv::Point2f> keptPoints;
    for (size_t i = 0; i < _points.size(); i++) {
      const cv::Point2f &p = next[i];
      if (!status[i] || !backStatus[i] || cv::norm(back[i] - _points[i]) > kForwardBackwardPixels
          || p.x < kBorderPixels || p.y < kBorderPixels
          || p.x > gray.cols - kBorderPixels || p.y > gray.rows - kBorderPixels) continue;
      Track &track = _tracks[i];
      Observation observation = current;
      observation.pixel = p;
      track.observations.push_back(observation);
      // Keep the first observation for maximum baseline; thin the middle.
      if (track.observations.size() > kMaxObservations) track.observations.erase(track.observations.begin() + 1);
      cv::Vec3d world;
      if (track.observations.size() >= 3 && Triangulate(track, world)) {
        const float xyz[3] = {(float)world[0], (float)world[1], (float)world[2]};
        [output appendBytes:xyz length:sizeof(xyz)];
        _triangulatedTotal++;
        continue;  // Retire the track so its slot can follow a new corner.
      }
      keptTracks.push_back(std::move(track));
      keptPoints.push_back(p);
    }
    _tracks.swap(keptTracks);
    _points.swap(keptPoints);
  }

  if (_points.size() < kMaxTracks * 8 / 10) {
    cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(255));
    for (const auto &p : _points) cv::circle(mask, p, (int)kMinCornerDistance, cv::Scalar(0), -1);
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, kMaxTracks - (int)_points.size(), 0.01, kMinCornerDistance, mask, 5);
    if (!corners.empty()) cv::cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1), criteria);
    for (const auto &corner : corners) {
      Observation observation = current;
      observation.pixel = corner;
      _tracks.push_back(Track{{observation}});
      _points.push_back(corner);
    }
  }
  _previous = gray;
  _activeTracks = (NSInteger)_points.size();
  return output;
}

@end
