// Host-side check of TareFeatureTracker against a ray-cast synthetic scene with
// known geometry: a textured table at y = 0 and a raised 12 cm block whose top is
// at y = 5 cm, viewed by an orbiting camera using ARKit conventions.
// Build and run with native-tests/run_tracker_test.sh.
#import <opencv2/core.hpp>
#import <opencv2/imgproc.hpp>
#import "../modules/tare-capture/ios/TareFeatureTracker.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int kWidth = 1920, kHeight = 1440;
constexpr float kFocal = 1450, kBlockHalf = 0.06f, kBlockTop = 0.05f;

// 1 m x 1 m texture over the table, blurred noise at several scales so LK has corners.
cv::Mat MakeTexture() {
  std::mt19937 rng(7);
  cv::Mat texture(2048, 2048, CV_32F, cv::Scalar(0));
  for (int scale : {8, 32, 128}) {
    cv::Mat noise(2048 / scale, 2048 / scale, CV_32F);
    cv::randu(noise, 0, 1);
    cv::resize(noise, noise, texture.size(), 0, 0, cv::INTER_CUBIC);
    texture += noise;
  }
  cv::normalize(texture, texture, 20, 235, cv::NORM_MINMAX);
  cv::GaussianBlur(texture, texture, cv::Size(0, 0), 1.5);
  return texture;
}

float Sample(const cv::Mat &texture, float x, float z) {
  const float u = (x + 0.5f) * texture.cols, v = (z + 0.5f) * texture.rows;
  if (u < 0 || v < 0 || u >= texture.cols - 1 || v >= texture.rows - 1) return 128;
  const int iu = (int)u, iv = (int)v;
  const float fu = u - iu, fv = v - iv;
  return (1 - fu) * (1 - fv) * texture.at<float>(iv, iu) + fu * (1 - fv) * texture.at<float>(iv, iu + 1)
       + (1 - fu) * fv * texture.at<float>(iv + 1, iu) + fu * fv * texture.at<float>(iv + 1, iu + 1);
}

// ARKit camera-to-world: columns are +X right, +Y up, +Z backward; the camera looks along -Z.
simd_float4x4 LookAt(simd_float3 eye, simd_float3 target) {
  const simd_float3 back = simd_normalize(eye - target);
  const simd_float3 right = simd_normalize(simd_cross(simd_make_float3(0, 1, 0), back));
  const simd_float3 up = simd_cross(back, right);
  return simd_matrix(simd_make_float4(right, 0), simd_make_float4(up, 0), simd_make_float4(back, 0),
                     simd_make_float4(eye, 1));
}

float SurfaceHeight(float x, float z) {
  return std::abs(x) <= kBlockHalf && std::abs(z) <= kBlockHalf ? kBlockTop : 0;
}

void Render(const cv::Mat &texture, simd_float4x4 pose, CVPixelBufferRef buffer) {
  CVPixelBufferLockBaseAddress(buffer, 0);
  auto *luma = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(buffer, 0);
  const size_t stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0);
  const simd_float3 eye = pose.columns[3].xyz;
  for (int v = 0; v < kHeight; v++) {
    for (int u = 0; u < kWidth; u++) {
      // Pixel ray in ARKit camera axes, then world.
      const simd_float3 local = simd_make_float3((u - kWidth / 2.0f) / kFocal, -(v - kHeight / 2.0f) / kFocal, -1);
      const simd_float3 dir = simd_mul(pose, simd_make_float4(local, 0)).xyz;
      float value = 128;
      if (dir.y < 0) {
        // Block top first, then the table; block sides are not modeled.
        const float t = (kBlockTop - eye.y) / dir.y;
        simd_float3 hit = eye + t * dir;
        if (!(std::abs(hit.x) <= kBlockHalf && std::abs(hit.z) <= kBlockHalf)) hit = eye + (-eye.y / dir.y) * dir;
        value = Sample(texture, hit.x, hit.z);
      }
      luma[v * stride + u] = (uint8_t)std::clamp(value, 0.0f, 255.0f);
    }
  }
  CVPixelBufferUnlockBaseAddress(buffer, 0);
}

}  // namespace

int main() {
  @autoreleasepool {
    const cv::Mat texture = MakeTexture();
    CVPixelBufferRef buffer = nullptr;
    CVPixelBufferCreate(nullptr, kWidth, kHeight, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, nullptr, &buffer);
    const simd_float3x3 intrinsics = simd_matrix(simd_make_float3(kFocal, 0, 0), simd_make_float3(0, kFocal, 0),
                                                 simd_make_float3(kWidth / 2.0f, kHeight / 2.0f, 1));
    TareFeatureTracker *tracker = [[TareFeatureTracker alloc] initWithDownscale:2];
    std::vector<simd_float3> points;
    // 4 s at 15 Hz: a 50-degree arc 40 cm from the plate, 35 cm up, like a slow hand-held orbit.
    const int frames = 60;
    for (int i = 0; i < frames; i++) {
      const float angle = -0.44f + 0.87f * i / (frames - 1);
      const simd_float3 eye = simd_make_float3(0.4f * std::sin(angle), 0.35f, 0.4f * std::cos(angle));
      const simd_float4x4 pose = LookAt(eye, simd_make_float3(0, 0, 0));
      Render(texture, pose, buffer);
      NSData *packed = [tracker processFrame:buffer intrinsics:intrinsics cameraToWorld:pose];
      const float *xyz = (const float *)packed.bytes;
      for (NSUInteger k = 0; k + 2 < packed.length / sizeof(float); k += 3)
        points.push_back(simd_make_float3(xyz[k], xyz[k + 1], xyz[k + 2]));
    }
    CVPixelBufferRelease(buffer);

    std::vector<float> errors;
    for (const auto &p : points) {
      // Points near the block outline may legitimately belong to either surface.
      const bool nearEdge = std::abs(std::max(std::abs(p.x), std::abs(p.z)) - kBlockHalf) < 0.01f;
      if (!nearEdge) errors.push_back(std::abs(p.y - SurfaceHeight(p.x, p.z)));
    }
    std::sort(errors.begin(), errors.end());
    auto pct = [&](double q) { return errors.empty() ? NAN : errors[std::min(errors.size() - 1, (size_t)(q * errors.size()))]; };
    size_t onBlock = std::count_if(points.begin(), points.end(), [](const simd_float3 &p) {
      return std::abs(p.x) < kBlockHalf - 0.01f && std::abs(p.z) < kBlockHalf - 0.01f;
    });
    std::printf("points=%zu (on block top: %zu)  height error mm: median=%.2f p90=%.2f p99=%.2f\n",
                points.size(), onBlock, pct(0.5) * 1000, pct(0.9) * 1000, pct(0.99) * 1000);
    // Gates sized for a prototype: hundreds of points, millimetre-level typical error.
    const bool ok = points.size() >= 300 && onBlock >= 20 && pct(0.5) < 0.002 && pct(0.9) < 0.005;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  }
}
