import Foundation
import simd

/// Single-view rim fallback (tare_impl.md §6.4/§6.6): projects rim pixels (learned
/// keypoints or a container-mask contour) onto the plane at a given rim height and
/// fits a circle with RANSAC. The height must come from elsewhere (e.g. the
/// learned RimHeightRegressor). Without one, the table plane is used, which
/// overstates the diameter by about D / (D − h) for camera height D (§3).
public struct ContourRimFitter {
  public var config: PipelineConfig.Rim

  public init(config: PipelineConfig.Rim = .init()) { self.config = config }

  public func fit(rimPx: [SIMD2<Double>], keyframe: Keyframe, tablePlane: Plane,
                  rimHeightM: Double?, seed: UInt64 = 1) -> Result<RimEstimate, RimFailure> {
    guard rimPx.count >= config.minKeypointsPerView else { return .failure(.insufficientKeypoints) }
    let heightM = rimHeightM ?? 0
    let rimPlane = tablePlane.offset(byM: heightM)
    let points = rimPx.compactMap { rimPlane.intersection(keyframe.rayWorld(throughPx: $0)).map(tablePlane.coordinates2D) }
    var rng = SplitMix64(seed: seed)
    guard let (circle, inliers) = CircleFit.ransac(points, inlierM: config.ransacInlierM,
                                                    iterations: config.ransacIterations, using: &rng),
          inliers.count >= config.minKeypointsPerView
    else { return .failure(.noConvergence) }

    let squared = inliers.compactMap { i -> Double? in
      let offset = points[i] - circle.center
      let rimPoint = tablePlane.pointWorld(at: circle.center + circle.radiusM * offset / simd_length(offset), heightM: heightM)
      guard let projected = keyframe.projectPx(rimPoint) else { return nil }
      return simd_length_squared(projected - rimPx[i])
    }
    let rmsPx = sqrt(squared.reduce(0, +) / Double(max(squared.count, 1)))
    let inlierFraction = Double(inliers.count) / Double(rimPx.count)
    // An assumed height leaves the scale biased; halve confidence so callers prefer measured paths.
    let confidence = pow(0.5, rmsPx / config.confidenceHalfResidualPx) * inlierFraction * (rimHeightM == nil ? 0.5 : 1)
    return .success(RimEstimate(
      centerWorld: tablePlane.pointWorld(at: circle.center, heightM: heightM), radiusM: circle.radiusM,
      heightM: heightM, radiusUncertaintyM: nil, heightUncertaintyM: nil, rmsResidualPx: rmsPx,
      method: .singleView, viewCount: 1, confidence: confidence))
  }
}
