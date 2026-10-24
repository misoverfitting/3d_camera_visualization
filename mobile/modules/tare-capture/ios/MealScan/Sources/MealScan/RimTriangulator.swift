import Foundation
import simd

public enum RimFailure: Error, Equatable {
  case insufficientViews
  case insufficientKeypoints
  case insufficientBaseline
  case noConvergence
}

/// Two-view rim geometry (tare_impl.md §6.5): the rim is a circle parallel to the
/// table with unknown center (2), radius and height. Height is unobservable from
/// one view (a cone of rays cut by any parallel plane is a circle), so at least two
/// views with real baseline are required; use `ContourRimFitter` otherwise.
public struct RimTriangulator {
  public var config: PipelineConfig.Rim

  public init(config: PipelineConfig.Rim = .init()) { self.config = config }

  public func triangulate(_ observations: [RimObservation], tablePlane: Plane) -> Result<RimEstimate, RimFailure> {
    let views = observations.filter { $0.weights.filter { $0 > 0 }.count >= config.minKeypointsPerView }
    guard observations.count >= 2 else { return .failure(.insufficientViews) }
    guard views.count >= 2 else { return .failure(.insufficientKeypoints) }
    let centers = views.map(\.keyframe.cameraCenterWorld)
    let baselineM = centers.flatMap { a in centers.map { simd_distance(a, $0) } }.max() ?? 0
    guard baselineM >= 0.01 else { return .failure(.insufficientBaseline) }

    // Pixel residuals of every weighted keypoint against a candidate rim circle.
    let samples = views.flatMap { view in
      zip(view.keypointsPx, view.weights).filter { $0.1 > 0 }.map {
        (keyframe: view.keyframe, pixel: $0.0, sqrtWeight: sqrt($0.1), ray: view.keyframe.rayWorld(throughPx: $0.0))
      }
    }
    func residuals(_ p: [Double]) -> [Double]? {
      let center = SIMD2(p[0], p[1]), radiusM = p[2], heightM = p[3]
      guard radiusM > 0 else { return nil }
      let rimPlane = tablePlane.offset(byM: heightM)
      var out: [Double] = []
      out.reserveCapacity(samples.count * 2)
      for s in samples {
        guard let hit = rimPlane.intersection(s.ray) else { return nil }
        let offset = tablePlane.coordinates2D(hit) - center
        let length = simd_length(offset)
        guard length > 1e-9 else { return nil }
        // Nearest rim point to the ray's hit, reprojected into the same view.
        let rimPoint = tablePlane.pointWorld(at: center + radiusM * offset / length, heightM: heightM)
        guard let projected = s.keyframe.projectPx(rimPoint) else { return nil }
        out.append((projected.x - s.pixel.x) * s.sqrtWeight)
        out.append((projected.y - s.pixel.y) * s.sqrtWeight)
      }
      return out
    }

    // Initialize by scanning heights: at the true height all views' rays land on one circle.
    var best: (params: [Double], cost: Double)?
    for heightM in stride(from: 0.0, through: config.maxRimHeightM, by: config.heightSearchStepM) {
      let rimPlane = tablePlane.offset(byM: heightM)
      let points = samples.compactMap { rimPlane.intersection($0.ray).map(tablePlane.coordinates2D) }
      guard points.count == samples.count, let circle = CircleFit.algebraic(points) else { continue }
      let params = [circle.center.x, circle.center.y, circle.radiusM, heightM]
      guard let r = residuals(params) else { continue }
      let cost = r.reduce(0) { $0 + $1 * $1 }
      if cost < best?.cost ?? .infinity { best = (params, cost) }
    }
    guard let initial = best?.params,
          let fit = Numerics.levenbergMarquardt(initial: initial, steps: [1e-7, 1e-7, 1e-7, 1e-7],
                                                maxIterations: config.maxIterations, residuals: residuals)
    else { return .failure(.noConvergence) }

    let p = fit.params
    let ssr = fit.residuals.reduce(0) { $0 + $1 * $1 }
    let rmsPx = sqrt(ssr / Double(samples.count))
    // Covariance = (JᵀJ)⁻¹ σ², with σ² estimated from the residuals.
    var radiusSigma: Double?, heightSigma: Double?
    let dof = fit.residuals.count - 4
    if dof > 0 {
      var jtj = [[Double]](repeating: [0, 0, 0, 0], count: 4)
      for row in fit.jacobian { for a in 0..<4 { for b in 0..<4 { jtj[a][b] += row[a] * row[b] } } }
      if let covariance = Numerics.inverse(jtj) {
        let variance = ssr / Double(dof)
        radiusSigma = sqrt(max(0, covariance[2][2] * variance))
        heightSigma = sqrt(max(0, covariance[3][3] * variance))
      }
    }
    let total = views.reduce(0) { $0 + $1.weights.count }
    let visibleFraction = views.reduce(0) { $0 + $1.weights.reduce(0, +) } / Double(max(total, 1))
    let confidence = pow(0.5, rmsPx / config.confidenceHalfResidualPx) * min(1, visibleFraction / config.minVisibleFraction)
    return .success(RimEstimate(
      centerWorld: tablePlane.pointWorld(at: SIMD2(p[0], p[1]), heightM: p[3]),
      radiusM: p[2], heightM: p[3], radiusUncertaintyM: radiusSigma, heightUncertaintyM: heightSigma,
      rmsResidualPx: rmsPx, method: .twoView, viewCount: views.count, confidence: confidence))
  }
}
