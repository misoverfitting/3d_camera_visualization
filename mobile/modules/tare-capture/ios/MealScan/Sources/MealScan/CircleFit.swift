import Foundation
import simd

public struct Circle2D: Sendable {
  public var center: SIMD2<Double>
  public var radiusM: Double

  public func distanceM(_ point: SIMD2<Double>) -> Double { abs(simd_length(point - center) - radiusM) }
}

/// Circle fitting in metric plane coordinates. A round rim seen in one view and
/// projected onto any plane parallel to it stays a circle, so plane-space circle
/// fitting replaces image-space ellipse fitting.
public enum CircleFit {
  /// Algebraic (Kasa) fit: least squares on x² + y² + Dx + Ey + F = 0.
  public static func algebraic(_ points: [SIMD2<Double>]) -> Circle2D? {
    guard points.count >= 3 else { return nil }
    var ata = [[Double]](repeating: [0, 0, 0], count: 3), atb = [0.0, 0, 0]
    for p in points {
      let row = [p.x, p.y, 1.0], rhs = -(p.x * p.x + p.y * p.y)
      for a in 0..<3 { atb[a] += row[a] * rhs; for b in 0..<3 { ata[a][b] += row[a] * row[b] } }
    }
    guard let s = Numerics.solve(ata, atb) else { return nil }
    let center = SIMD2(-s[0] / 2, -s[1] / 2)
    let squared = simd_length_squared(center) - s[2]
    return squared > 0 ? Circle2D(center: center, radiusM: sqrt(squared)) : nil
  }

  /// Geometric refinement minimizing radial distances.
  public static func refined(_ points: [SIMD2<Double>], from initial: Circle2D) -> Circle2D? {
    guard points.count >= 3,
          let fit = Numerics.levenbergMarquardt(
            initial: [initial.center.x, initial.center.y, initial.radiusM], steps: [1e-7, 1e-7, 1e-7],
            maxIterations: 50, residuals: { p in points.map { simd_length($0 - SIMD2(p[0], p[1])) - p[2] } })
    else { return nil }
    return Circle2D(center: SIMD2(fit.params[0], fit.params[1]), radiusM: abs(fit.params[2]))
  }

  /// RANSAC over 3-point circles, then refinement on the best inlier set.
  public static func ransac<G: RandomNumberGenerator>(_ points: [SIMD2<Double>], inlierM: Double, iterations: Int,
                                                      using rng: inout G) -> (circle: Circle2D, inliers: [Int])? {
    guard points.count >= 3 else { return nil }
    var best: (circle: Circle2D, inliers: [Int])?
    for _ in 0..<iterations {
      let i = Int.random(in: 0..<points.count, using: &rng)
      let j = Int.random(in: 0..<points.count, using: &rng)
      let k = Int.random(in: 0..<points.count, using: &rng)
      guard i != j, j != k, i != k, let circle = through(points[i], points[j], points[k]) else { continue }
      let inliers = points.indices.filter { circle.distanceM(points[$0]) <= inlierM }
      if inliers.count > (best?.inliers.count ?? 2) { best = (circle, inliers) }
    }
    guard var result = best else { return nil }
    // Refit on the inliers, then tighten the band to ~3σ of the actual residuals
    // (robust σ from the median absolute residual) so outliers that happened to
    // fall inside the generous RANSAC band stop biasing the fit.
    var threshold = inlierM
    for _ in 0..<3 {
      let inlierPoints = result.inliers.map { points[$0] }
      guard let circle = refined(inlierPoints, from: algebraic(inlierPoints) ?? result.circle) else { break }
      let sigma = 1.4826 * (Numerics.percentile(inlierPoints.map(circle.distanceM), 0.5) ?? 0)
      threshold = min(inlierM, max(3 * sigma, inlierM / 20))
      let inliers = points.indices.filter { circle.distanceM(points[$0]) <= threshold }
      guard inliers.count >= 3 else { break }
      result = (circle, inliers)
    }
    return result
  }

  static func through(_ a: SIMD2<Double>, _ b: SIMD2<Double>, _ c: SIMD2<Double>) -> Circle2D? {
    let d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
    guard abs(d) > 1e-15 else { return nil }
    let a2 = simd_length_squared(a), b2 = simd_length_squared(b), c2 = simd_length_squared(c)
    let center = SIMD2((a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d,
                       (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d)
    return Circle2D(center: center, radiusM: simd_length(a - center))
  }
}
