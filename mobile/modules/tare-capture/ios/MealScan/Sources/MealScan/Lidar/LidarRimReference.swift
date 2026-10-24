import Foundation
import simd

/// Rim geometry measured directly from LiDAR depth, for evaluation only: it scores
/// the camera-only rim pipeline on LiDAR captures of the same scene (tare_impl.md §4,
/// "LiDAR depth may be recorded as an extra validation signal"). Never feed it into
/// production sizing; LiDAR phones run the same geometric pipeline as everyone else.
public struct RimReference: Sendable {
  public var centerWorld: SIMD3<Double>
  /// Outer rim radius: the extent of the elevated region seen from above.
  public var radiusM: Double
  /// Rim top above the table.
  public var heightM: Double
  public var boundarySamples: Int
  public var boundaryRmsM: Double

  public struct Comparison: Sendable {
    public var diameterErrorM: Double
    public var diameterErrorFraction: Double
    public var heightErrorM: Double
    public var centerOffsetM: Double
  }

  /// Signed errors of an estimate against this reference (estimate − reference).
  public func compare(_ estimate: RimEstimate) -> Comparison {
    Comparison(diameterErrorM: estimate.diameterM - 2 * radiusM,
               diameterErrorFraction: (estimate.diameterM - 2 * radiusM) / (2 * radiusM),
               heightErrorM: estimate.heightM - heightM,
               centerOffsetM: simd_distance(estimate.centerWorld, centerWorld))
  }
}

public struct LidarRimReference {
  public var config: PipelineConfig.Lidar

  public init(config: PipelineConfig.Lidar = .init()) { self.config = config }

  /// `approximateCenterWorld` seeds the search (e.g. the triangulated rim or a tap);
  /// points farther than `searchRadiusM` from it in the table plane are ignored.
  public func estimate(keyframes: [Keyframe], tablePlane: Plane, approximateCenterWorld: SIMD3<Double>,
                       searchRadiusM: Double, seed: UInt64 = 1) -> RimReference? {
    let seedCenter = tablePlane.coordinates2D(approximateCenterWorld)
    // Elevated dish/food points as (in-plane position, height above table).
    var elevated: [(position: SIMD2<Double>, heightM: Double)] = []
    for keyframe in keyframes {
      guard let depth = keyframe.depth else { continue }
      for p in depth.pointsWorld(minConfidence: config.minConfidence) {
        let position = tablePlane.coordinates2D(p), heightM = tablePlane.signedDistanceM(p)
        if heightM > config.elevatedThresholdM && simd_distance(position, seedCenter) <= searchRadiusM {
          elevated.append((position, heightM))
        }
      }
    }
    guard elevated.count >= 50 else { return nil }

    // Outer boundary: the farthest elevated point per angular bin around the current
    // center, then a robust circle fit; repeat once from the fitted center.
    var center = seedCenter
    var fit: (circle: Circle2D, inliers: [Int])?
    var boundary: [SIMD2<Double>] = []
    var rng = SplitMix64(seed: seed)
    for _ in 0..<2 {
      var farthest = [(radiusM: Double, position: SIMD2<Double>)?](repeating: nil, count: config.rimBoundaryBins)
      for e in elevated {
        let offset = e.position - center
        let radiusM = simd_length(offset)
        let bin = min(config.rimBoundaryBins - 1,
                      Int((atan2(offset.y, offset.x) + .pi) / (2 * .pi) * Double(config.rimBoundaryBins)))
        if radiusM > farthest[bin]?.radiusM ?? -1 { farthest[bin] = (radiusM, e.position) }
      }
      boundary = farthest.compactMap { $0?.position }
      guard boundary.count >= 12,
            let next = CircleFit.ransac(boundary, inlierM: 0.003, iterations: 300, using: &rng) else { return nil }
      fit = next; center = next.circle.center
    }
    guard let circle = fit?.circle, let inliers = fit?.inliers else { return nil }
    let rimHeights = elevated.filter {
      let r = simd_distance($0.position, circle.center)
      return r >= circle.radiusM - config.rimAnnulusM && r <= circle.radiusM + 0.002
    }.map(\.heightM)
    guard let heightM = Numerics.percentile(rimHeights, 0.95) else { return nil }
    let rms = sqrt(inliers.map { pow(circle.distanceM(boundary[$0]), 2) }.reduce(0, +) / Double(max(inliers.count, 1)))
    return RimReference(centerWorld: tablePlane.pointWorld(at: circle.center, heightM: heightM), radiusM: circle.radiusM,
                        heightM: heightM, boundarySamples: inliers.count, boundaryRmsM: rms)
  }
}
