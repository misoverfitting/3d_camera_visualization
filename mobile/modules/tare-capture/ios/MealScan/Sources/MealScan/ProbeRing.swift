import Foundation
import simd

public struct ProbeSpot: Sendable {
  public var index: Int
  public var positionWorld: SIMD3<Double>
  public var surfaceHeightM: Double
  /// False when no 3D points were near the spot and the fallback height was used.
  public var measuredSurface: Bool
}

/// Spectrometer probe spots evenly spaced on a ring a fixed inset inside the rim, placed
/// on the visible surface (plate or food) where 3D points exist.
public enum ProbeRing {
  public static func spots(rim: RimEstimate, tablePlane: Plane, surfacePointsWorld: [SIMD3<Double>] = [],
                           startAngleRad: Double = 0, config: PipelineConfig.Probe = .init()) -> [ProbeSpot] {
    let radiusM = rim.radiusM - config.insetM
    guard radiusM > 0, config.count > 0 else { return [] }
    let frame = PlaneFrame(tablePlane)
    let center = frame.coordinates(rim.centerWorld)
    let surface = surfacePointsWorld.map { (position: frame.coordinates($0), heightM: frame.height($0)) }
      .filter { $0.heightM > -0.005 && $0.heightM < rim.heightM + 0.1 && simd_distance($0.position, center) < rim.radiusM }
    return (0..<config.count).map { i in
      let angle = startAngleRad + 2 * .pi * Double(i) / Double(config.count)
      let position = center + radiusM * SIMD2(cos(angle), sin(angle))
      let nearby = surface.filter { simd_distance($0.position, position) <= config.surfaceSearchRadiusM }.map(\.heightM)
      let measured = nearby.count >= 3 ? Numerics.percentile(nearby, 0.5) : nil
      let heightM = measured ?? min(config.fallbackSurfaceHeightM, rim.heightM)
      return ProbeSpot(index: i, positionWorld: frame.world(position, heightM: heightM),
                       surfaceHeightM: heightM, measuredSurface: measured != nil)
    }
  }
}
