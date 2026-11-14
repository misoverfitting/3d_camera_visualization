import Foundation
import simd

/// Picks keyframes while the user orbits the dish: sharp (slow-moving) frames with the
/// dish in view, spread over the hemisphere so every part of the rim is seen from
/// several directions. Coverage drives the "keep moving" guidance.
public struct OrbitSelector {
  public enum Rejection: String, Sendable { case trackingLimited, moving, outOfFrame, tooLow, distance, redundant, full }

  public var config: PipelineConfig.Orbit
  public let targetWorld: SIMD3<Double>
  public let tablePlane: Plane
  /// Unit directions from the target to each accepted camera.
  public private(set) var directions: [SIMD3<Double>] = []
  private var cells = Set<Int>()

  public init(targetWorld: SIMD3<Double>, tablePlane: Plane, config: PipelineConfig.Orbit = .init()) {
    self.targetWorld = targetWorld; self.tablePlane = tablePlane; self.config = config
  }

  /// Fraction of hemisphere cells (azimuth x elevation band) holding a keyframe.
  public var coverage: Double {
    Double(cells.count) / Double(config.azimuthBins * (config.elevationBandsDeg.count - 1))
  }

  /// `angularSpeedDegS` and `speedMS` come from consecutive tracking poses.
  public mutating func consider(_ keyframe: Keyframe, angularSpeedDegS: Double, speedMS: Double) -> Rejection? {
    guard keyframe.tracking == .normal else { return .trackingLimited }
    guard angularSpeedDegS <= config.maxAngularSpeedDegS, speedMS <= config.maxSpeedMS else { return .moving }
    let k = keyframe.intrinsics
    guard let pixel = keyframe.projectPx(targetWorld),
          abs(pixel.x - k.cxPx) <= config.maxCenterOffsetFraction * Double(k.widthPx) / 2,
          abs(pixel.y - k.cyPx) <= config.maxCenterOffsetFraction * Double(k.heightPx) / 2 else { return .outOfFrame }
    let offset = keyframe.cameraCenterWorld - targetWorld
    let distanceM = simd_length(offset)
    guard distanceM >= config.minDistanceM, distanceM <= config.maxDistanceM else { return .distance }
    let direction = offset / distanceM
    let elevationDeg = asin(min(1, max(-1, simd_dot(direction, tablePlane.normalWorld)))) * 180 / .pi
    guard elevationDeg >= config.minElevationDeg else { return .tooLow }
    let minCos = cos(config.minAngleDeg * .pi / 180)
    guard !directions.contains(where: { simd_dot($0, direction) > minCos }) else { return .redundant }
    guard directions.count < config.maxKeyframes else { return .full }
    directions.append(direction)
    cells.insert(cell(direction, elevationDeg: elevationDeg))
    return nil
  }

  private func cell(_ direction: SIMD3<Double>, elevationDeg: Double) -> Int {
    let frame = tablePlane.planeFromWorld
    let local = frame * SIMD4(direction, 0)
    let azimuth = atan2(local.z, local.x) + .pi  // 0...2π
    let azimuthBin = min(config.azimuthBins - 1, Int(azimuth / (2 * .pi) * Double(config.azimuthBins)))
    let bands = config.elevationBandsDeg
    let band = (1..<bands.count).first { elevationDeg <= bands[$0] }.map { $0 - 1 } ?? bands.count - 2
    return band * config.azimuthBins + azimuthBin
  }
}
