import simd

public enum ContainerType: String, Codable, Sendable { case plate, bowl, other }
public enum ContainerShape: String, Codable, Sendable { case round, polygon }
public enum SizingMethod: String, Codable, Sendable { case twoView, singleView, orbitEdges }

/// Rim keypoints detected in one keyframe (tare_impl.md §6.4): pixels on the outer
/// rim, each with a weight in 0...1 from predicted visibility/uncertainty. Keypoints
/// need not correspond across views.
public struct RimObservation: Sendable {
  public var keyframe: Keyframe
  public var keypointsPx: [SIMD2<Double>]
  public var weights: [Double]

  public init(keyframe: Keyframe, keypointsPx: [SIMD2<Double>], weights: [Double]? = nil) {
    precondition(weights == nil || weights!.count == keypointsPx.count)
    self.keyframe = keyframe; self.keypointsPx = keypointsPx
    self.weights = weights ?? Array(repeating: 1, count: keypointsPx.count)
  }
}

/// A round rim modeled as a circle parallel to the table plane.
public struct RimEstimate: Sendable {
  public var centerWorld: SIMD3<Double>
  public var radiusM: Double
  /// Rim height above the table plane.
  public var heightM: Double
  /// One-sigma uncertainties from the fit covariance; nil when not estimable.
  public var radiusUncertaintyM: Double?
  public var heightUncertaintyM: Double?
  public var rmsResidualPx: Double
  public var method: SizingMethod
  public var viewCount: Int
  /// Heuristic 0...1 (see `PipelineConfig.Rim`).
  public var confidence: Double

  public var diameterM: Double { 2 * radiusM }
}

public struct ContainerGeometry: Sendable {
  public var type: ContainerType
  public var shape: ContainerShape
  public var tablePlane: Plane
  public var rim: RimEstimate

  public init(type: ContainerType, shape: ContainerShape = .round, tablePlane: Plane, rim: RimEstimate) {
    self.type = type; self.shape = shape; self.tablePlane = tablePlane; self.rim = rim
  }

  /// The plane through the rim, parallel to the table.
  public var rimPlane: Plane { tablePlane.offset(byM: rim.heightM) }
}

public enum MeasurementSource: String, Codable, Sendable { case lidar, probe, estimated }

/// Container cross-section (tare_impl.md §6.8). Radial samples run from the rim
/// center outward; depth is measured down from the rim plane.
public struct ContainerProfile: Sendable {
  public struct Sample: Sendable {
    public var radiusM: Double
    public var depthBelowRimM: Double
    public var sampleCount: Int
  }

  public var samples: [Sample]
  /// Rim plane to the lowest observed interior surface.
  public var innerDepthM: Double
  public var source: MeasurementSource
  /// The profile is the visible surface: where food covers the interior it
  /// measures the food, not the container. Consumers decide how to use that.
  public var measuresVisibleSurface: Bool
}

/// Hook for bowl interior / cross-section estimation (tare_impl.md §6.8).
public protocol ContainerProfileProvider {
  func profile(for container: ContainerGeometry, keyframes: [Keyframe]) -> ContainerProfile?
}

/// Default provider: no profile.
public struct NoContainerProfile: ContainerProfileProvider {
  public init() {}
  public func profile(for container: ContainerGeometry, keyframes: [Keyframe]) -> ContainerProfile? { nil }
}
