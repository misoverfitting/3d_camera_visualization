import simd

public enum TrackingState: String, Codable, Sendable {
  case normal, limited, notAvailable
}

/// One captured view as plain values (tare_impl.md §6.1). Built from an ARFrame
/// only inside the capture layer; every pipeline module consumes this instead.
public struct Keyframe: Sendable {
  public var id: Int
  public var timestampS: Double
  public var intrinsics: CameraIntrinsics
  public var worldFromCamera: simd_double4x4
  public var tracking: TrackingState
  /// ARKit raw feature points in world coordinates.
  public var featurePointsWorld: [SIMD3<Double>]
  /// Sharpness score, higher is sharper (e.g. variance of Laplacian). Computed by
  /// the capture layer; nil when not measured.
  public var sharpness: Double?
  /// Opaque reference to the stored image (recording-relative path).
  public var imagePath: String?
  /// LiDAR depth from the same ARFrame. Only for LiDAR hook providers and ground
  /// truth; rim and container sizing must never read it (tare_impl.md §4).
  public var depth: DepthFrame?

  public init(id: Int, timestampS: Double, intrinsics: CameraIntrinsics, worldFromCamera: simd_double4x4,
              tracking: TrackingState = .normal, featurePointsWorld: [SIMD3<Double>] = [],
              sharpness: Double? = nil, imagePath: String? = nil, depth: DepthFrame? = nil) {
    self.id = id; self.timestampS = timestampS; self.intrinsics = intrinsics
    self.worldFromCamera = worldFromCamera; self.tracking = tracking
    self.featurePointsWorld = featurePointsWorld; self.sharpness = sharpness
    self.imagePath = imagePath; self.depth = depth
  }

  public var cameraCenterWorld: SIMD3<Double> {
    SIMD3(worldFromCamera.columns.3.x, worldFromCamera.columns.3.y, worldFromCamera.columns.3.z)
  }

  /// Unit viewing direction (camera -Z) in world coordinates.
  public var forwardWorld: SIMD3<Double> { simd_normalize(-worldFromCamera.transformDirection(SIMD3(0, 0, 1))) }

  public func rayWorld(throughPx pixel: SIMD2<Double>) -> Ray {
    Ray(originWorld: cameraCenterWorld,
        directionWorld: worldFromCamera.transformDirection(intrinsics.directionCamera(throughPx: pixel)))
  }

  public func projectPx(_ pointWorld: SIMD3<Double>) -> SIMD2<Double>? {
    intrinsics.projectPx(worldFromCamera.inverse.transformPoint(pointWorld))
  }
}

/// A LiDAR depth map (ARKit `sceneDepth`/`smoothedSceneDepth`) as plain values.
public struct DepthFrame: Sendable {
  public var widthPx: Int
  public var heightPx: Int
  /// Row-major depth along the camera's viewing axis (not ray length), meters; 0 = invalid.
  public var depthM: [Float]
  /// Row-major ARConfidenceLevel raw values (0 low, 1 medium, 2 high); nil = all high.
  public var confidence: [UInt8]?
  /// Intrinsics at the depth map's resolution (see `CameraIntrinsics.scaled`).
  public var intrinsics: CameraIntrinsics
  public var worldFromCamera: simd_double4x4

  public init(widthPx: Int, heightPx: Int, depthM: [Float], confidence: [UInt8]?,
              intrinsics: CameraIntrinsics, worldFromCamera: simd_double4x4) {
    precondition(depthM.count == widthPx * heightPx && (confidence?.count ?? depthM.count) == depthM.count)
    self.widthPx = widthPx; self.heightPx = heightPx; self.depthM = depthM
    self.confidence = confidence; self.intrinsics = intrinsics; self.worldFromCamera = worldFromCamera
  }

  /// World point at a depth pixel, or nil when invalid or below `minConfidence`.
  public func pointWorld(x: Int, y: Int, minConfidence: UInt8 = 2) -> SIMD3<Double>? {
    let index = y * widthPx + x
    let depth = Double(depthM[index])
    guard depth > 0, depth.isFinite, (confidence?[index] ?? 2) >= minConfidence else { return nil }
    let direction = intrinsics.directionCamera(throughPx: SIMD2(Double(x), Double(y)))
    return worldFromCamera.transformPoint(direction * depth)  // direction.z == -1
  }
}
