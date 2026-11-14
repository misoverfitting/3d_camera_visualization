import simd

// Conventions (tare_impl.md §13):
// - World: ARKit world frame, meters, +Y up (gravity aligned).
// - Camera: ARKit camera axes, +X right, +Y up, +Z backward; the camera looks along -Z.
//   Poses are named `worldFromCamera` (camera-to-world).
// - Image: native sensor orientation (unrotated), u right, v down, integer pixel
//   centers, matching ARKit's intrinsics.

/// Pinhole intrinsics for one image resolution.
public struct CameraIntrinsics: Equatable, Sendable {
  public var fxPx: Double
  public var fyPx: Double
  public var cxPx: Double
  public var cyPx: Double
  public var widthPx: Int
  public var heightPx: Int

  public init(fxPx: Double, fyPx: Double, cxPx: Double, cyPx: Double, widthPx: Int, heightPx: Int) {
    self.fxPx = fxPx; self.fyPx = fyPx; self.cxPx = cxPx; self.cyPx = cyPx
    self.widthPx = widthPx; self.heightPx = heightPx
  }

  /// The same camera at another resolution with the same field of view (e.g. a
  /// LiDAR depth map or a model input), preserving the pixel-center convention.
  public func scaled(toWidthPx width: Int, heightPx height: Int) -> CameraIntrinsics {
    let sx = Double(width) / Double(widthPx), sy = Double(height) / Double(heightPx)
    return CameraIntrinsics(fxPx: fxPx * sx, fyPx: fyPx * sy,
                            cxPx: (cxPx + 0.5) * sx - 0.5, cyPx: (cyPx + 0.5) * sy - 0.5,
                            widthPx: width, heightPx: height)
  }

  /// Unnormalized ray direction in camera axes through a pixel.
  public func directionCamera(throughPx pixel: SIMD2<Double>) -> SIMD3<Double> {
    SIMD3((pixel.x - cxPx) / fxPx, -(pixel.y - cyPx) / fyPx, -1)
  }

  /// Pixel of a camera-frame point, or nil when it is not in front of the camera.
  public func projectPx(_ pointCamera: SIMD3<Double>) -> SIMD2<Double>? {
    let depthM = -pointCamera.z
    guard depthM > 1e-9 else { return nil }
    return SIMD2(cxPx + fxPx * pointCamera.x / depthM, cyPx - fyPx * pointCamera.y / depthM)
  }

  public func contains(_ pixel: SIMD2<Double>) -> Bool {
    pixel.x >= -0.5 && pixel.y >= -0.5 && pixel.x < Double(widthPx) - 0.5 && pixel.y < Double(heightPx) - 0.5
  }
}

public struct Ray: Sendable {
  public var originWorld: SIMD3<Double>
  /// Unit length.
  public var directionWorld: SIMD3<Double>

  public init(originWorld: SIMD3<Double>, directionWorld: SIMD3<Double>) {
    self.originWorld = originWorld; self.directionWorld = simd_normalize(directionWorld)
  }

  public func pointWorld(atM distanceM: Double) -> SIMD3<Double> { originWorld + distanceM * directionWorld }
}

/// An oriented plane, e.g. the table or a rim plane parallel to it.
public struct Plane: Sendable {
  public var pointWorld: SIMD3<Double>
  /// Unit length; points away from the table (up) for support planes.
  public var normalWorld: SIMD3<Double>

  public init(pointWorld: SIMD3<Double>, normalWorld: SIMD3<Double>) {
    self.pointWorld = pointWorld; self.normalWorld = simd_normalize(normalWorld)
  }

  public func signedDistanceM(_ pointWorld: SIMD3<Double>) -> Double {
    simd_dot(pointWorld - self.pointWorld, normalWorld)
  }

  /// The parallel plane `distanceM` along the normal.
  public func offset(byM distanceM: Double) -> Plane {
    Plane(pointWorld: pointWorld + distanceM * normalWorld, normalWorld: normalWorld)
  }

  /// Forward intersection, or nil when the ray is parallel or points away.
  public func intersection(_ ray: Ray) -> SIMD3<Double>? {
    let denominator = simd_dot(ray.directionWorld, normalWorld)
    guard abs(denominator) > 1e-12 else { return nil }
    let distanceM = simd_dot(pointWorld - ray.originWorld, normalWorld) / denominator
    return distanceM > 0 ? ray.pointWorld(atM: distanceM) : nil
  }

  /// Plane frame: origin at `pointWorld`, +Y along the normal, X/Z spanning the plane.
  /// For a gravity-aligned table, plane X/Z follow world X/Z.
  public var worldFromPlane: simd_double4x4 {
    let y = normalWorld
    let reference = abs(y.x) < 0.9 ? SIMD3<Double>(1, 0, 0) : SIMD3<Double>(0, 0, 1)
    let z = simd_normalize(simd_cross(reference, y))
    let x = simd_cross(y, z)
    return simd_double4x4(columns: (SIMD4(x, 0), SIMD4(y, 0), SIMD4(z, 0), SIMD4(pointWorld, 1)))
  }

  public var planeFromWorld: simd_double4x4 { worldFromPlane.inverse }

  /// In-plane 2D coordinates (plane X, plane Z) of a world point.
  public func coordinates2D(_ pointWorld: SIMD3<Double>) -> SIMD2<Double> {
    let p = planeFromWorld * SIMD4(pointWorld, 1)
    return SIMD2(p.x, p.z)
  }

  /// World point at in-plane coordinates, `heightM` along the normal.
  public func pointWorld(at coordinates: SIMD2<Double>, heightM: Double = 0) -> SIMD3<Double> {
    let p = worldFromPlane * SIMD4(coordinates.x, heightM, coordinates.y, 1)
    return SIMD3(p.x, p.y, p.z)
  }
}

extension simd_double4x4 {
  /// Converts an ARKit `simd_float4x4` without losing the frame semantics.
  public init(_ m: simd_float4x4) {
    self.init(columns: (SIMD4<Double>(m.columns.0), SIMD4<Double>(m.columns.1),
                        SIMD4<Double>(m.columns.2), SIMD4<Double>(m.columns.3)))
  }

  func transformPoint(_ p: SIMD3<Double>) -> SIMD3<Double> {
    let r = self * SIMD4(p, 1)
    return SIMD3(r.x, r.y, r.z)
  }

  func transformDirection(_ d: SIMD3<Double>) -> SIMD3<Double> {
    let r = self * SIMD4(d, 0)
    return SIMD3(r.x, r.y, r.z)
  }
}
