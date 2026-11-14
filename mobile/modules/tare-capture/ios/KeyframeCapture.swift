import ARKit
import simd

// Capture layer (tare_impl.md §13): the only place ARKit types become MealScan
// pipeline values. Everything downstream takes Keyframe / DepthFrame / Plane.

extension CameraIntrinsics {
  init(_ camera: ARCamera) {
    let k = camera.intrinsics
    self.init(fxPx: Double(k[0][0]), fyPx: Double(k[1][1]), cxPx: Double(k[2][0]), cyPx: Double(k[2][1]),
              widthPx: Int(camera.imageResolution.width), heightPx: Int(camera.imageResolution.height))
  }
}

extension Keyframe {
  /// `includeDepth` copies LiDAR depth when the session provides it. Depth feeds only
  /// the LiDAR hook providers and ground truth, never rim sizing.
  init(frame: ARFrame, id: Int, includeDepth: Bool, sharpness: Double? = nil, imagePath: String? = nil) {
    let intrinsics = CameraIntrinsics(frame.camera)
    let worldFromCamera = simd_double4x4(frame.camera.transform)
    let tracking: TrackingState
    switch frame.camera.trackingState {
    case .normal: tracking = .normal
    case .limited: tracking = .limited
    case .notAvailable: tracking = .notAvailable
    }
    let depthData = includeDepth ? (frame.smoothedSceneDepth ?? frame.sceneDepth) : nil
    self.init(id: id, timestampS: frame.timestamp, intrinsics: intrinsics, worldFromCamera: worldFromCamera,
              tracking: tracking,
              featurePointsWorld: frame.rawFeaturePoints?.points.map { SIMD3<Double>($0) } ?? [],
              sharpness: sharpness, imagePath: imagePath,
              depth: depthData.flatMap { DepthFrame($0, imageIntrinsics: intrinsics, worldFromCamera: worldFromCamera) })
  }
}

extension DepthFrame {
  /// Copies ARKit depth (Float32 meters) and confidence out of their pixel buffers,
  /// so no ARFrame or buffer is retained by the pipeline.
  init?(_ data: ARDepthData, imageIntrinsics: CameraIntrinsics, worldFromCamera: simd_double4x4) {
    let map = data.depthMap
    guard CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32 else { return nil }
    let width = CVPixelBufferGetWidth(map), height = CVPixelBufferGetHeight(map)
    func copyRows<T>(_ buffer: CVPixelBuffer, as type: T.Type) -> [T]? {
      CVPixelBufferLockBaseAddress(buffer, .readOnly)
      defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
      guard let base = CVPixelBufferGetBaseAddress(buffer),
            CVPixelBufferGetWidth(buffer) == width, CVPixelBufferGetHeight(buffer) == height else { return nil }
      let stride = CVPixelBufferGetBytesPerRow(buffer)
      var values: [T] = []
      values.reserveCapacity(width * height)
      for y in 0..<height {
        values.append(contentsOf: UnsafeBufferPointer(start: (base + y * stride).assumingMemoryBound(to: T.self), count: width))
      }
      return values
    }
    guard let depth = copyRows(map, as: Float32.self) else { return nil }
    let confidence = data.confidenceMap.flatMap { copyRows($0, as: UInt8.self) }
    self.init(widthPx: width, heightPx: height, depthM: depth, confidence: confidence,
              intrinsics: imageIntrinsics.scaled(toWidthPx: width, heightPx: height), worldFromCamera: worldFromCamera)
  }
}

extension Plane {
  /// World-space plane of an ARKit plane anchor (anchor +Y is the plane normal).
  init(_ anchor: ARPlaneAnchor) {
    let worldFromAnchor = simd_double4x4(anchor.transform)
    let centerWorld = worldFromAnchor * SIMD4(SIMD3<Double>(anchor.center), 1)
    let normal = worldFromAnchor.columns.1
    self.init(pointWorld: SIMD3(centerWorld.x, centerWorld.y, centerWorld.z), normalWorld: SIMD3(normal.x, normal.y, normal.z))
  }
}
