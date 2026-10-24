import Foundation
import simd
@testable import MealScan

/// Synthetic captures with exactly known geometry (tare_impl.md §13: "rendered rim
/// at known r, h"). The table sits at a realistic ARKit world height, not y = 0, so
/// tests catch code that assumes the table is the world origin.
enum Synthetic {
  static let tableY = -0.72
  static let table = Plane(pointWorld: SIMD3(0.05, tableY, -0.30), normalWorld: SIMD3(0, 1, 0))
  /// Dish center on the table, in world coordinates.
  static let dishCenter = SIMD3(0.05, tableY, -0.30)
  /// iPhone-like camera at full RGB resolution.
  static let rgb = CameraIntrinsics(fxPx: 1450, fyPx: 1450, cxPx: 959.5, cyPx: 719.5, widthPx: 1920, heightPx: 1440)

  /// ARKit camera-to-world looking from `eye` at `target` (+Y up, camera looks along -Z).
  static func pose(eye: SIMD3<Double>, target: SIMD3<Double>) -> simd_double4x4 {
    let back = simd_normalize(eye - target)
    let right = simd_normalize(simd_cross(SIMD3(0, 1, 0), back))
    let up = simd_cross(back, right)
    return simd_double4x4(columns: (SIMD4(right, 0), SIMD4(up, 0), SIMD4(back, 0), SIMD4(eye, 1)))
  }

  /// Camera `heightM` above the table, offset horizontally from the dish center.
  static func keyframe(id: Int, offset: SIMD3<Double>, heightM: Double = 0.40,
                       intrinsics: CameraIntrinsics = rgb, depth: Heightfield? = nil) -> Keyframe {
    let eye = dishCenter + offset + SIMD3(0, heightM, 0)
    var keyframe = Keyframe(id: id, timestampS: Double(id), intrinsics: intrinsics,
                            worldFromCamera: pose(eye: eye, target: dishCenter), sharpness: 200)
    if let depth { keyframe.depth = render(depth, from: keyframe) }
    return keyframe
  }

  /// Pixels of `count` evenly spaced points on a rim circle, keeping those in frame.
  static func rimKeypoints(radiusM: Double, heightM: Double, in keyframe: Keyframe, count: Int = 32,
                           phase: Double = 0) -> [SIMD2<Double>] {
    let center = table.coordinates2D(dishCenter)
    return (0..<count).compactMap { i in
      let angle = phase + 2 * .pi * Double(i) / Double(count)
      let point = table.pointWorld(at: center + radiusM * SIMD2(cos(angle), sin(angle)), heightM: heightM)
      return keyframe.projectPx(point).flatMap { keyframe.intrinsics.contains($0) ? $0 : nil }
    }
  }

  /// A scene visible from above as a height over the table, as a function of the
  /// in-plane offset from the dish center. Overhangs are not representable.
  struct Heightfield {
    var maxHeightM: Double
    var heightM: (SIMD2<Double>) -> Double
  }

  /// Plate: flat floor, a lip rising to the rim, nothing outside the rim.
  static func plate(radiusM: Double, rimHeightM: Double, floorM: Double = 0.006, lipM: Double = 0.03,
                    extra: ((SIMD2<Double>) -> Double?)? = nil) -> Heightfield {
    Heightfield(maxHeightM: 0.12) { p in
      let r = simd_length(p)
      guard r <= radiusM else { return 0 }
      let surface = r < radiusM - lipM ? floorM : floorM + (rimHeightM - floorM) * (r - (radiusM - lipM)) / lipM
      return max(surface, extra?(p) ?? 0)
    }
  }

  /// Bowl: paraboloid interior from `floorM` at the center to the rim height.
  static func bowl(radiusM: Double, rimHeightM: Double, floorM: Double) -> Heightfield {
    Heightfield(maxHeightM: 0.15) { p in
      let r = simd_length(p)
      return r <= radiusM ? floorM + (rimHeightM - floorM) * (r * r) / (radiusM * radiusM) : 0
    }
  }

  /// LiDAR-like depth (256 x 192, depth along the view axis) by ray marching the heightfield.
  static func render(_ scene: Heightfield, from keyframe: Keyframe, widthPx: Int = 256, heightPx: Int = 192) -> DepthFrame {
    let intrinsics = keyframe.intrinsics.scaled(toWidthPx: widthPx, heightPx: heightPx)
    let center = table.coordinates2D(dishCenter)
    let cameraFromWorld = keyframe.worldFromCamera.inverse
    func above(_ p: SIMD3<Double>) -> Double {
      table.signedDistanceM(p) - scene.heightM(table.coordinates2D(p) - center)
    }
    var depth = [Float](repeating: 0, count: widthPx * heightPx)
    for y in 0..<heightPx {
      for x in 0..<widthPx {
        let ray = Ray(originWorld: keyframe.cameraCenterWorld, directionWorld: keyframe.worldFromCamera.transformDirection(
          intrinsics.directionCamera(throughPx: SIMD2(Double(x), Double(y)))))
        guard let top = table.offset(byM: scene.maxHeightM).intersection(ray) else { continue }
        var near = simd_distance(top, ray.originWorld), far = near
        while above(ray.pointWorld(atM: far)) > 0 && far < near + 2 { far += 0.001 }
        near = max(near, far - 0.001)
        for _ in 0..<30 {  // Bisect to sub-micrometer precision.
          let mid = (near + far) / 2
          if above(ray.pointWorld(atM: mid)) > 0 { near = mid } else { far = mid }
        }
        depth[y * widthPx + x] = Float(-cameraFromWorld.transformPoint(ray.pointWorld(atM: far)).z)
      }
    }
    return DepthFrame(widthPx: widthPx, heightPx: heightPx, depthM: depth, confidence: nil,
                      intrinsics: intrinsics, worldFromCamera: keyframe.worldFromCamera)
  }
}

/// Seeded Gaussian noise for reproducible noisy-input tests.
struct GaussianNoise {
  var rng: SplitMix64
  init(seed: UInt64) { rng = SplitMix64(seed: seed) }
  mutating func next(sigma: Double) -> Double {
    let u1 = max(Double.random(in: 0..<1, using: &rng), 1e-12), u2 = Double.random(in: 0..<1, using: &rng)
    return sigma * sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
  }
}
