import Foundation
import simd

/// The region of a keyframe image sent for food identification: a crop in sensor
/// pixels (edge coordinates, top-left origin), optionally rotated 90° clockwise so
/// the photo is upright. Maps the labeler's normalized points back to sensor pixels.
public struct ImageCrop: Sendable {
  public var x0Px: Double, y0Px: Double, widthPx: Double, heightPx: Double
  public var rotatedClockwise: Bool

  public init(x0Px: Double, y0Px: Double, widthPx: Double, heightPx: Double, rotatedClockwise: Bool) {
    self.x0Px = x0Px; self.y0Px = y0Px; self.widthPx = widthPx; self.heightPx = heightPx
    self.rotatedClockwise = rotatedClockwise
  }

  /// Crop around a rim circle (scaled by `margin`) as seen from `keyframe`, clamped to the image.
  public static func around(_ rim: RimEstimate, tablePlane: Plane, in keyframe: Keyframe, margin: Double,
                            rotatedClockwise: Bool) -> ImageCrop? {
    let frame = PlaneFrame(tablePlane), center = frame.coordinates(rim.centerWorld)
    let pixels = (0..<48).compactMap { i -> SIMD2<Double>? in
      let a = 2 * .pi * Double(i) / 48
      return keyframe.projectPx(frame.world(center + margin * rim.radiusM * SIMD2(cos(a), sin(a)), heightM: rim.heightM))
    }
    guard pixels.count == 48 else { return nil }
    let k = keyframe.intrinsics
    let x0 = max(0, pixels.map(\.x).min()! + 0.5), x1 = min(Double(k.widthPx), pixels.map(\.x).max()! + 0.5)
    let y0 = max(0, pixels.map(\.y).min()! + 0.5), y1 = min(Double(k.heightPx), pixels.map(\.y).max()! + 0.5)
    guard x1 - x0 >= 32, y1 - y0 >= 32 else { return nil }
    return ImageCrop(x0Px: x0.rounded(.down), y0Px: y0.rounded(.down),
                     widthPx: (x1 - x0).rounded(.down), heightPx: (y1 - y0).rounded(.down), rotatedClockwise: rotatedClockwise)
  }

  /// Normalized (0...1, top-left origin) point in the sent image → sensor pixel (pixel-center coordinates).
  public func sensorPixel(normalized p: SIMD2<Double>) -> SIMD2<Double> {
    let edge: SIMD2<Double>
    if rotatedClockwise {
      // Upright image is heightPx wide and widthPx tall; its +x runs up the sensor image.
      edge = SIMD2(x0Px + p.y * widthPx, y0Px + (1 - p.x) * heightPx)
    } else {
      edge = SIMD2(x0Px + p.x * widthPx, y0Px + p.y * heightPx)
    }
    return edge - 0.5
  }
}

public struct FoodPoint: Sendable {
  public var positionWorld: SIMD3<Double>
  /// True when the point came from depth; false when the ray was cut by the surface plane.
  public var fromDepth: Bool
}

/// Lifts labeled image points onto the plate in 3D.
public enum FoodPointLifter {
  /// Depth at the pixel when the keyframe has it and the hit lies on the dish; otherwise
  /// the pixel's ray cut by the plane `surfaceHeightM` above the table. Nil if off the dish.
  public static func lift(_ pixel: SIMD2<Double>, keyframe: Keyframe, rim: RimEstimate, tablePlane: Plane,
                          surfaceHeightM: Double) -> FoodPoint? {
    let frame = PlaneFrame(tablePlane), center = frame.coordinates(rim.centerWorld)
    func onDish(_ p: SIMD3<Double>) -> Bool {
      let h = frame.height(p)
      return simd_distance(frame.coordinates(p), center) <= rim.radiusM * 1.02 && h > -0.01 && h < rim.heightM + 0.15
    }
    if let depth = keyframe.depth {
      let sx = Double(depth.widthPx) / Double(keyframe.intrinsics.widthPx)
      let sy = Double(depth.heightPx) / Double(keyframe.intrinsics.heightPx)
      let dx = Int(((pixel.x + 0.5) * sx - 0.5).rounded()), dy = Int(((pixel.y + 0.5) * sy - 0.5).rounded())
      // Median-height valid sample in a 3x3 neighborhood resists single flying pixels.
      let samples = (-1...1).flatMap { oy in (-1...1).compactMap { ox -> SIMD3<Double>? in
        let x = dx + ox, y = dy + oy
        guard x >= 0, y >= 0, x < depth.widthPx, y < depth.heightPx else { return nil }
        return depth.pointWorld(x: x, y: y, minConfidence: 1)
      } }.filter(onDish)
      if !samples.isEmpty {
        let sorted = samples.sorted { frame.height($0) < frame.height($1) }
        return FoodPoint(positionWorld: sorted[sorted.count / 2], fromDepth: true)
      }
    }
    guard let hit = tablePlane.offset(byM: surfaceHeightM).intersection(keyframe.rayWorld(throughPx: pixel)),
          onDish(hit) else { return nil }
    return FoodPoint(positionWorld: hit, fromDepth: false)
  }
}
