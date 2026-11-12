import ARKit
import simd

/// Where metric 3D points come from. Both sources write into the same `PointCloud`,
/// so a LiDAR scan can serve as ground truth for the multi-view path on one device.
enum DepthMode: String {
  case lidar
  case multiView = "multiview"

  static var lidarSupported: Bool { ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) }
}

/// The world-aligned scan box: origin at bottom center, extending upward. A small
/// margin below the origin keeps the support surface, which plate fitting needs.
struct ScanBounds {
  let origin: SIMD3<Float>
  let size: Float

  func contains(_ p: SIMD3<Float>) -> Bool {
    let d = p - origin, half = size / 2
    return abs(d.x) <= half && abs(d.z) <= half && d.y >= -0.02 && d.y <= size
  }
}

/// Voxel-averaged world points. Accessed only on the depth queue.
final class PointCloud {
  struct Cell { var sum: SIMD3<Float>; var count: UInt32 }

  let voxel: Float
  let capacity: Int
  private var cells: [SIMD3<Int32>: Cell] = [:]

  init(voxel: Float = 0.005, capacity: Int = 400_000) {
    self.voxel = voxel; self.capacity = capacity
  }

  var count: Int { cells.count }

  func add(_ points: [SIMD3<Float>]) {
    for p in points where p.x.isFinite && p.y.isFinite && p.z.isFinite {
      let key = SIMD3<Int32>(p / voxel, rounding: .down)
      if var cell = cells[key] {
        cell.sum += p; cell.count += 1; cells[key] = cell
      } else if cells.count < capacity {
        cells[key] = Cell(sum: p, count: 1)
      }
    }
  }

  func clear() { cells.removeAll(keepingCapacity: true) }

  /// Mean position of each voxel observed at least `minCount` times, thinned to `limit`.
  func points(minCount: UInt32 = 1, limit: Int = .max) -> [SIMD3<Float>] {
    let kept = cells.values.filter { $0.count >= minCount }
    let step = max(1, kept.count / max(1, limit))
    return Swift.stride(from: 0, to: kept.count, by: step).map { kept[$0].sum / Float(kept[$0].count) }
  }

  /// Binary little-endian PLY with per-voxel observation counts for downstream filtering.
  func plyData() -> Data {
    var data = Data("""
      ply
      format binary_little_endian 1.0
      element vertex \(cells.count)
      property float x
      property float y
      property float z
      property uint observations
      end_header

      """.utf8)
    data.reserveCapacity(data.count + cells.count * 16)
    for cell in cells.values {
      let mean = cell.sum / Float(cell.count)
      for value in [mean.x, mean.y, mean.z] { withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) } }
      withUnsafeBytes(of: cell.count.littleEndian) { data.append(contentsOf: $0) }
    }
    return data
  }
}

enum LidarSampler {
  /// Unprojects high-confidence scene-depth pixels into ARKit world coordinates.
  /// The depth map is aligned with, and lower resolution than, the captured image.
  static func worldPoints(_ depth: ARDepthData, intrinsics k: simd_float3x3, imageSize: CGSize,
                          cameraToWorld: simd_float4x4, step: Int = 2, maxDepth: Float = 1.5) -> [SIMD3<Float>] {
    let map = depth.depthMap
    guard let confidence = depth.confidenceMap else { return [] }
    CVPixelBufferLockBaseAddress(map, .readOnly); CVPixelBufferLockBaseAddress(confidence, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(confidence, .readOnly); CVPixelBufferUnlockBaseAddress(map, .readOnly)
    }
    guard let depthBase = CVPixelBufferGetBaseAddress(map),
          let confidenceBase = CVPixelBufferGetBaseAddress(confidence) else { return [] }
    let width = CVPixelBufferGetWidth(map), height = CVPixelBufferGetHeight(map)
    let depthStride = CVPixelBufferGetBytesPerRow(map), confidenceStride = CVPixelBufferGetBytesPerRow(confidence)
    let sx = Float(imageSize.width) / Float(width), sy = Float(imageSize.height) / Float(height)
    let fx = k[0][0], fy = k[1][1], cx = k[2][0], cy = k[2][1]
    let high = UInt8(ARConfidenceLevel.high.rawValue)
    var points: [SIMD3<Float>] = []
    points.reserveCapacity((width / step) * (height / step))
    for y in Swift.stride(from: 0, to: height, by: step) {
      let depthRow = (depthBase + y * depthStride).assumingMemoryBound(to: Float32.self)
      let confidenceRow = (confidenceBase + y * confidenceStride).assumingMemoryBound(to: UInt8.self)
      for x in Swift.stride(from: 0, to: width, by: step) {
        let d = depthRow[x]
        guard confidenceRow[x] >= high, d > 0.05, d < maxDepth else { continue }
        // Depth pixel center in captured-image pixel coordinates.
        let u = (Float(x) + 0.5) * sx - 0.5, v = (Float(y) + 0.5) * sy - 0.5
        // Image rays are +Y down / +Z forward; ARKit cameras are +Y up / -Z forward.
        let world = cameraToWorld * SIMD4<Float>((u - cx) / fx * d, -(v - cy) / fy * d, -d, 1)
        points.append(SIMD3(world.x, world.y, world.z))
      }
    }
    return points
  }
}
