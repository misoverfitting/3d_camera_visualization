import Foundation

/// Area sampling in sensor coordinates. Fractional footprints preserve the same
/// pixel centers as CameraIntrinsics.scaled, including non-integer scale factors.
public enum LumaResampler {
  public static func downsample(_ bytes: UnsafeBufferPointer<UInt8>, width: Int, height: Int,
                                bytesPerRow: Int, targetWidth: Int) -> GrayImage? {
    guard width > 0, height > 0, targetWidth > 0, bytesPerRow >= width,
          bytes.count >= height * bytesPerRow else { return nil }
    let w = min(width, targetWidth)
    let h = max(1, Int((Double(height) * Double(w) / Double(width)).rounded()))
    let sx = Double(width) / Double(w), sy = Double(height) / Double(h)
    var pixels = [Float](repeating: 0, count: w * h)
    for y in 0..<h {
      let top = Double(y) * sy, bottom = Double(y + 1) * sy
      for x in 0..<w {
        let left = Double(x) * sx, right = Double(x + 1) * sx
        var sum = 0.0
        for iy in Int(floor(top))..<min(height, Int(ceil(bottom))) {
          let wy = min(bottom, Double(iy + 1)) - max(top, Double(iy))
          for ix in Int(floor(left))..<min(width, Int(ceil(right))) {
            let wx = min(right, Double(ix + 1)) - max(left, Double(ix))
            sum += Double(bytes[iy * bytesPerRow + ix]) * wx * wy
          }
        }
        pixels[y * w + x] = Float(sum / (sx * sy))
      }
    }
    return GrayImage(widthPx: w, heightPx: h, pixels: pixels)
  }
}

extension GrayImage {
  /// Laplacian variance on fitting-resolution luma, used only to compare images
  /// within one coverage cell, not as a device-independent quality threshold.
  public var sharpness: Double {
    guard widthPx > 2, heightPx > 2 else { return 0 }
    var sum = 0.0, squared = 0.0, count = 0.0
    for y in 1..<(heightPx - 1) {
      for x in 1..<(widthPx - 1) {
        let i = y * widthPx + x
        let lap = Double(pixels[i - 1] + pixels[i + 1] + pixels[i - widthPx] + pixels[i + widthPx] - 4 * pixels[i])
        sum += lap; squared += lap * lap; count += 1
      }
    }
    return max(0, squared / count - pow(sum / count, 2))
  }
}
