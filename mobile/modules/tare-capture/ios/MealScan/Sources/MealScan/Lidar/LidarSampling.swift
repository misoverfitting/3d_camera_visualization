import simd

extension DepthFrame {
  /// World points of depth pixels, optionally restricted to a mask drawn on an image
  /// of the same camera at another resolution. With `erosionPx` > 0, a depth pixel is
  /// kept only if every depth pixel within that radius also falls in the mask, which
  /// drops mixed-depth samples along item edges.
  func pointsWorld(insideMask mask: ImageMask? = nil, erosionPx: Int = 0, minConfidence: UInt8) -> [SIMD3<Double>] {
    let sx = mask.map { Double($0.widthPx) / Double(widthPx) } ?? 1
    let sy = mask.map { Double($0.heightPx) / Double(heightPx) } ?? 1
    func inMask(_ x: Int, _ y: Int) -> Bool {
      guard let mask else { return true }
      guard x >= 0, y >= 0, x < widthPx, y < heightPx else { return false }
      // Depth pixel center in mask pixels (integer-center convention).
      return mask[Int(((Double(x) + 0.5) * sx - 0.5).rounded()), Int(((Double(y) + 0.5) * sy - 0.5).rounded())]
    }
    var points: [SIMD3<Double>] = []
    for y in 0..<heightPx {
      for x in 0..<widthPx where inMask(x, y) {
        if mask != nil && erosionPx > 0 {
          var interior = true
          for dy in -erosionPx...erosionPx where interior {
            for dx in -erosionPx...erosionPx where !inMask(x + dx, y + dy) { interior = false; break }
          }
          if !interior { continue }
        }
        if let p = pointWorld(x: x, y: y, minConfidence: minConfidence) { points.append(p) }
      }
    }
    return points
  }
}
