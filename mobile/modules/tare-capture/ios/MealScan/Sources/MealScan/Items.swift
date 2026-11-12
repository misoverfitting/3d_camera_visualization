import simd

/// A binary mask in one keyframe's image pixels (row-major).
public struct ImageMask: Sendable {
  public var widthPx: Int
  public var heightPx: Int
  public var bits: [Bool]

  public init(widthPx: Int, heightPx: Int, bits: [Bool]) {
    precondition(bits.count == widthPx * heightPx)
    self.widthPx = widthPx; self.heightPx = heightPx; self.bits = bits
  }

  public init(widthPx: Int, heightPx: Int, contains: (Int, Int) -> Bool) {
    var bits = [Bool](repeating: false, count: widthPx * heightPx)
    for y in 0..<heightPx { for x in 0..<widthPx where contains(x, y) { bits[y * widthPx + x] = true } }
    self.init(widthPx: widthPx, heightPx: heightPx, bits: bits)
  }

  public subscript(x: Int, y: Int) -> Bool {
    x >= 0 && y >= 0 && x < widthPx && y < heightPx && bits[y * widthPx + x]
  }

  public var area: Int { bits.reduce(0) { $0 + ($1 ? 1 : 0) } }

  /// Mask pixels with a 4-neighbor outside the mask, as pixel coordinates.
  public func boundaryPx() -> [SIMD2<Double>] {
    var points: [SIMD2<Double>] = []
    for y in 0..<heightPx {
      for x in 0..<widthPx where self[x, y] && (!self[x - 1, y] || !self[x + 1, y] || !self[x, y - 1] || !self[x, y + 1]) {
        points.append(SIMD2(Double(x), Double(y)))
      }
    }
    return points
  }

  /// Removes pixels within `radiusPx` of the boundary (square structuring element).
  public func eroded(radiusPx: Int) -> ImageMask {
    guard radiusPx > 0 else { return self }
    return ImageMask(widthPx: widthPx, heightPx: heightPx) { x, y in
      for dy in -radiusPx...radiusPx { for dx in -radiusPx...radiusPx where !self[x + dx, y + dy] { return false } }
      return true
    }
  }
}

/// One food item (tare_impl.md §6.9–6.12), segmented in a specific keyframe.
public struct ItemSegment: Sendable {
  public var id: Int
  public var keyframeId: Int
  public var mask: ImageMask
  public var label: String?
  public var confidence: Double?

  public init(id: Int, keyframeId: Int, mask: ImageMask, label: String? = nil, confidence: Double? = nil) {
    self.id = id; self.keyframeId = keyframeId; self.mask = mask; self.label = label; self.confidence = confidence
  }
}

/// Which plane an item height is measured from. tare_impl.md §6.15 specifies the
/// plate/rim plane; the table is offered for volume work, which needs a floor.
public enum HeightReference: String, Codable, Sendable { case rimPlane, table }

public struct ItemHeight: Sendable {
  /// Height of the item's top surface above `reference`.
  public var valueM: Double
  public var uncertaintyM: Double
  public var reference: HeightReference
  public var source: MeasurementSource
  /// Mean surface height above `reference`, when the provider measures a surface.
  public var meanSurfaceM: Double?
  public var sampleCount: Int
}

/// Hook for item heights (tare_impl.md §6.15).
public protocol ItemHeightProvider {
  /// Height of the item's top surface above the plate/rim plane, in meters.
  func height(for item: ItemSegment, container: ContainerGeometry, keyframes: [Keyframe]) -> ItemHeight?
}

/// Default provider: no height; footprints project onto the plate plane uncorrected.
public struct NoItemHeight: ItemHeightProvider {
  public init() {}
  public func height(for item: ItemSegment, container: ContainerGeometry, keyframes: [Keyframe]) -> ItemHeight? { nil }
}
