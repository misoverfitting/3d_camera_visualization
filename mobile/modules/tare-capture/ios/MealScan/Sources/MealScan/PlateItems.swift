import Foundation
import simd

/// An RGBA8 image (row-major, 4 bytes per pixel, top-left origin).
public struct RGBImage: Sendable {
  public var widthPx: Int
  public var heightPx: Int
  public var rgba: [UInt8]

  public init(widthPx: Int, heightPx: Int, rgba: [UInt8]) {
    precondition(rgba.count == widthPx * heightPx * 4)
    self.widthPx = widthPx; self.heightPx = heightPx; self.rgba = rgba
  }
}

/// One food item found on the plate, on the analysis grid.
public struct ItemRegion: Sendable {
  public var widthPx: Int
  public var heightPx: Int
  public var mask: [Bool]
  public var areaPx: Int
  /// Normalized [x0, y0, x1, y1], top-left origin.
  public var box: SIMD4<Double>
  /// 1-3 points well inside the item, normalized.
  public var points: [SIMD2<Double>]
}

/// Zero-training item proposals for the on-device food ID (tare_impl.md §6.9 baseline):
/// pixels inside the rim that differ from the bare plate in color or texture are food;
/// food pixels are split into items by color clusters and connected components. Each
/// item gets 2-3 points from its distance transform, so they sit well inside it.
public enum PlateItems {
  public struct Config: Sendable {
    /// Items smaller than this fraction of the plate interior are dropped (§6.10: 1.5%).
    public var minItemFraction = 0.015
    /// Color distance from the plate (CIELAB, lightness down-weighted) that counts as food.
    public var colorThreshold = 13.0
    public var lightnessWeight = 0.35
    /// Texture above the plate's by this many MADs (and above `minTexture`) counts as food.
    public var textureMADs = 4.0
    public var minTexture = 3.5
    public var maxClusters = 4
    /// A cluster is split off only when it cuts the color spread by at least this much.
    public var clusterGain = 0.25
    public init() {}
  }

  /// Pixels inside a normalized polygon.
  public static func rasterize(_ polygon: [SIMD2<Double>], widthPx w: Int, heightPx h: Int) -> [Bool] {
    var out = [Bool](repeating: false, count: w * h)
    guard polygon.count >= 3 else { return out }
    let pts = polygon.map { SIMD2($0.x * Double(w), $0.y * Double(h)) }
    for y in 0..<h {
      let cy = Double(y) + 0.5
      var xs: [Double] = []
      for i in 0..<pts.count {
        let a = pts[i], b = pts[(i + 1) % pts.count]
        if (a.y <= cy) != (b.y <= cy) { xs.append(a.x + (cy - a.y) / (b.y - a.y) * (b.x - a.x)) }
      }
      xs.sort()
      var k = 0
      while k + 1 < xs.count {
        let x0 = max(0, Int((xs[k] - 0.5).rounded(.up))), x1 = min(w - 1, Int((xs[k + 1] - 0.5).rounded(.down)))
        if x0 <= x1 { for x in x0...x1 { out[y * w + x] = true } }
        k += 2
      }
    }
    return out
  }

  /// Food items inside `interior`. `ring` marks pixels that are almost surely bare plate
  /// (a band just inside the rim) and sets the plate's color and texture.
  public static func segment(_ image: RGBImage, interior: [Bool], ring: [Bool], config: Config = .init()) -> [ItemRegion] {
    let w = image.widthPx, h = image.heightPx, n = w * h
    precondition(interior.count == n && ring.count == n)
    let lab = labImage(image)
    let texture = textureImage(lab, w, h)
    let ringIdx = (0..<n).filter { ring[$0] }
    let interiorArea = interior.reduce(0) { $0 + ($1 ? 1 : 0) }
    guard ringIdx.count >= 20, interiorArea >= 100 else { return [] }

    let plate = SIMD3(median(ringIdx.map { lab[$0].x }), median(ringIdx.map { lab[$0].y }), median(ringIdx.map { lab[$0].z }))
    let ringTexture = ringIdx.map { texture[$0] }
    let textureMedian = median(ringTexture)
    let textureMAD = max(0.5, median(ringTexture.map { abs($0 - textureMedian) }))
    let textureLimit = max(config.minTexture, textureMedian + config.textureMADs * textureMAD)
    let weight = SIMD3(config.lightnessWeight, 1, 1)

    var food = [Bool](repeating: false, count: n), textured = [Bool](repeating: false, count: n)
    for i in 0..<n where interior[i] {
      food[i] = simd_length((lab[i] - plate) * weight) > config.colorThreshold
      textured[i] = texture[i] > textureLimit
    }
    // The 5x5 texture window bleeds 2 px past every edge; trim that halo so plate pixels
    // next to an item don't become a ring-shaped "item" of plate color.
    textured = dilate(erode(erode(textured, w, h), w, h), w, h)
    for i in 0..<n where textured[i] { food[i] = true }
    food = open(close(food, w, h, 1), w, h, 2)
    for i in 0..<n where !interior[i] { food[i] = false }

    let foodIdx = (0..<n).filter { food[$0] }
    let minArea = max(12, Int(config.minItemFraction * Double(interiorArea)))
    guard foodIdx.count >= minArea else { return [] }
    let features = foodIdx.map { lab[$0] * weight }
    let assignment = clusters(features, maxK: config.maxClusters, gain: config.clusterGain)
    let k = (assignment.max() ?? 0) + 1

    var regions: [ItemRegion] = []
    for c in 0..<k {
      var mask = [Bool](repeating: false, count: n)
      for (j, i) in foodIdx.enumerated() where assignment[j] == c { mask[i] = true }
      mask = open(close(mask, w, h, 1), w, h, 1)
      for component in components(mask, w, h) where component.count >= minArea {
        var item = [Bool](repeating: false, count: n)
        var x0 = w, y0 = h, x1 = 0, y1 = 0
        for i in component {
          item[i] = true
          let x = i % w, y = i / w
          x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
        }
        let count = Double(component.count) >= 0.04 * Double(interiorArea) ? 3 : 2
        regions.append(ItemRegion(widthPx: w, heightPx: h, mask: item, areaPx: component.count,
                                  box: SIMD4(Double(x0) / Double(w), Double(y0) / Double(h), Double(x1 + 1) / Double(w), Double(y1 + 1) / Double(h)),
                                  points: points(in: item, widthPx: w, heightPx: h, count: count)))
      }
    }
    return regions.sorted { $0.areaPx > $1.areaPx }
  }

  /// Up to `count` points deep inside the mask and spread apart: the deepest pixel first,
  /// then the deep pixel (at least 45% of the maximum depth) farthest from those chosen.
  public static func points(in mask: [Bool], widthPx w: Int, heightPx h: Int, count: Int) -> [SIMD2<Double>] {
    let depth = distanceTransform(mask, w, h)
    guard let maxDepth = depth.max(), maxDepth > 0 else { return [] }
    let area = mask.reduce(0) { $0 + ($1 ? 1 : 0) }
    let spacing = 0.25 * sqrt(Double(area))
    let deep = (0..<(w * h)).filter { depth[$0] >= 0.45 * maxDepth }
    var chosen = [depth.firstIndex(of: maxDepth)!]
    func xy(_ i: Int) -> SIMD2<Double> { SIMD2(Double(i % w), Double(i / w)) }
    while chosen.count < count {
      guard let next = deep.max(by: { a, b in
        chosen.map { simd_distance(xy(a), xy($0)) }.min()! < chosen.map { simd_distance(xy(b), xy($0)) }.min()!
      }), chosen.map({ simd_distance(xy(next), xy($0)) }).min()! >= spacing else { break }
      chosen.append(next)
    }
    return chosen.map { SIMD2((Double($0 % w) + 0.5) / Double(w), (Double($0 / w) + 0.5) / Double(h)) }
  }

  // MARK: - Pixel helpers

  static func labImage(_ image: RGBImage) -> [SIMD3<Double>] {
    func linear(_ v: UInt8) -> Double { let c = Double(v) / 255; return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
    func f(_ t: Double) -> Double { t > 0.008856 ? cbrt(t) : 7.787 * t + 16 / 116 }
    let table = (0..<256).map { linear(UInt8($0)) }
    return (0..<(image.widthPx * image.heightPx)).map { i in
      let r = table[Int(image.rgba[4 * i])], g = table[Int(image.rgba[4 * i + 1])], b = table[Int(image.rgba[4 * i + 2])]
      let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
      let y = 0.2126 * r + 0.7152 * g + 0.0722 * b
      let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
      let fx = f(x), fy = f(y), fz = f(z)
      return SIMD3(116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))
    }
  }

  /// Mean lightness gradient magnitude over a 5x5 window.
  static func textureImage(_ lab: [SIMD3<Double>], _ w: Int, _ h: Int) -> [Double] {
    var gradient = [Double](repeating: 0, count: w * h)
    for y in 1..<max(1, h - 1) { for x in 1..<max(1, w - 1) {
      let gx = lab[y * w + x + 1].x - lab[y * w + x - 1].x, gy = lab[(y + 1) * w + x].x - lab[(y - 1) * w + x].x
      gradient[y * w + x] = 0.5 * sqrt(gx * gx + gy * gy)
    } }
    return boxMean(gradient, w, h, radius: 2)
  }

  static func boxMean(_ values: [Double], _ w: Int, _ h: Int, radius r: Int) -> [Double] {
    var integral = [Double](repeating: 0, count: (w + 1) * (h + 1))
    for y in 0..<h { var row = 0.0; for x in 0..<w { row += values[y * w + x]; integral[(y + 1) * (w + 1) + x + 1] = integral[y * (w + 1) + x + 1] + row } }
    var out = [Double](repeating: 0, count: w * h)
    for y in 0..<h { for x in 0..<w {
      let x0 = max(0, x - r), x1 = min(w, x + r + 1), y0 = max(0, y - r), y1 = min(h, y + r + 1)
      let sum = integral[y1 * (w + 1) + x1] - integral[y0 * (w + 1) + x1] - integral[y1 * (w + 1) + x0] + integral[y0 * (w + 1) + x0]
      out[y * w + x] = sum / Double((x1 - x0) * (y1 - y0))
    } }
    return out
  }

  static func erode(_ m: [Bool], _ w: Int, _ h: Int) -> [Bool] {
    var out = m
    for y in 0..<h { for x in 0..<w where m[y * w + x] {
      if x == 0 || y == 0 || x == w - 1 || y == h - 1 || !m[y * w + x - 1] || !m[y * w + x + 1] || !m[(y - 1) * w + x] || !m[(y + 1) * w + x] {
        out[y * w + x] = false
      }
    } }
    return out
  }

  static func dilate(_ m: [Bool], _ w: Int, _ h: Int) -> [Bool] {
    var out = m
    for y in 0..<h { for x in 0..<w where !m[y * w + x] {
      if (x > 0 && m[y * w + x - 1]) || (x < w - 1 && m[y * w + x + 1]) || (y > 0 && m[(y - 1) * w + x]) || (y < h - 1 && m[(y + 1) * w + x]) {
        out[y * w + x] = true
      }
    } }
    return out
  }

  static func open(_ m: [Bool], _ w: Int, _ h: Int, _ n: Int) -> [Bool] {
    var out = m
    for _ in 0..<n { out = erode(out, w, h) }
    for _ in 0..<n { out = dilate(out, w, h) }
    return out
  }

  static func close(_ m: [Bool], _ w: Int, _ h: Int, _ n: Int) -> [Bool] {
    var out = m
    for _ in 0..<n { out = dilate(out, w, h) }
    for _ in 0..<n { out = erode(out, w, h) }
    return out
  }

  /// 4-connected components, as pixel index lists.
  static func components(_ m: [Bool], _ w: Int, _ h: Int) -> [[Int]] {
    var seen = [Bool](repeating: false, count: w * h), out: [[Int]] = []
    for start in 0..<(w * h) where m[start] && !seen[start] {
      var stack = [start], component: [Int] = []
      seen[start] = true
      while let i = stack.popLast() {
        component.append(i)
        let x = i % w, y = i / w
        for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] where nx >= 0 && ny >= 0 && nx < w && ny < h {
          let j = ny * w + nx
          if m[j] && !seen[j] { seen[j] = true; stack.append(j) }
        }
      }
      out.append(component)
    }
    return out
  }

  /// Chamfer (3-4) distance to the nearest pixel outside the mask, in pixels.
  static func distanceTransform(_ m: [Bool], _ w: Int, _ h: Int) -> [Double] {
    let big = Double(3 * (w + h))
    var d = m.map { $0 ? big : 0 }
    func at(_ x: Int, _ y: Int) -> Double { x < 0 || y < 0 || x >= w || y >= h ? 0 : d[y * w + x] }
    for y in 0..<h { for x in 0..<w where m[y * w + x] {
      d[y * w + x] = min(d[y * w + x], at(x - 1, y) + 3, at(x, y - 1) + 3, at(x - 1, y - 1) + 4, at(x + 1, y - 1) + 4)
    } }
    for y in stride(from: h - 1, through: 0, by: -1) { for x in stride(from: w - 1, through: 0, by: -1) where m[y * w + x] {
      d[y * w + x] = min(d[y * w + x], at(x + 1, y) + 3, at(x, y + 1) + 3, at(x + 1, y + 1) + 4, at(x - 1, y + 1) + 4)
    } }
    return d.map { $0 / 3 }
  }

  /// k-means with deterministic farthest-point seeding; k grows while each extra cluster
  /// cuts the within-cluster spread by at least `gain`. Returns a cluster per feature.
  static func clusters(_ x: [SIMD3<Double>], maxK: Int, gain: Double) -> [Int] {
    guard !x.isEmpty else { return [] }
    // Subsample for speed; assign every point to the final centers at the end.
    let step = max(1, x.count / 4000)
    let sample = stride(from: 0, to: x.count, by: step).map { x[$0] }
    func run(_ k: Int) -> (centers: [SIMD3<Double>], sse: Double) {
      let mean = sample.reduce(.zero, +) / Double(sample.count)
      var centers = [sample.max { simd_distance_squared($0, mean) < simd_distance_squared($1, mean) }!]
      while centers.count < k {
        centers.append(sample.max { a, b in
          centers.map { simd_distance_squared(a, $0) }.min()! < centers.map { simd_distance_squared(b, $0) }.min()!
        }!)
      }
      var sse = 0.0
      for _ in 0..<12 {
        var sums = [SIMD3<Double>](repeating: .zero, count: k), counts = [Int](repeating: 0, count: k)
        sse = 0
        for p in sample {
          let (c, d) = nearest(p, centers)
          sums[c] += p; counts[c] += 1; sse += d
        }
        for c in 0..<k where counts[c] > 0 { centers[c] = sums[c] / Double(counts[c]) }
      }
      return (centers, sse)
    }
    var best = run(1)
    if maxK > 1 {
      for k in 2...maxK {
        let next = run(k)
        guard next.sse < (1 - gain) * best.sse else { break }
        best = next
      }
    }
    return x.map { nearest($0, best.centers).0 }
  }

  private static func nearest(_ p: SIMD3<Double>, _ centers: [SIMD3<Double>]) -> (Int, Double) {
    var best = (0, Double.infinity)
    for (i, c) in centers.enumerated() { let d = simd_distance_squared(p, c); if d < best.1 { best = (i, d) } }
    return best
  }

  private static func median(_ v: [Double]) -> Double {
    guard !v.isEmpty else { return 0 }
    let s = v.sorted(); return s[s.count / 2]
  }
}
