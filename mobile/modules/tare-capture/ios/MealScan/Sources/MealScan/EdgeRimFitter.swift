import Foundation
import simd

/// Grayscale image, row-major, any consistent intensity scale.
public struct GrayImage: Sendable {
  public var widthPx: Int
  public var heightPx: Int
  public var pixels: [Float]

  public init(widthPx: Int, heightPx: Int, pixels: [Float]) {
    precondition(pixels.count == widthPx * heightPx)
    self.widthPx = widthPx; self.heightPx = heightPx; self.pixels = pixels
  }
}

/// One orbit keyframe for edge fitting. The keyframe's intrinsics must describe `image`.
public struct EdgeView: Sendable {
  public var keyframe: Keyframe
  public var image: GrayImage

  public init(keyframe: Keyframe, image: GrayImage) {
    precondition(keyframe.intrinsics.widthPx == image.widthPx && keyframe.intrinsics.heightPx == image.heightPx)
    self.keyframe = keyframe; self.image = image
  }
}

public struct EdgeRimFit: Sendable {
  public var rim: RimEstimate
  /// Ridge score of the coarse vote (dimensionless, for diagnostics).
  public var voteScore: Double
  /// Mean normalized edge response along the projected rim, 0...1.
  public var edgeSupport: Double
}

/// Automatic rim fitting from image edges seen across an orbit (demo stand-in for the
/// learned RimDetector, tare_impl.md §6.4). Never reads depth.
///
/// 1. Vote: for each candidate rim height, every strong edge pixel's ray is cut by the
///    plane at that height and splatted into a 2D histogram. At the true height the rim
///    edges from all views land on one thin circle; elsewhere they smear.
/// 2. Pick: ring "ridge" scores (on the ring minus beside it) over center, radius and
///    height. The rim is the outermost strong ring, so a plate's inner well edge or round
///    food does not win.
/// 3. Refine: maximize the gradient response across the edge of the projected circle in
///    every view (Nelder-Mead over center, radius, height).
public struct EdgeRimFitter {
  public var config: PipelineConfig.EdgeRim

  public init(config: PipelineConfig.EdgeRim = .init()) { self.config = config }

  public func fit(_ views: [EdgeView], tablePlane: Plane, seedWorld: SIMD3<Double>) -> Result<EdgeRimFit, RimFailure> {
    guard views.count >= config.minViews else { return .failure(.insufficientViews) }
    let frame = PlaneFrame(tablePlane)
    let seed = frame.coordinates(seedWorld)
    let prepared = views.map { PreparedView($0, config: config, frame: frame) }
    guard let coarse = vote(prepared, frame: frame, seed: seed) else { return .failure(.insufficientKeypoints) }
    let refined = refine(prepared, frame: frame, from: coarse.params)
    let p = refined.params
    let rim = RimEstimate(
      centerWorld: frame.world(SIMD2(p[0], p[1]), heightM: p[3]), radiusM: p[2], heightM: p[3],
      radiusUncertaintyM: nil, heightUncertaintyM: nil, rmsResidualPx: 0,
      method: .orbitEdges, viewCount: views.count, confidence: refined.support)
    return .success(EdgeRimFit(rim: rim, voteScore: coarse.score, edgeSupport: refined.support))
  }

  // MARK: Vote

  private func vote(_ views: [PreparedView], frame: PlaneFrame, seed: SIMD2<Double>) -> (params: [Double], score: Double)? {
    let bin = config.binM
    let ridgeOffset = 2  // Radius steps between the ring and its "beside" rings.
    let extent = config.maxRadiusM + config.centerSearchM + Double(ridgeOffset + 1) * config.coarseRadiusStepM
    let size = Int((2 * extent / bin).rounded(.up)) + 1
    let radii = stride(from: config.minRadiusM - Double(ridgeOffset) * config.coarseRadiusStepM,
                       through: config.maxRadiusM + Double(ridgeOffset) * config.coarseRadiusStepM + 1e-9,
                       by: config.coarseRadiusStepM).map { $0 }
    let steps = Int((config.centerSearchM / config.coarseCenterStepM).rounded())
    let offsets = (-steps...steps).map { Double($0) * config.coarseCenterStepM }
    let ring = (0..<config.ringSamples).map { i -> SIMD2<Double> in
      let a = 2 * .pi * Double(i) / Double(config.ringSamples)
      return SIMD2(cos(a), sin(a))
    }
    // Per radius index: best (score, cx, cz, h).
    var best = [(score: Double, params: [Double])](repeating: (-.infinity, []), count: radii.count)
    var histogram = [Float](repeating: 0, count: size * size)
    for heightM in stride(from: 0.0, through: config.maxHeightM + 1e-9, by: config.heightStepM) {
      for i in histogram.indices { histogram[i] = 0 }
      for view in views {
        let weight = Float(1.0 / Double(max(view.edges.count, 1)))
        for edge in view.edges {
          let rise = heightM - edge.originHeightM
          guard edge.directionHeight < -1e-6, rise < 0 else { continue }
          let t = rise / edge.directionHeight
          let gx = (edge.origin2D.x + t * edge.direction2D.x - seed.x + extent) / bin
          let gy = (edge.origin2D.y + t * edge.direction2D.y - seed.y + extent) / bin
          let x0 = Int(gx.rounded(.down)), y0 = Int(gy.rounded(.down))
          guard x0 >= 0, y0 >= 0, x0 + 1 < size, y0 + 1 < size else { continue }
          let fx = Float(gx - Double(x0)), fy = Float(gy - Double(y0))
          let base = y0 * size + x0
          histogram[base] += weight * (1 - fx) * (1 - fy)
          histogram[base + 1] += weight * fx * (1 - fy)
          histogram[base + size] += weight * (1 - fx) * fy
          histogram[base + size + 1] += weight * fx * fy
        }
      }
      smooth(&histogram, size: size)
      histogram.withUnsafeBufferPointer { h in
        func sample(_ x: Double, _ y: Double) -> Double {
          let gx = (x + extent) / bin, gy = (y + extent) / bin
          let x0 = Int(gx), y0 = Int(gy)
          guard x0 >= 0, y0 >= 0, x0 + 1 < size, y0 + 1 < size else { return 0 }
          let fx = gx - Double(x0), fy = gy - Double(y0), b = y0 * size + x0
          return Double(h[b]) * (1 - fx) * (1 - fy) + Double(h[b + 1]) * fx * (1 - fy)
            + Double(h[b + size]) * (1 - fx) * fy + Double(h[b + size + 1]) * fx * fy
        }
        var means = [Double](repeating: 0, count: radii.count)
        for dz in offsets {
          for dx in offsets {
            for (r, radiusM) in radii.enumerated() {
              var sum = 0.0
              for u in ring { sum += sample(dx + radiusM * u.x, dz + radiusM * u.y) }
              means[r] = sum / Double(ring.count)
            }
            for r in ridgeOffset..<(radii.count - ridgeOffset) {
              let score = means[r] - 0.5 * (means[r - ridgeOffset] + means[r + ridgeOffset])
              if score > best[r].score { best[r] = (score, [seed.x + dx, seed.y + dz, radii[r], heightM]) }
            }
          }
        }
      }
    }
    let candidates = (ridgeOffset..<(radii.count - ridgeOffset)).filter { best[$0].score > 0 }
    guard let top = candidates.map({ best[$0].score }).max() else { return nil }
    // Outermost local maximum that is nearly as strong as the best ring.
    let peaks = candidates.filter { r in
      best[r].score >= config.outermostScoreFraction * top
        && best[r].score >= best[max(r - 1, 0)].score && best[r].score >= best[min(r + 1, radii.count - 1)].score
    }
    guard let rim = peaks.max() else { return nil }
    return (best[rim].params, best[rim].score)
  }

  /// Separable [1 2 1]/4 blur so nearby splats reinforce each other.
  private func smooth(_ h: inout [Float], size: Int) {
    var tmp = h
    for y in 0..<size { for x in 1..<(size - 1) { tmp[y * size + x] = 0.25 * h[y * size + x - 1] + 0.5 * h[y * size + x] + 0.25 * h[y * size + x + 1] } }
    for y in 1..<(size - 1) { for x in 0..<size { h[y * size + x] = 0.25 * tmp[(y - 1) * size + x] + 0.5 * tmp[y * size + x] + 0.25 * tmp[(y + 1) * size + x] } }
  }

  // MARK: Refine

  private func refine(_ views: [PreparedView], frame: PlaneFrame, from initial: [Double]) -> (params: [Double], support: Double) {
    let window = config.refineWindowM
    let samples = 2 * config.ringSamples
    let angles = (0..<samples).map { 2 * .pi * Double($0) / Double(samples) }
    func support(_ p: [Double]) -> Double {
      let center = SIMD2(p[0], p[1]), radiusM = p[2], heightM = p[3]
      var total = 0.0, valid = 0
      for view in views {
        for a in angles {
          let u = SIMD2(cos(a), sin(a))
          let point = frame.world(center + radiusM * u, heightM: heightM)
          let ahead = frame.world(center + radiusM * u + 0.002 * SIMD2(-u.y, u.x), heightM: heightM)
          guard let p0 = view.keyframe.projectPx(point), let p1 = view.keyframe.projectPx(ahead),
                let g = view.gradient(at: p0) else { continue }
          let tangent = p1 - p0, length = simd_length(tangent)
          guard length > 1e-9 else { continue }
          let normal = SIMD2(-tangent.y, tangent.x) / length
          total += min(1, abs(simd_dot(g, normal)) / view.gradientScale)
          valid += 1
        }
      }
      let expected = views.count * samples
      guard valid > 0 else { return 0 }
      // Mostly out-of-frame circles cannot win by scoring only their visible part.
      return total / Double(valid) * min(1, Double(valid) / (0.3 * Double(expected)))
    }
    let result = Numerics.nelderMead(initial: initial, steps: [0.003, 0.003, 0.003, 0.003], maxEvaluations: 300) { p in
      guard abs(p[0] - initial[0]) <= window, abs(p[1] - initial[1]) <= window,
            abs(p[2] - initial[2]) <= window, abs(p[3] - initial[3]) <= window, p[2] > 0, p[3] >= 0 else { return 1 }
      return -support(p)
    }
    let refinedSupport = -result.value
    let initialSupport = support(initial)
    return refinedSupport >= initialSupport ? (result.params, refinedSupport) : (initial, initialSupport)
  }
}

/// Table-plane coordinates computed once (`Plane.coordinates2D` inverts a matrix per call).
struct PlaneFrame {
  let origin: SIMD3<Double>, x: SIMD3<Double>, z: SIMD3<Double>, normal: SIMD3<Double>

  init(_ plane: Plane) {
    let m = plane.worldFromPlane
    x = SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z)
    normal = SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z)
    z = SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z)
    origin = plane.pointWorld
  }

  func coordinates(_ p: SIMD3<Double>) -> SIMD2<Double> { SIMD2(simd_dot(p - origin, x), simd_dot(p - origin, z)) }
  func height(_ p: SIMD3<Double>) -> Double { simd_dot(p - origin, normal) }
  func world(_ c: SIMD2<Double>, heightM: Double) -> SIMD3<Double> { origin + c.x * x + c.y * z + heightM * normal }
}

/// Blurred gradients plus the strongest thinned edge pixels as rays in the table frame.
struct PreparedView {
  struct Edge {
    var origin2D: SIMD2<Double>, originHeightM: Double
    var direction2D: SIMD2<Double>, directionHeight: Double
  }

  let keyframe: Keyframe
  let width: Int, height: Int
  let gx: [Float], gy: [Float]
  /// Robust "strong edge" gradient magnitude used to normalize responses.
  let gradientScale: Double
  var edges: [Edge] = []

  init(_ view: EdgeView, config: PipelineConfig.EdgeRim, frame: PlaneFrame) {
    keyframe = view.keyframe
    width = view.image.widthPx; height = view.image.heightPx
    let blurred = PreparedView.blur(view.image.pixels, width: width, height: height, sigma: config.blurSigmaPx)
    var gx = [Float](repeating: 0, count: width * height), gy = gx, magnitude = gx
    for y in 1..<(height - 1) {
      for x in 1..<(width - 1) {
        let i = y * width + x
        let a = blurred[i - width - 1], b = blurred[i - width], c = blurred[i - width + 1]
        let d = blurred[i - 1], f = blurred[i + 1]
        let g = blurred[i + width - 1], h = blurred[i + width], k = blurred[i + width + 1]
        gx[i] = ((c + 2 * f + k) - (a + 2 * d + g)) / 8
        gy[i] = ((g + 2 * h + k) - (a + 2 * b + c)) / 8
        magnitude[i] = (gx[i] * gx[i] + gy[i] * gy[i]).squareRoot()
      }
    }
    self.gx = gx; self.gy = gy
    // Threshold from a strided sample of magnitudes.
    let sample = Swift.stride(from: 0, to: magnitude.count, by: 7).map { Double(magnitude[$0]) }.sorted()
    let threshold = sample[min(sample.count - 1, Int(Double(sample.count) * (1 - config.edgeFraction)))]
    gradientScale = max(1e-6, sample[min(sample.count - 1, Int(Double(sample.count) * 0.98))])

    let center = keyframe.cameraCenterWorld
    let origin2D = frame.coordinates(center), originHeightM = frame.height(center)
    var edges: [Edge] = []
    for y in 2..<(height - 2) {
      for x in 2..<(width - 2) {
        let i = y * width + x, m = magnitude[i]
        guard Double(m) >= threshold, m > 0 else { continue }
        // Non-maximum suppression along the dominant gradient axis keeps edges one pixel thin.
        let horizontal = abs(gx[i]) >= abs(gy[i])
        let step = horizontal ? 1 : width
        guard m >= magnitude[i - step], m >= magnitude[i + step] else { continue }
        let direction = keyframe.rayWorld(throughPx: SIMD2(Double(x), Double(y))).directionWorld
        edges.append(Edge(origin2D: origin2D, originHeightM: originHeightM,
                          direction2D: SIMD2(simd_dot(direction, frame.x), simd_dot(direction, frame.z)),
                          directionHeight: simd_dot(direction, frame.normal)))
      }
    }
    self.edges = edges
  }

  func gradient(at p: SIMD2<Double>) -> SIMD2<Double>? {
    let x0 = Int(p.x.rounded(.down)), y0 = Int(p.y.rounded(.down))
    guard x0 >= 1, y0 >= 1, x0 + 2 < width, y0 + 2 < height else { return nil }
    let fx = Float(p.x - Double(x0)), fy = Float(p.y - Double(y0)), i = y0 * width + x0
    func lerp(_ v: [Float]) -> Double {
      Double(v[i] * (1 - fx) * (1 - fy) + v[i + 1] * fx * (1 - fy) + v[i + width] * (1 - fx) * fy + v[i + width + 1] * fx * fy)
    }
    return SIMD2(lerp(gx), lerp(gy))
  }

  static func blur(_ pixels: [Float], width: Int, height: Int, sigma: Double) -> [Float] {
    guard sigma > 0 else { return pixels }
    let radius = Int((3 * sigma).rounded(.up))
    var kernel = (-radius...radius).map { Float(exp(-Double($0 * $0) / (2 * sigma * sigma))) }
    let total = kernel.reduce(0, +)
    kernel = kernel.map { $0 / total }
    var tmp = pixels, out = pixels
    for y in 0..<height {
      for x in 0..<width {
        var s: Float = 0
        for k in -radius...radius { s += kernel[k + radius] * pixels[y * width + min(width - 1, max(0, x + k))] }
        tmp[y * width + x] = s
      }
    }
    for y in 0..<height {
      for x in 0..<width {
        var s: Float = 0
        for k in -radius...radius { s += kernel[k + radius] * tmp[min(height - 1, max(0, y + k)) * width + x] }
        out[y * width + x] = s
      }
    }
    return out
  }
}
