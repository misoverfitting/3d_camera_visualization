import CoreImage
import CoreML
import Vision
import simd

// On-device food identification, no network: `PlateItems` splits the plate interior into
// items (color/texture against the bare plate), and the bundled Core ML classifier
// (ml/foodclass: MobileNetV3 fine-tuned on Food-101 + FoodSeg103 + Nutrition5k) names each
// item from a padded crop with everything else dimmed, like the training crops. The whole
// plate is classified once too, restricted to the Food-101 dish classes, for a dish name.

struct LocalFoodResult {
  var label: String
  var category: String
  var confidence: Double
  var topK: [(String, Double)]
  /// Normalized points in the upright crop.
  var points: [SIMD2<Double>]
}

/// Everything one identification saw, kept for the field log (FoodScanLog.swift).
struct LocalFoodScan {
  struct Region {
    var box: SIMD4<Double>
    var areaFraction: Double
    var topK: [(String, Double)]
    var points: [SIMD2<Double>]
    var crop: CGImage
  }
  /// Regions merged by label: what the app shows.
  var items: [LocalFoodResult] = []
  /// Every region the segmenter proposed, before merging, in segmenter order.
  var regions: [Region] = []
  /// Best Food-101 dish classes for the whole plate.
  var dish: [(String, Double)] = []
  /// Best classes of any kind for the whole plate.
  var wholePlate: [(String, Double)] = []
  var plateInput: CGImage?
  var overlay: CGImage?
}

final class LocalFoodClassifier {
  static let name = "FoodClassifier"
  private static var cached: LocalFoodClassifier??

  /// Nil when the model is not in the app bundle.
  static var shared: LocalFoodClassifier? {
    if let cached { return cached }
    let made = try? LocalFoodClassifier()
    cached = .some(made)
    return made
  }

  let model: VNCoreMLModel
  let categories: [String: String]
  let version: String
  /// Food-101 classes: whole dishes, as opposed to the ingredients FoodSeg103 adds.
  let dishes: Set<String>

  private init() throws {
    let bundle = Bundle.main.url(forResource: "TareCaptureAssets", withExtension: "bundle").flatMap(Bundle.init(url:)) ?? .main
    guard let url = bundle.url(forResource: Self.name, withExtension: "mlmodelc") else { throw RimTapException("No on-device food model in the app.") }
    let config = MLModelConfiguration()
    config.computeUnits = .all
    let ml = try MLModel(contentsOf: url, configuration: config)
    model = try VNCoreMLModel(for: ml)
    let meta = ml.modelDescription.metadata
    version = meta[.versionString] as? String ?? "unknown"
    let user = meta[.creatorDefinedKey] as? [String: String] ?? [:]
    categories = user["categories"].flatMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: String] } ?? [:]
    let labels = bundle.url(forResource: "\(Self.name).labels", withExtension: "json")
      .flatMap { try? Data(contentsOf: $0) }
      .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
    dishes = Set(labels?["dishes"] as? [String] ?? [])
  }

  /// All classes, best first.
  func classifyAll(_ image: CGImage) throws -> [(String, Double)] {
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .scaleFill  // Training squashes the crop to 224x224 too.
    try VNImageRequestHandler(cgImage: image).perform([request])
    let results = (request.results as? [VNClassificationObservation]) ?? []
    return results.map { ($0.identifier, Double($0.confidence)) }
  }

  func classify(_ image: CGImage) throws -> [(String, Double)] {
    Array(try classifyAll(image).prefix(3))
  }

  /// `interior` and `ring` are rim outlines in the image's normalized coordinates: food
  /// is searched inside `interior`; between `ring` and `interior` is taken as bare plate.
  func identify(_ image: CGImage, interior: [SIMD2<Double>], ring: [SIMD2<Double>], context: CIContext) throws -> LocalFoodScan {
    let analysisSide = 256.0
    let scale = min(1, analysisSide / Double(max(image.width, image.height)))
    let w = max(16, Int((Double(image.width) * scale).rounded())), h = max(16, Int((Double(image.height) * scale).rounded()))
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let drawn = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return LocalFoodScan() }
    drawn.interpolationQuality = .medium
    drawn.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    let inside = PlateItems.rasterize(interior, widthPx: w, heightPx: h)
    let bare = zip(inside, PlateItems.rasterize(ring, widthPx: w, heightPx: h)).map { $0 && !$1 }
    let found = PlateItems.segment(RGBImage(widthPx: w, heightPx: h, rgba: rgba), interior: inside, ring: bare)

    let full = CIImage(cgImage: image)
    let insideCount = max(1, inside.filter { $0 }.count)
    var scan = LocalFoodScan()
    var results: [LocalFoodResult] = []
    for region in found {
      guard let crop = itemCrop(full, region: region, context: context) else { continue }
      let top = try classify(crop)
      scan.regions.append(.init(box: region.box, areaFraction: Double(region.areaPx) / Double(insideCount),
                                topK: top, points: region.points, crop: crop))
      guard let best = top.first else { continue }
      results.append(LocalFoodResult(label: best.0, category: categories[best.0] ?? "other", confidence: best.1, topK: top, points: region.points))
    }
    scan.items = Self.mergeSameLabel(results)

    // Whole plate: the interior with the rest dimmed, as one "item".
    if let plate = Self.region(of: inside, widthPx: w, heightPx: h), let input = itemCrop(full, region: plate, context: context) {
      let all = try classifyAll(input)
      scan.plateInput = input
      scan.wholePlate = Array(all.prefix(3))
      scan.dish = Array(all.filter { dishes.contains($0.0) }.prefix(3))
    }
    scan.overlay = overlay(full, regions: found, inside: inside, bare: bare, widthPx: w, heightPx: h, context: context)
    return scan
  }

  /// Color clustering often splits one food (light and shaded rice, say) into several
  /// regions; regions that get the same label are one item, keeping 3 well-spread points.
  static func mergeSameLabel(_ results: [LocalFoodResult]) -> [LocalFoodResult] {
    var merged: [LocalFoodResult] = []
    for result in results {
      if let i = merged.firstIndex(where: { $0.label == result.label }) {
        merged[i].points += result.points
        merged[i].confidence = max(merged[i].confidence, result.confidence)
      } else {
        merged.append(result)
      }
    }
    return merged.map { item in
      var item = item
      guard item.points.count > 3 else { return item }
      var chosen = [item.points[0]]  // Points of the largest region come first.
      while chosen.count < 3 {
        let next = item.points.max { a, b in
          chosen.map { simd_distance(a, $0) }.min()! < chosen.map { simd_distance(b, $0) }.min()!
        }!
        chosen.append(next)
      }
      item.points = chosen
      return item
    }
  }

  /// A mask as a region, with its bounding box; nil when empty.
  static func region(of mask: [Bool], widthPx w: Int, heightPx h: Int) -> ItemRegion? {
    var x0 = w, y0 = h, x1 = -1, y1 = -1, area = 0
    for y in 0..<h { for x in 0..<w where mask[y * w + x] {
      x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y); area += 1
    } }
    guard area > 0 else { return nil }
    let box = SIMD4(Double(x0) / Double(w), Double(y0) / Double(h), Double(x1 + 1) / Double(w), Double(y1 + 1) / Double(h))
    return ItemRegion(widthPx: w, heightPx: h, mask: mask, areaPx: area, box: box, points: [])
  }

  /// Box padded 12.5% per side, background dimmed to 35% (autolabel.verify.crop).
  private func itemCrop(_ full: CIImage, region: ItemRegion, context: CIContext) -> CGImage? {
    let W = full.extent.width, H = full.extent.height
    let bw = (region.box.z - region.box.x) * W, bh = (region.box.w - region.box.y) * H
    let x0 = max(0, region.box.x * W - 0.125 * bw), x1 = min(W, region.box.z * W + 0.125 * bw)
    let y0 = max(0, region.box.y * H - 0.125 * bh), y1 = min(H, region.box.w * H + 0.125 * bh)
    guard x1 - x0 >= 8, y1 - y0 >= 8 else { return nil }
    // Mask as a gray image (Core Image's origin is bottom-left, so flip rows).
    var gray = [UInt8](repeating: 0, count: region.widthPx * region.heightPx)
    for y in 0..<region.heightPx { for x in 0..<region.widthPx where region.mask[y * region.widthPx + x] {
      gray[(region.heightPx - 1 - y) * region.widthPx + x] = 255
    } }
    let maskImage = CIImage(bitmapData: Data(gray), bytesPerRow: region.widthPx,
                            size: CGSize(width: region.widthPx, height: region.heightPx), format: .L8, colorSpace: nil)
      .transformed(by: CGAffineTransform(scaleX: W / CGFloat(region.widthPx), y: H / CGFloat(region.heightPx)))
    let dim = full.applyingFilter("CIColorMatrix", parameters: [
      "inputRVector": CIVector(x: 0.35, y: 0, z: 0, w: 0), "inputGVector": CIVector(x: 0, y: 0.35, z: 0, w: 0),
      "inputBVector": CIVector(x: 0, y: 0, z: 0.35, w: 0)])
    let blended = full.applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: dim, kCIInputMaskImageKey: maskImage])
    // Box is in top-left coordinates; Core Image's y runs up.
    let rect = CGRect(x: x0, y: H - y1, width: x1 - x0, height: y1 - y0)
    return context.createCGImage(blended, from: rect)
  }

  /// The plate photo with each region tinted in palette order (region i = palette i), the
  /// bare-plate sample ring tinted blue and everything outside the interior darkened.
  private func overlay(_ full: CIImage, regions: [ItemRegion], inside: [Bool], bare: [Bool],
                       widthPx w: Int, heightPx h: Int, context: CIContext) -> CGImage? {
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    func paint(_ i: Int, _ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat) {
      let o = ((h - 1 - i / w) * w + i % w) * 4  // Flip rows for Core Image.
      rgba[o] = UInt8(r * a * 255); rgba[o + 1] = UInt8(g * a * 255); rgba[o + 2] = UInt8(b * a * 255); rgba[o + 3] = UInt8(a * 255)
    }
    for i in 0..<(w * h) {
      if bare[i] { paint(i, 0.2, 0.4, 1, 0.3) } else if !inside[i] { paint(i, 0, 0, 0, 0.55) }
    }
    for (index, region) in regions.enumerated() {
      var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
      FoodState.palette[index % FoodState.palette.count].0.getRed(&r, green: &g, blue: &b, alpha: &a)
      for i in 0..<(w * h) where region.mask[i] { paint(i, r, g, b, 0.5) }
    }
    guard let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
    let tint = CIImage(bitmapData: Data(rgba), bytesPerRow: w * 4, size: CGSize(width: w, height: h), format: .RGBA8, colorSpace: space)
      .samplingNearest()
      .transformed(by: CGAffineTransform(scaleX: full.extent.width / CGFloat(w), y: full.extent.height / CGFloat(h)))
    return context.createCGImage(tint.composited(over: full), from: full.extent)
  }
}
