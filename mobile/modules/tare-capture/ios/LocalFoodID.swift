import CoreImage
import CoreML
import Vision
import simd

// On-device food identification, no network: `PlateItems` splits the plate interior into
// items (color/texture against the bare plate), and the bundled Core ML classifier
// (ml/foodclass: MobileNetV3 fine-tuned on Food-101 + FoodSeg103) names each item from a
// padded crop with everything else dimmed, like the training crops.

struct LocalFoodResult {
  var label: String
  var category: String
  var confidence: Double
  var topK: [(String, Double)]
  /// Normalized points in the upright crop.
  var points: [SIMD2<Double>]
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
  }

  func classify(_ image: CGImage) throws -> [(String, Double)] {
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .scaleFill  // Training squashes the crop to 224x224 too.
    try VNImageRequestHandler(cgImage: image).perform([request])
    let results = (request.results as? [VNClassificationObservation]) ?? []
    return results.prefix(3).map { ($0.identifier, Double($0.confidence)) }
  }

  /// `interior` and `ring` are rim outlines in the image's normalized coordinates: food
  /// is searched inside `interior`; between `ring` and `interior` is taken as bare plate.
  func identify(_ image: CGImage, interior: [SIMD2<Double>], ring: [SIMD2<Double>], context: CIContext) throws -> [LocalFoodResult] {
    let analysisSide = 256.0
    let scale = min(1, analysisSide / Double(max(image.width, image.height)))
    let w = max(16, Int((Double(image.width) * scale).rounded())), h = max(16, Int((Double(image.height) * scale).rounded()))
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let drawn = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return [] }
    drawn.interpolationQuality = .medium
    drawn.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    let inside = PlateItems.rasterize(interior, widthPx: w, heightPx: h)
    let bare = PlateItems.rasterize(ring, widthPx: w, heightPx: h)
    let regions = PlateItems.segment(RGBImage(widthPx: w, heightPx: h, rgba: rgba), interior: inside,
                                     ring: zip(inside, bare).map { $0 && !$1 })

    let full = CIImage(cgImage: image)
    let results = try regions.compactMap { region -> LocalFoodResult? in
      guard let crop = itemCrop(full, region: region, context: context) else { return nil }
      let top = try classify(crop)
      guard let best = top.first else { return nil }
      return LocalFoodResult(label: best.0, category: categories[best.0] ?? "other", confidence: best.1, topK: top, points: region.points)
    }
    return Self.mergeSameLabel(results)
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
}
