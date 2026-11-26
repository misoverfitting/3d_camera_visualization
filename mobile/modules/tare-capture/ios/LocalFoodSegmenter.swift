import CoreML
import Vision

// Learned item finder (ml/foodseg): a Core ML segmentation model that marks which pixels
// of the upright plate crop are food and which food. Used instead of PlateItems' color
// and texture rules when FoodSegmenter.mlmodelc is in the app; items come from
// PlateItems.regions(labels:...), which splits the food into connected pieces of one class.

final class LocalFoodSegmenter {
  static let name = "FoodSegmenter"
  private static var cached: LocalFoodSegmenter??

  /// Nil when the model is not in the app bundle.
  static var shared: LocalFoodSegmenter? {
    if let cached { return cached }
    let made = try? LocalFoodSegmenter()
    cached = .some(made)
    return made
  }

  let model: VNCoreMLModel
  let version: String
  /// Class names by index (0 = background).
  let classes: [String]
  /// Food probability above which a pixel counts as food.
  var threshold: Float = 0.5

  private init() throws {
    let bundle = Bundle.main.url(forResource: "TareCaptureAssets", withExtension: "bundle").flatMap(Bundle.init(url:)) ?? .main
    guard let url = bundle.url(forResource: Self.name, withExtension: "mlmodelc") else { throw RimTapException("No food segmenter in the app.") }
    let config = MLModelConfiguration()
    config.computeUnits = .all
    let ml = try MLModel(contentsOf: url, configuration: config)
    model = try VNCoreMLModel(for: ml)
    let meta = ml.modelDescription.metadata
    version = meta[.versionString] as? String ?? "unknown"
    let user = meta[.creatorDefinedKey] as? [String: String] ?? [:]
    classes = user["classes"].flatMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String] } ?? []
  }

  /// Per-pixel labels on a w x h grid over `image` (0 = not food, else the food class).
  func labels(_ image: CGImage, widthPx w: Int, heightPx h: Int) throws -> [Int] {
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .scaleFill  // Training and eval squash the image too.
    try VNImageRequestHandler(cgImage: image).perform([request])
    let outputs = (request.results as? [VNCoreMLFeatureValueObservation]) ?? []
    guard let food = outputs.first(where: { $0.featureName == "food" })?.featureValue.multiArrayValue,
          let classes = outputs.first(where: { $0.featureName == "classes" })?.featureValue.multiArrayValue else {
      throw RimTapException("The food segmenter returned no masks.")
    }
    let foodGrid = Grid(food), classGrid = Grid(classes)
    var out = [Int](repeating: 0, count: w * h)
    for y in 0..<h {
      for x in 0..<w {
        let u = (Double(x) + 0.5) / Double(w), v = (Double(y) + 0.5) / Double(h)
        if foodGrid.value(u, v) > Double(threshold) { out[y * w + x] = Int(classGrid.value(u, v)) }
      }
    }
    return out
  }

  func className(_ index: Int?) -> String? {
    guard let index, classes.indices.contains(index) else { return nil }
    return classes[index]
  }

  /// Nearest-pixel reads from a (1, H, W) or (H, W) multiarray of any numeric type.
  private struct Grid {
    let array: MLMultiArray
    let height: Int, width: Int, rowStride: Int, colStride: Int

    init(_ array: MLMultiArray) {
      self.array = array
      let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
      height = shape[shape.count - 2]; width = shape[shape.count - 1]
      rowStride = strides[strides.count - 2]; colStride = strides[strides.count - 1]
    }

    func value(_ u: Double, _ v: Double) -> Double {
      let x = min(width - 1, Int(u * Double(width))), y = min(height - 1, Int(v * Double(height)))
      let i = y * rowStride + x * colStride
      switch array.dataType {
      case .float16: return Double(array.dataPointer.assumingMemoryBound(to: Float16.self)[i])
      case .float32: return Double(array.dataPointer.assumingMemoryBound(to: Float32.self)[i])
      case .double: return array.dataPointer.assumingMemoryBound(to: Double.self)[i]
      case .int32: return Double(array.dataPointer.assumingMemoryBound(to: Int32.self)[i])
      default: return array[i].doubleValue
      }
    }
  }
}
