import ARKit
import CoreImage
import ImageIO
import SceneKit
import simd

// Food identification for the plate demo: the most overhead orbit photo, cropped to
// the plate, goes to the backend (server/app/food_vision.py), which asks Claude for
// each item's label and 2-3 points inside it. Points are lifted onto the plate with
// LiDAR depth when available (else the plate surface plane) and drawn in the item's color.

struct FoodPhoto {
  var keyframe: Keyframe  // Full-resolution intrinsics, with depth when the session has it.
  var jpeg: Data
  var tiltDeg: Double
}

struct FoodItem {
  var label: String
  var category: String
  var confidence: Double
  var colorIndex: Int
  var points: [FoodPoint]
}

struct FoodState {
  enum Phase: String { case idle, working, done, failed }
  var phase = Phase.idle
  var error: String?
  var items: [FoodItem] = []

  static let palette: [(UIColor, String)] = [
    "#E4572E", "#17BEBB", "#FFC914", "#76B041", "#B07CFF", "#FF6F91", "#2E86AB", "#F18F01",
  ].map { hex in
    let v = Int(hex.dropFirst(), radix: 16)!
    return (UIColor(red: CGFloat((v >> 16) & 255) / 255, green: CGFloat((v >> 8) & 255) / 255,
                    blue: CGFloat(v & 255) / 255, alpha: 1), hex)
  }

  var status: [String: Any] {
    var out: [String: Any] = ["phase": phase.rawValue]
    if let error { out["error"] = error }
    out["items"] = items.map { item -> [String: Any] in
      ["label": item.label, "category": item.category, "confidence": item.confidence,
       "color": Self.palette[item.colorIndex % Self.palette.count].1, "points": item.points.count]
    }
    return out
  }
}

extension TareCaptureView {
  /// Keeps the most overhead accepted frame (tilt under 45°) as the food photo.
  func considerFoodPhoto(_ frame: ARFrame, pose: simd_double4x4) {
    let forward = -SIMD3(pose.columns.2.x, pose.columns.2.y, pose.columns.2.z)
    let tilt = acos(min(1, max(-1, simd_dot(simd_normalize(forward), SIMD3(0, -1, 0))))) * 180 / .pi
    guard tilt < min(45, demo.photo?.tiltDeg ?? 45) else { return }
    let keyframe = Keyframe(frame: frame, id: -1, includeDepth: true)
    let buffer = frame.capturedImage, generation = demo.generation, context = demo.ciContext
    demo.photo = FoodPhoto(keyframe: keyframe, jpeg: Data(), tiltDeg: tilt)  // Claim the slot; JPEG follows.
    demo.queue.async { [weak self] in
      guard let color = CGColorSpace(name: CGColorSpace.sRGB),
            let jpeg = context.jpegRepresentation(of: CIImage(cvPixelBuffer: buffer), colorSpace: color,
              options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.9])
      else { return }
      DispatchQueue.main.async {
        guard let self, generation == self.demo.generation, self.demo.photo?.tiltDeg == tilt else { return }
        self.demo.photo?.jpeg = jpeg
      }
    }
  }

  func demoIdentify(_ completion: @escaping (Result<[String: Any], Error>) -> Void) {
    guard demo.stage == .done, let rim = demo.fit?.rim, let table = demo.table else {
      return completion(.failure(RimTapException("Finish the plate scan first.")))
    }
    guard let photo = demo.photo, !photo.jpeg.isEmpty else {
      return completion(.failure(RimTapException("No overhead photo yet. Restart and include a view from above.")))
    }
    guard let base = Bundle.main.object(forInfoDictionaryKey: "TareApiURL") as? String,
          let url = URL(string: base.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/api/food/identify") else {
      return completion(.failure(RimTapException("No backend configured. Build with TARE_API_URL set.")))
    }
    guard demo.food.phase != .working else { return completion(.failure(RimTapException("Already identifying."))) }
    demo.food = FoodState(phase: .working)
    emit("Identifying the food…")
    let token = Bundle.main.object(forInfoDictionaryKey: "TareApiToken") as? String
    let generation = demo.generation, context = demo.ciContext
    let surfaceHeightM = Numerics.percentile(demo.spots.map(\.surfaceHeightM), 0.5) ?? PipelineConfig.Probe().fallbackSurfaceHeightM

    func finish(_ result: Result<[FoodItem], Error>) {
      DispatchQueue.main.async { [weak self] in
        guard let self, generation == self.demo.generation else { return }
        switch result {
        case .success(let items):
          self.demo.food = FoodState(phase: .done, items: items)
          self.emit(items.isEmpty ? "No food found on the plate." : "Found \(items.count) item\(items.count == 1 ? "" : "s").")
        case .failure(let error):
          self.demo.food = FoodState(phase: .failed, error: "\(error)")
          self.emit("Food identification failed: \(error)")
        }
        self.drawDemo()
        switch result {
        case .success: completion(.success(self.demo.food.status))
        case .failure(let error): completion(.failure(error))
        }
      }
    }

    demo.queue.async {
      guard let crop = ImageCrop.around(rim, tablePlane: table, in: photo.keyframe, margin: 1.15, rotatedClockwise: true),
            let upload = Self.uprightCropJPEG(photo.jpeg, crop: crop, maxSidePx: 1092, context: context) else {
        return finish(.failure(RimTapException("The plate is not inside the overhead photo.")))
      }
      var request = URLRequest(url: url, timeoutInterval: 90)
      request.httpMethod = "POST"
      let boundary = "tare-\(UUID().uuidString)"
      request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
      if let token { request.setValue(token, forHTTPHeaderField: "X-Tare-Token") }
      var body = Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"image\"; filename=\"plate.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".utf8)
      body.append(upload)
      body.append(Data("\r\n--\(boundary)--\r\n".utf8))
      URLSession.shared.uploadTask(with: request, from: body) { data, response, error in
        if let error { return finish(.failure(RimTapException("Could not reach the server: \(error.localizedDescription)"))) }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard let data, let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
          return finish(.failure(RimTapException("Server returned HTTP \(status) with no JSON.")))
        }
        guard status == 200, let items = json["items"] as? [[String: Any]] else {
          return finish(.failure(RimTapException("Server error \(status): \(json["detail"] as? String ?? "unknown")")))
        }
        let lifted = items.enumerated().compactMap { index, item -> FoodItem? in
          let points = (item["points"] as? [[Double]] ?? []).compactMap { p -> FoodPoint? in
            guard p.count == 2 else { return nil }
            return FoodPointLifter.lift(crop.sensorPixel(normalized: SIMD2(p[0], p[1])), keyframe: photo.keyframe,
                                        rim: rim, tablePlane: table, surfaceHeightM: surfaceHeightM)
          }
          guard !points.isEmpty else { return nil }
          return FoodItem(label: item["label"] as? String ?? "food", category: item["category"] as? String ?? "other",
                          confidence: item["confidence"] as? Double ?? 0, colorIndex: index, points: points)
        }
        finish(.success(lifted))
      }.resume()
    }
  }

  /// Crops the sensor-orientation JPEG, rotates it upright and scales it for upload.
  static func uprightCropJPEG(_ jpeg: Data, crop: ImageCrop, maxSidePx: Double, context: CIContext) -> Data? {
    guard let source = CGImageSourceCreateWithData(jpeg as CFData, nil),
          let full = CGImageSourceCreateImageAtIndex(source, 0, nil),
          let cropped = full.cropping(to: CGRect(x: crop.x0Px, y: crop.y0Px, width: crop.widthPx, height: crop.heightPx)),
          let color = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
    var image = CIImage(cgImage: cropped)
    if crop.rotatedClockwise { image = image.oriented(.right) }
    let scale = min(1, maxSidePx / max(image.extent.width, image.extent.height))
    image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    return context.jpegRepresentation(of: image, colorSpace: color,
      options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.88])
  }

  /// Colored discs on each item's points and a floating label above them.
  func drawFood(table: Plane) {
    let up = SIMD3<Float>(table.normalWorld)
    for item in demo.food.items {
      let color = FoodState.palette[item.colorIndex % FoodState.palette.count].0
      for point in item.points {
        let disc = SCNCylinder(radius: 0.006, height: 0.001)
        disc.materials = [demoMaterial(color)]
        let node = SCNNode(geometry: disc)
        node.simdPosition = SIMD3<Float>(point.positionWorld + (demo.heightOffsetM + 0.0005) * table.normalWorld)
        node.simdOrientation = simd_quatf(from: SIMD3(0, 1, 0), to: up)
        demo.node.addChildNode(node)
      }
      let centroid = item.points.map(\.positionWorld).reduce(.zero, +) / Double(item.points.count)
      let text = SCNText(string: item.label, extrusionDepth: 0)
      text.font = UIFont.systemFont(ofSize: 10, weight: .bold)
      text.flatness = 0.2
      text.materials = [demoMaterial(color)]
      let label = SCNNode(geometry: text)
      let (minBound, maxBound) = label.boundingBox
      label.pivot = SCNMatrix4MakeTranslation((minBound.x + maxBound.x) / 2, minBound.y, 0)
      label.scale = SCNVector3(0.0015, 0.0015, 0.0015)  // 10 pt glyphs ≈ 1.5 cm tall.
      label.simdPosition = SIMD3<Float>(centroid + (demo.heightOffsetM + 0.03) * table.normalWorld)
      let billboard = SCNBillboardConstraint()
      billboard.freeAxes = .Y
      label.constraints = [billboard]
      demo.node.addChildNode(label)
    }
  }
}
