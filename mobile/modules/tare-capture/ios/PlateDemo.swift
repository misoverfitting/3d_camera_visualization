import ARKit
import CoreImage
import SceneKit
import simd

// Plate demo: tap the plate, orbit it, and probe spots appear ~1 in inside the rim.
// Keyframes are picked by `OrbitSelector`, the rim comes from `EdgeRimFitter` (image
// edges only, never depth) and spots from `ProbeRing`. LiDAR depth, when present, only
// places spots on the surface and cross-checks the rim.

/// Main-thread demo state; `queue` builds images and runs fits.
final class PlateDemo {
  enum Stage: String { case idle, capturing, done }

  static let fitWidthPx = 480, fitHeightPx = 360

  var enabled = false
  var stage = Stage.idle
  var table: Plane?
  var taps: [SIMD3<Double>] = []
  var selector: OrbitSelector?
  var views: [EdgeView] = []
  var building = 0
  var lastPose: simd_float4x4?
  var lastConsidered: TimeInterval = 0
  var hint: String?
  var fitting = false
  var finalPending = false
  var fittedViews = 0
  var generation = 0
  var fit: EdgeRimFit?
  var fitError: String?
  var spots: [ProbeSpot] = []
  var lidar: RimReference?
  /// Manual nudge added to every spot's height (demo tuning), meters.
  var heightOffsetM = 0.0
  /// Most overhead color photo of the orbit, sent for food identification (FoodID.swift).
  var photo: FoodPhoto?
  var food = FoodState()
  let ciContext = CIContext()
  let node = SCNNode()
  let queue = DispatchQueue(label: "tare.demo", qos: .userInitiated)

  /// Mean of the taps on the plate, on the table plane.
  var seed: SIMD3<Double>? { taps.isEmpty ? nil : taps.reduce(.zero, +) / Double(taps.count) }

  func reset() {
    stage = .idle; table = nil; taps = []; selector = nil; views = []; lastPose = nil; hint = nil
    fit = nil; fitError = nil; spots = []; lidar = nil; fittedViews = 0; finalPending = false; generation += 1
    photo = nil; food = FoodState()
    node.childNodes.forEach { $0.removeFromParentNode() }
  }

  var status: [String: Any] {
    var out: [String: Any] = ["stage": stage.rawValue, "views": views.count, "taps": taps.count,
                              "coverage": selector?.coverage ?? 0, "fitting": fitting, "spots": spots.count]
    if let hint { out["hint"] = hint }
    if let fitError { out["fitError"] = fitError }
    if let rim = fit?.rim {
      out["rim"] = ["diameterMm": 2000 * rim.radiusM, "heightMm": 1000 * rim.heightM, "support": rim.confidence]
    }
    out["heightOffsetMm"] = 1000 * heightOffsetM
    if let lidar { out["lidar"] = ["diameterMm": 2000 * lidar.radiusM, "heightMm": 1000 * lidar.heightM] }
    out["food"] = food.status
    out["hasPhoto"] = photo != nil
    return out
  }

  /// 4x4 box-averaged luma from the camera's Y plane (pixel centers stay consistent with
  /// `CameraIntrinsics.scaled`).
  static func grayImage(_ buffer: CVPixelBuffer) -> GrayImage? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard CVPixelBufferGetPlaneCount(buffer) >= 1, let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return nil }
    let width = CVPixelBufferGetWidthOfPlane(buffer, 0), height = CVPixelBufferGetHeightOfPlane(buffer, 0)
    let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
    guard width == fitWidthPx * 4, height == fitHeightPx * 4 else { return nil }
    let luma = base.assumingMemoryBound(to: UInt8.self)
    var pixels = [Float](repeating: 0, count: fitWidthPx * fitHeightPx)
    for y in 0..<fitHeightPx {
      for x in 0..<fitWidthPx {
        var sum = 0
        for dy in 0..<4 {
          let row = luma + (4 * y + dy) * stride + 4 * x
          sum += Int(row[0]) + Int(row[1]) + Int(row[2]) + Int(row[3])
        }
        pixels[y * fitWidthPx + x] = Float(sum) / 16
      }
    }
    return GrayImage(widthPx: fitWidthPx, heightPx: fitHeightPx, pixels: pixels)
  }
}

extension TareCaptureView {
  func demoTap(_ location: CGPoint) {
    guard demo.stage != .done else { return }
    var hit: ARRaycastResult?
    for target in [ARRaycastQuery.Target.existingPlaneInfinite, .estimatedPlane] {
      if let query = preview.raycastQuery(from: location, allowing: target, alignment: .horizontal),
         let first = preview.session.raycast(query).first { hit = first; break }
    }
    guard let hit, let frame = preview.session.currentFrame, case .normal = frame.camera.trackingState else {
      emit("No table under that tap yet. Move sideways until cyan surfaces appear, then tap the plate."); return
    }
    let position = simd_double4x4(hit.worldTransform).columns.3
    let point = SIMD3(position.x, position.y, position.z)
    if demo.stage == .idle {
      let table = (hit.anchor as? ARPlaneAnchor).map(Plane.init) ?? Plane(pointWorld: point, normalWorld: SIMD3(0, 1, 0))
      demo.table = table
      demo.taps = [point]
      demo.selector = OrbitSelector(targetWorld: point, tablePlane: table)
      demo.stage = .capturing
      emit("Plate selected. Circle it slowly, from high above to low at the side.")
    } else if let seed = demo.seed, simd_distance(seed, point) < 0.15 {
      demo.taps.append(point)
    }
    drawDemo()
  }

  func demoFinish() {
    guard demo.stage == .capturing else { return }
    demo.stage = .done
    runDemoFit(final: true)
  }

  func demoNudge(_ deltaMm: Double) {
    demo.heightOffsetM = min(0.05, max(-0.05, demo.heightOffsetM + deltaMm / 1000))
    drawDemo()
    emit(String(format: "Probe spots moved to %+.0f mm.", 1000 * demo.heightOffsetM))
  }

  func demoReset() {
    demo.reset()
    emit("Tap the middle of the plate to start.")
  }

  func demoConsider(_ frame: ARFrame) {
    guard demo.stage == .capturing, var selector = demo.selector,
          frame.timestamp - demo.lastConsidered >= 0.1 else { return }
    let pose = frame.camera.transform
    defer { demo.lastPose = pose; demo.lastConsidered = frame.timestamp }
    guard let previous = demo.lastPose else { return }
    let dt = max(1e-3, frame.timestamp - demo.lastConsidered)
    let rotation = simd_quatf(simd_float3x3(previous.columns.0.xyz, previous.columns.1.xyz, previous.columns.2.xyz))
      .inverse * simd_quatf(simd_float3x3(pose.columns.0.xyz, pose.columns.1.xyz, pose.columns.2.xyz))
    let angularSpeed = Double(rotation.angle) * 180 / .pi / dt
    let speed = Double(simd_distance(previous.columns.3.xyz, pose.columns.3.xyz)) / dt

    var keyframe = Keyframe(frame: frame, id: demo.views.count + demo.building, includeDepth: false)
    keyframe.intrinsics = keyframe.intrinsics.scaled(toWidthPx: PlateDemo.fitWidthPx, heightPx: PlateDemo.fitHeightPx)
    if let rejection = selector.consider(keyframe, angularSpeedDegS: angularSpeed, speedMS: speed) {
      demo.hint = rejection.rawValue
      if rejection == .full { demoFinish() }
      return
    }
    demo.selector = selector; demo.hint = nil
    // Depth on every other keyframe: enough for spot heights and the LiDAR check.
    if keyframe.id % 2 == 0, let depth = frame.smoothedSceneDepth ?? frame.sceneDepth {
      keyframe.depth = DepthFrame(depth, imageIntrinsics: CameraIntrinsics(frame.camera), worldFromCamera: keyframe.worldFromCamera)
    }
    considerFoodPhoto(frame, pose: keyframe.worldFromCamera)
    demo.building += 1
    let buffer = frame.capturedImage, generation = demo.generation
    demo.queue.async { [weak self] in
      let image = PlateDemo.grayImage(buffer)
      DispatchQueue.main.async {
        guard let self, generation == self.demo.generation else { return }
        self.demo.building -= 1
        guard let image else { return }
        self.demo.views.append(EdgeView(keyframe: keyframe, image: image))
        if selector.coverage >= 0.6 && self.demo.views.count >= 18 { self.demoFinish() } else { self.runDemoFit(final: false) }
      }
    }
  }

  /// Refits every few keyframes so the ring appears and settles during the orbit.
  /// A final fit requested while another runs is started when that one completes.
  func runDemoFit(final requested: Bool) {
    if requested { demo.finalPending = true }
    guard !demo.fitting, let table = demo.table, let seed = demo.seed else { return }
    let views = demo.views, final = demo.finalPending
    guard final || (views.count >= 6 && views.count - demo.fittedViews >= 3) else { return }
    demo.finalPending = false
    demo.fitting = true; demo.fittedViews = views.count
    let generation = demo.generation
    let cloudSnapshot = { [cloud] in cloud.points(minCount: 1) }
    demo.queue.async { [weak self, depthQueue] in
      let result = EdgeRimFitter().fit(views, tablePlane: table, seedWorld: seed)
      let rim = try? result.get().rim
      let lidar = final ? rim.flatMap { rim in
        LidarRimReference().estimate(keyframes: views.map(\.keyframe), tablePlane: table,
                                     approximateCenterWorld: rim.centerWorld, searchRadiusM: rim.radiusM * 1.3 + 0.03)
      } : nil
      depthQueue.async {
        let points = rim == nil ? [] : cloudSnapshot().map { SIMD3<Double>($0) }
        let spots = rim.map { ProbeRing.spots(rim: $0, tablePlane: table, surfacePointsWorld: points) } ?? []
        DispatchQueue.main.async {
          guard let self, generation == self.demo.generation else { return }
          self.demo.fitting = false
          switch result {
          case .success(let fit): self.demo.fit = fit; self.demo.fitError = nil; self.demo.spots = spots
          case .failure(let failure): self.demo.fitError = "\(failure)"
          }
          if final { self.demo.lidar = lidar }
          self.drawDemo()
          self.emit(final ? (rim == nil ? "Could not find the rim. Restart and orbit more slowly." : "Done. Probe spots are on the plate.")
                          : "Keep circling the plate.")
          if self.demo.finalPending { self.runDemoFit(final: true) }
        }
      }
    }
  }

  func demoMessage(tracking: String, planes: Int) -> String {
    switch demo.stage {
    case .idle:
      if tracking != "normal" { return "Move slowly over the table to start tracking." }
      return planes == 0 ? "Look down at the table and move sideways until cyan surfaces appear."
                         : "Tap the middle of the plate to start."
    case .capturing:
      let covered = "\(Int((demo.selector?.coverage ?? 0) * 100))% covered"
      switch demo.hint.flatMap(OrbitSelector.Rejection.init(rawValue:)) {
      case .moving: return "Slow down. \(covered)"
      case .outOfFrame: return "Keep the plate in the middle of the screen. \(covered)"
      case .tooLow: return "Raise the phone a little. \(covered)"
      case .distance: return "Stay 20–60 cm from the plate. \(covered)"
      case .trackingLimited: return "Tracking is limited. Move slowly. \(covered)"
      default: return "Circle the plate: high above, then lower at the sides. \(covered)"
      }
    case .done:
      if demo.fitting || demo.finalPending { return "Finding the rim…" }
      return demo.fit == nil ? "Could not find the rim. Restart and circle more slowly." : "Done. Probe spots are on the plate."
    }
  }

  func drawDemo() {
    demo.node.childNodes.forEach { $0.removeFromParentNode() }
    if demo.fit == nil, let seed = demo.seed {
      let marker = SCNNode(geometry: SCNSphere(radius: 0.004))
      marker.geometry?.materials = [demoMaterial(.systemYellow)]
      marker.simdPosition = SIMD3<Float>(seed)
      demo.node.addChildNode(marker)
    }
    guard let rim = demo.fit?.rim, let table = demo.table else { return }
    let torus = SCNTorus(ringRadius: CGFloat(rim.radiusM), pipeRadius: 0.0012)
    torus.materials = [demoMaterial(demo.stage == .done ? .systemGreen : .white)]
    let ring = SCNNode(geometry: torus)
    ring.simdPosition = SIMD3<Float>(rim.centerWorld)
    ring.simdOrientation = simd_quatf(from: SIMD3(0, 1, 0), to: SIMD3<Float>(table.normalWorld))
    demo.node.addChildNode(ring)
    // Flat discs lying on the surface read as "on the plate"; spheres looked like they floated.
    let up = SIMD3<Float>(table.normalWorld)
    // Once food is labeled, its points replace the generic ring of probe spots.
    if !demo.food.items.isEmpty { drawFood(table: table); return }
    for spot in demo.spots {
      let disc = SCNCylinder(radius: 0.006, height: 0.001)
      disc.materials = [demoMaterial(spot.measuredSurface ? .systemOrange : .systemRed)]
      let node = SCNNode(geometry: disc)
      node.simdPosition = SIMD3<Float>(spot.positionWorld + (demo.heightOffsetM + 0.0005) * table.normalWorld)
      node.simdOrientation = simd_quatf(from: SIMD3(0, 1, 0), to: up)
      demo.node.addChildNode(node)
    }
  }

  func demoMaterial(_ color: UIColor) -> SCNMaterial {
    let material = SCNMaterial()
    material.diffuse.contents = color
    material.lightingModel = .constant
    material.readsFromDepthBuffer = false
    return material
  }
}

extension SIMD4 where Scalar == Float {
  var xyz: SIMD3<Float> { SIMD3(x, y, z) }
}
