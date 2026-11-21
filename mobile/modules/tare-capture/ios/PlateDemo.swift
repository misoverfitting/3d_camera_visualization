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
  var tableMarkerWorld: SIMD3<Double>?
  var tableOffsetM = 0.0
  var taps: [SIMD3<Double>] = []
  var selector: OrbitSelector?
  var views: [EdgeView] = []
  var building = 0
  var finishRequested = false
  var imageBuildFailures = 0
  var rejectionCounts: [String: Int] = [:]
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
  /// The prototype probe pen visiting each point (ProbeSim.swift).
  let probe = ProbeSim()
  let ciContext = CIContext()
  let node = SCNNode()
  let queue = DispatchQueue(label: "tare.demo.images", qos: .userInitiated)
  let fitQueue = DispatchQueue(label: "tare.demo.fit", qos: .userInitiated)

  /// Mean of the taps on the plate, on the table plane.
  var seed: SIMD3<Double>? { taps.isEmpty ? nil : taps.reduce(.zero, +) / Double(taps.count) }

  func reset() {
    stage = .idle; table = nil; tableMarkerWorld = nil; tableOffsetM = 0; taps = []; selector = nil; views = []; lastPose = nil; hint = nil
    fit = nil; fitError = nil; spots = []; lidar = nil; fittedViews = 0; finalPending = false; generation += 1
    photo = nil; food = FoodState(); probe.stop()
    building = 0; finishRequested = false; imageBuildFailures = 0; rejectionCounts = [:]
    lastConsidered = 0; fitting = false
    node.childNodes.forEach { $0.removeFromParentNode() }
  }

  var status: [String: Any] {
    var out: [String: Any] = ["stage": stage.rawValue, "views": views.count, "taps": taps.count,
                              "coverage": selector?.coverage ?? 0, "fitting": fitting, "spots": spots.count]
    out["tableSet"] = table != nil
    out["tableOffsetMm"] = tableOffsetM * 1000
    out["setupStep"] = table == nil ? "table" : (stage == .idle ? "plate" : "capture")
    out["coveredCellsByBand"] = selector?.coveredCellsByBand ?? []
    out["rejections"] = rejectionCounts
    out["imageBuildFailures"] = imageBuildFailures
    out["building"] = building
    out["fitReady"] = fitError == nil && (fit?.edgeSupport ?? 0) >= (selector?.config.minRimSupport ?? 0.3)
    out["coverageCells"] = selector?.coverageCells ?? []
    if let hint { out["hint"] = hint }
    if let fitError { out["fitError"] = fitError }
    if let rim = fit?.rim {
      out["rim"] = ["diameterMm": 2000 * rim.radiusM, "heightMm": 1000 * rim.heightM, "support": rim.confidence]
    }
    out["heightOffsetMm"] = 1000 * heightOffsetM
    if let lidar { out["lidar"] = ["diameterMm": 2000 * lidar.radiusM, "heightMm": 1000 * lidar.heightM] }
    out["food"] = food.status
    out["hasPhoto"] = photo?.jpeg.isEmpty == false
    out["probe"] = probe.status
    return out
  }

  /// Downsample the sensor luma plane without assuming a specific ARKit video format.
  static func grayImage(_ buffer: CVPixelBuffer) -> GrayImage? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard CVPixelBufferGetPlaneCount(buffer) >= 1, let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return nil }
    let width = CVPixelBufferGetWidthOfPlane(buffer, 0), height = CVPixelBufferGetHeightOfPlane(buffer, 0)
    let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
    return LumaResampler.downsample(UnsafeBufferPointer(start: base.assumingMemoryBound(to: UInt8.self), count: height * stride),
                                    width: width, height: height, bytesPerRow: stride, targetWidth: fitWidthPx)
  }

}

extension TareCaptureView {
  func demoTap(_ location: CGPoint) {
    guard demo.stage != .done else { return }
    guard let frame = preview.session.currentFrame, case .normal = frame.camera.trackingState else {
      emit("Move slowly over the bare table until tracking is ready."); return
    }
    if demo.table == nil {
      // Never infer the table from a tap on elevated food or extrapolate an unseen plane.
      guard let query = preview.raycastQuery(from: location, allowing: .existingPlaneGeometry, alignment: .horizontal),
            let hit = preview.session.raycast(query).first,
            let anchor = hit.anchor as? ARPlaneAnchor else {
        demo.hint = "tableNotObserved"
        emit(demoMessage(tracking: "normal", planes: 0)); return
      }
      let table = Plane(anchor)
      guard simd_dot(table.normalWorld, SIMD3(0, 1, 0)) > 0.98 else {
        demo.hint = "tableNotFlat"
        emit(demoMessage(tracking: "normal", planes: 0)); return
      }
      let position = simd_double4x4(hit.worldTransform).columns.3
      let hitWorld = SIMD3(position.x, position.y, position.z)
      let camera = simd_double4x4(frame.camera.transform).columns.3
      let distance = simd_distance(hitWorld, SIMD3(camera.x, camera.y, camera.z))
      guard distance >= 0.15, distance <= 1.0 else {
        demo.hint = "tableTooFar"
        emit(demoMessage(tracking: "normal", planes: 0)); return
      }
      demo.hint = nil
      demo.table = table
      demo.tableMarkerWorld = hitWorld - table.signedDistanceM(hitWorld) * table.normalWorld
      emit(demoMessage(tracking: "normal", planes: 1))
    } else if demo.stage == .idle, let table = demo.table,
              let query = preview.raycastQuery(from: location, allowing: .existingPlaneGeometry, alignment: .horizontal) {
      // Use only the selected table plane; do not raycast again against food/floor candidates.
      let ray = Ray(originWorld: SIMD3<Double>(query.origin), directionWorld: SIMD3<Double>(query.direction))
      guard let seed = SupportPlacement.target(ray: ray, tablePlane: table) else {
        demo.hint = "plateOutOfRange"
        emit(demoMessage(tracking: "normal", planes: 1)); return
      }
      demo.hint = nil
      demo.taps = [seed]
      demo.selector = OrbitSelector(targetWorld: seed, tablePlane: table)
      demo.stage = .capturing
      emit("Plate center set on your table. Move around it slowly, including lower views.")
    } else if demo.stage == .capturing {
      demo.hint = "recenter"
      emit("Restart to select the table and plate center again.")
    }
    drawDemo()
  }

  /// Calibrates the entire support plane before any scan geometry is captured.
  func demoAdjustTable(_ deltaMm: Double) {
    guard demo.stage == .idle, let table = demo.table, let marker = demo.tableMarkerWorld,
          deltaMm.isFinite else { return }
    let nextOffset = min(0.1, max(-0.1, demo.tableOffsetM + deltaMm / 1000))
    let delta = nextOffset - demo.tableOffsetM
    demo.table = table.offset(byM: delta)
    demo.tableMarkerWorld = marker + delta * table.normalWorld
    demo.tableOffsetM = nextOffset
    demo.hint = nil
    drawDemo()
    emit(demoMessage(tracking: "normal", planes: 1))
  }

  func demoFinish() {
    guard demo.stage == .capturing else { return }
    // Include the last reserved view in the final fit, even if Finish was tapped while building.
    guard demo.building == 0 else { demo.finishRequested = true; return }
    demo.finishRequested = false
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
    emit("First tap bare table beside the plate to set its height.")
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

    // Select the food image independently: useful overhead shots may share an occupied cell.
    if angularSpeed <= selector.config.maxAngularSpeedDegS, speed <= selector.config.maxSpeedMS,
       case .normal = frame.camera.trackingState { considerFoodPhoto(frame, pose: simd_double4x4(pose)) }
    // At most one reservation. Pose sampling above still runs while the worker is busy.
    guard demo.building == 0, !demo.finishRequested else { return }
    var keyframe = Keyframe(frame: frame, id: demo.views.count, includeDepth: false)
    if let rejection = selector.consider(keyframe, angularSpeedDegS: angularSpeed, speedMS: speed) {
      demo.hint = rejection.rawValue
      demo.rejectionCounts[rejection.rawValue, default: 0] += 1
      if rejection == .full { demoFinish() }
      return
    }
    demo.hint = nil
    // Depth on every other keyframe: enough for spot heights and the LiDAR check.
    if keyframe.id % 2 == 0, let depth = frame.smoothedSceneDepth ?? frame.sceneDepth {
      keyframe.depth = DepthFrame(depth, imageIntrinsics: CameraIntrinsics(frame.camera), worldFromCamera: keyframe.worldFromCamera)
    }
    demo.building += 1
    let buffer = frame.capturedImage, generation = demo.generation
    demo.queue.async { [weak self] in
      let image = PlateDemo.grayImage(buffer)
      DispatchQueue.main.async {
        guard let self, generation == self.demo.generation else { return }
        self.demo.building -= 1
        guard let image else {
          self.demo.imageBuildFailures += 1
          self.demo.hint = "imageBuildFailed"
          if self.demo.finishRequested { self.demoFinish() }
          return
        }
        // Commit the tentative selector only after its image exists.
        var fittedKeyframe = keyframe
        fittedKeyframe.intrinsics = keyframe.intrinsics.scaled(toWidthPx: image.widthPx, heightPx: image.heightPx)
        let sharpness = image.sharpness
        fittedKeyframe.sharpness = sharpness
        if let index = selector.replacementIndex {
          // Never replace a sharper image with a blurrier one just to refresh a cell.
          guard sharpness > (self.demo.views[index].keyframe.sharpness ?? 0) * 1.1 else {
            if self.demo.finishRequested { self.demoFinish() }
            return
          }
          self.demo.views[index] = EdgeView(keyframe: fittedKeyframe, image: image)
          self.demo.fittedViews = max(0, self.demo.views.count - 3)
        } else {
          self.demo.views.append(EdgeView(keyframe: fittedKeyframe, image: image))
        }
        self.demo.selector = selector
        if self.demo.finishRequested || selector.isFull { self.demoFinish() }
        else { self.runDemoFit(final: false) }
      }
    }
  }

  /// Refits every few keyframes so the ring appears and settles during the orbit.
  /// A final fit requested while another runs is started when that one completes.
  func runDemoFit(final requested: Bool) {
    if requested { demo.finalPending = true }
    guard !demo.fitting, let table = demo.table, let seed = demo.seed else { return }
    let views = demo.views, final = demo.finalPending
    let selectorSnapshot = demo.selector
    let minSupport = selectorSnapshot?.config.minRimSupport ?? 0.3
    guard final || (views.count >= 6 && views.count - demo.fittedViews >= 3) else { return }
    demo.finalPending = false
    demo.fitting = true; demo.fittedViews = views.count
    let generation = demo.generation
    // Unverified multi-view feature points can float above the dish. Use only
    // repeatedly observed LiDAR points for surface heights; non-LiDAR uses the plane.
    let useMeasuredSurface = usesMeasuredSurfaceDepth
    let cloudSnapshot = { [cloud] in cloud.points(minCount: 3) }
    demo.fitQueue.async { [weak self, depthQueue] in
      let result = EdgeRimFitter().fit(views, tablePlane: table, seedWorld: seed)
      let rim = try? result.get().rim
      let supported = ((try? result.get().edgeSupport) ?? 0) >= minSupport
      let lidar = final ? rim.flatMap { rim in
        LidarRimReference().estimate(keyframes: views.map(\.keyframe), tablePlane: table,
                                     approximateCenterWorld: rim.centerWorld, searchRadiusM: rim.radiusM * 1.3 + 0.03)
      } : nil
      depthQueue.async {
        let points = rim == nil || !useMeasuredSurface ? [] : cloudSnapshot().map { SIMD3<Double>($0) }
        let spots = rim.map { ProbeRing.spots(rim: $0, tablePlane: table, surfacePointsWorld: points) } ?? []
        DispatchQueue.main.async {
          guard let self, generation == self.demo.generation else { return }
          self.demo.fitting = false
          switch result {
          case .success(let fit):
            self.demo.fit = fit
            self.demo.fitError = fit.edgeSupport < (self.demo.selector?.config.minRimSupport ?? 0.3) ? "The rim outline is uncertain" : nil
            self.demo.spots = spots
          case .failure(let failure): self.demo.fitError = "\(failure)"
          }
          if final { self.demo.lidar = lidar }
          self.drawDemo()
          self.emit(final ? (supported ? "Check that the outline follows your plate before identifying food." : "Plate outline uncertain. Restart, tap the center and capture new sides.")
                          : "Keep circling the plate.")
          if self.demo.finalPending { self.runDemoFit(final: true) }
          else if !final, self.demo.stage == .capturing,
                  let selector = selectorSnapshot, selector.hasEnoughViews,
                  case .success(let fit) = result, fit.edgeSupport >= selector.config.minRimSupport {
            self.demoFinish()
          } else if !final, self.demo.stage == .capturing {
            // Views may have arrived while fitting. Do not wait for another accepted cell to refit.
            self.runDemoFit(final: false)
          }
        }
      }
    }
  }

  func demoMessage(tracking: String, planes: Int) -> String {
    switch demo.stage {
    case .idle:
      if tracking != "normal" { return "Move slowly over the table to start tracking." }
      switch demo.hint {
      case "tableNotObserved": return "No tracked table here yet. Move sideways over textured bare table beside the plate, then tap it."
      case "tableNotFlat": return "Choose a flat patch of bare table beside the plate."
      case "tableTooFar": return "Move 15-100 cm from the tabletop and tap it again; avoid the floor."
      case "plateOutOfRange": return "Keep the plate centered and 15-100 cm away, then tap its center."
      default: break
      }
      if demo.table != nil { return "Look from an angle: the yellow marker should sit on the table. Adjust Surface down/up if needed, then tap the plate center." }
      return planes == 0 ? "Move slowly sideways over textured bare table beside the plate until a surface appears, then tap it."
                         : "Tap bare table beside the plate, not the food, to set the surface height."
    case .capturing:
      let covered = "\(Int((demo.selector?.coverage ?? 0) * 100))% covered"
      if demo.hint == "recenter" { return "Restart to choose a new plate center. \(covered)" }
      if demo.hint == "imageBuildFailed" { return "Could not read the camera image. Try again or restart the scan. \(covered)" }
      switch demo.hint.flatMap(OrbitSelector.Rejection.init(rawValue:)) {
      case .moving: return "Slow down. \(covered)"
      case .outOfFrame: return "Keep the plate in the middle of the screen. \(covered)"
      case .tooLow: return "Raise the phone a little. \(covered)"
      case .distance: return "Stay 15\u{2013}100 cm from the plate. \(covered)"
      case .trackingLimited: return "Tracking is limited. Move slowly. \(covered)"
      default:
        if demo.fitError != nil { return "Rim outline uncertain. Move around new sides with the plate centered; restart surface setup if markers float. \(covered)" }
        let bands = demo.selector?.coveredCellsByBand ?? [0, 0]
        if let low = bands.first, let high = bands.last, low < high {
          return "Lower the phone below 45 degrees and move around the plate, keeping it centered. \(covered)"
        }
        return "Move around the plate from above to cover new sides; turning the phone in place adds no coverage. \(covered)"
      }
    case .done:
      if demo.fitting || demo.finalPending { return "Finding the rim…" }
      let probe = demo.probe
      if probe.running { return "Probing \(probe.current ?? "the plate") (\(min(probe.done + 1, probe.total)) of \(probe.total))…" }
      if probe.total > 0 && probe.done == probe.total { return "Probed \(probe.total) spot\(probe.total == 1 ? "" : "s")." }
      return demo.fitError != nil || demo.fit == nil ? "Plate outline uncertain. Restart and align the table surface before choosing the plate center." : "Check the entire outer rim. If the outline is off, restart surface setup; otherwise choose Use this plate & identify food."
    }
  }

  func drawDemo() {
    demo.node.childNodes.forEach { $0.removeFromParentNode() }
    if demo.fit == nil, let table = demo.table, let point = demo.seed ?? demo.tableMarkerWorld {
      // A thin, flat target makes height alignment visible; a sphere itself protrudes.
      let marker = SCNNode(geometry: SCNCylinder(radius: 0.012, height: 0.0003))
      marker.geometry?.materials = [demoMaterial(.systemYellow)]
      marker.simdPosition = SIMD3<Float>(point + 0.00015 * table.normalWorld)
      marker.simdOrientation = simd_quatf(from: SIMD3(0, 1, 0), to: SIMD3<Float>(table.normalWorld))
      demo.node.addChildNode(marker)
    }
    guard let rim = demo.fit?.rim, let table = demo.table else { return }
    let torus = SCNTorus(ringRadius: CGFloat(rim.radiusM), pipeRadius: 0.0012)
    let supported = (demo.fit?.edgeSupport ?? 0) >= (demo.selector?.config.minRimSupport ?? 0.3)
    torus.materials = [demoMaterial(supported ? UIColor(red: 1, green: 0.416, blue: 0, alpha: 1) : .systemYellow)]
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
