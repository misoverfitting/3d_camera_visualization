import ExpoModulesCore
import ARKit
import CoreImage
import ImageIO
import simd

// Temporary "tap the rim" test. Stands in for the RimDetector (tare_impl.md §6.4) so the
// two-view RimTriangulator can run on real plates: the user taps rim points on two
// still photos and everything downstream is the real MealScan pipeline.
//
// Photos are saved upright (portrait); taps arrive in portrait pixels and are mapped back
// to native sensor pixels, which is what the keyframe intrinsics describe.

struct RimShot {
  var keyframe: Keyframe
  var tablePlane: Plane
  var tableAnchor: UUID?
  var imageURL: URL
}

final class RimTapException: Exception, @unchecked Sendable {
  private let text: String
  init(_ text: String) { self.text = text; super.init() }
  override var reason: String { text }
}

/// Portrait pixel → native sensor pixel. ARKit images are landscape with the phone's
/// bottom edge at +u; the portrait photo is the sensor image rotated 90° clockwise.
func sensorPixel(portrait p: SIMD2<Double>, sensorHeightPx: Int) -> SIMD2<Double> {
  SIMD2(p.y, Double(sensorHeightPx - 1) - p.x)
}

extension TareCaptureView {
  func rimSnapshot(slot: Int) throws -> [String: Any] {
    guard (0...1).contains(slot) else { throw RimTapException("Unknown photo slot \(slot)") }
    guard let frame = preview.session.currentFrame, case .normal = frame.camera.trackingState else {
      throw RimTapException("Tracking is not ready. Move slowly over the table first.")
    }
    guard let (table, anchor) = tablePlane(in: frame) else {
      throw RimTapException("No table detected yet. Move sideways until cyan surfaces appear.")
    }
    let keyframe = Keyframe(frame: frame, id: slot, includeDepth: true)
    let image = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
    guard let color = CGColorSpace(name: CGColorSpace.sRGB),
          let jpeg = CIContext().jpegRepresentation(of: image, colorSpace: color,
            options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.92])
    else { throw RimTapException("Could not encode the photo.") }
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("rim-\(slot)-\(UUID().uuidString).jpg")
    try jpeg.write(to: url, options: .atomic)
    if let old = rimShots[slot] { try? FileManager.default.removeItem(at: old.imageURL) }
    rimShots[slot] = RimShot(keyframe: keyframe, tablePlane: table, tableAnchor: anchor, imageURL: url)

    var result: [String: Any] = [
      "uri": url.absoluteString,
      "width": keyframe.intrinsics.heightPx, "height": keyframe.intrinsics.widthPx,
      "tiltDeg": KeyframeSelector.overheadTiltDeg(keyframe),
      "cameraHeightM": table.signedDistanceM(keyframe.cameraCenterWorld),
      "hasDepth": keyframe.depth != nil,
    ]
    if let other = rimShots[1 - slot] {
      result["baselineM"] = simd_distance(other.keyframe.cameraCenterWorld, keyframe.cameraCenterWorld)
    }
    return result
  }

  func rimReset() {
    rimShots.values.forEach { try? FileManager.default.removeItem(at: $0.imageURL) }
    rimShots.removeAll()
  }

  /// `taps[slot]` holds [x, y] portrait pixels. `rulerDiameterMm` is optional ground truth.
  func rimMeasure(taps: [[[Double]]], rulerDiameterMm: Double?) throws -> [String: Any] {
    let shots = (0...1).compactMap { rimShots[$0] }
    guard !shots.isEmpty else { throw RimTapException("Capture the photos first.") }
    let table = currentTablePlane(for: shots)
    let observations = shots.map { shot in
      let sensorHeight = shot.keyframe.intrinsics.heightPx
      let points = (shot.keyframe.id < taps.count ? taps[shot.keyframe.id] : [])
        .filter { $0.count == 2 }.map { sensorPixel(portrait: SIMD2($0[0], $0[1]), sensorHeightPx: sensorHeight) }
      return RimObservation(keyframe: shot.keyframe, keypointsPx: points)
    }

    var result: [String: Any] = ["cameraHeightsM": shots.map { table.signedDistanceM($0.keyframe.cameraCenterWorld) }]
    if shots.count == 2 {
      result["baselineM"] = simd_distance(shots[0].keyframe.cameraCenterWorld, shots[1].keyframe.cameraCenterWorld)
    }
    var twoView: RimEstimate?
    switch RimTriangulator().triangulate(observations, tablePlane: table) {
    case .success(let rim): twoView = rim; result["twoView"] = describe(rim)
    case .failure(let failure): result["twoViewError"] = "\(failure)"
    }
    // Single-view fits show the height bias: rim assumed on the table vs. at the measured height.
    result["singleView"] = observations.map { observation -> [String: Any] in
      var entry: [String: Any] = ["slot": observation.keyframe.id]
      let fitter = ContourRimFitter()
      if case .success(let rim) = fitter.fit(rimPx: observation.keypointsPx, keyframe: observation.keyframe,
                                             tablePlane: table, rimHeightM: nil) {
        entry["onTable"] = describe(rim)
      }
      if let twoView, case .success(let rim) = fitter.fit(rimPx: observation.keypointsPx, keyframe: observation.keyframe,
                                                          tablePlane: table, rimHeightM: twoView.heightM) {
        entry["atMeasuredHeight"] = describe(rim)
      }
      return entry
    }
    // LiDAR ground truth, seeded at whichever estimate exists.
    let seed = twoView ?? observations.lazy.compactMap { o -> RimEstimate? in
      if case .success(let rim) = ContourRimFitter().fit(rimPx: o.keypointsPx, keyframe: o.keyframe, tablePlane: table, rimHeightM: nil) { return rim }
      return nil
    }.first
    if let seed, shots.contains(where: { $0.keyframe.depth != nil }),
       let reference = LidarRimReference().estimate(keyframes: shots.map(\.keyframe), tablePlane: table,
                                                    approximateCenterWorld: seed.centerWorld,
                                                    searchRadiusM: seed.radiusM * 1.3 + 0.03) {
      var lidar: [String: Any] = ["diameterMm": 2000 * reference.radiusM, "heightMm": 1000 * reference.heightM,
                                  "boundaryRmsMm": 1000 * reference.boundaryRmsM]
      if let twoView {
        let comparison = reference.compare(twoView)
        lidar["twoViewDiameterErrorMm"] = 1000 * comparison.diameterErrorM
        lidar["twoViewDiameterErrorPct"] = 100 * comparison.diameterErrorFraction
        lidar["twoViewHeightErrorMm"] = 1000 * comparison.heightErrorM
      }
      result["lidar"] = lidar
    }
    if let rulerDiameterMm, rulerDiameterMm > 0, let twoView {
      result["rulerErrorMm"] = 2000 * twoView.radiusM - rulerDiameterMm
      result["rulerErrorPct"] = 100 * (2000 * twoView.radiusM - rulerDiameterMm) / rulerDiameterMm
    }
    if let folder = try? saveRimTest(shots: shots, observations: observations, table: table,
                                     rulerDiameterMm: rulerDiameterMm, result: result) {
      result["savedTo"] = folder.path
    }
    return result
  }

  private func describe(_ rim: RimEstimate) -> [String: Any] {
    var out: [String: Any] = ["diameterMm": 2000 * rim.radiusM, "heightMm": 1000 * rim.heightM,
                              "rmsPx": rim.rmsResidualPx, "confidence": rim.confidence, "views": rim.viewCount]
    if let sigma = rim.radiusUncertaintyM { out["diameterSigmaMm"] = 2000 * sigma }
    if let sigma = rim.heightUncertaintyM { out["heightSigmaMm"] = 1000 * sigma }
    return out
  }

  /// Nearest horizontal plane under the screen center (the table, not the floor below it).
  private func tablePlane(in frame: ARFrame) -> (Plane, UUID?)? {
    let center = CGPoint(x: preview.bounds.midX, y: preview.bounds.midY)
    if let query = preview.raycastQuery(from: center, allowing: .existingPlaneInfinite, alignment: .horizontal),
       let anchor = preview.session.raycast(query).first?.anchor as? ARPlaneAnchor {
      return (Plane(anchor), anchor.identifier)
    }
    let cameraY = frame.camera.transform.columns.3.y
    let below = frame.anchors.compactMap { $0 as? ARPlaneAnchor }
      .filter { $0.alignment == .horizontal && $0.transform.columns.3.y < cameraY }
    guard let highest = below.max(by: { $0.transform.columns.3.y < $1.transform.columns.3.y }) else { return nil }
    return (Plane(highest), highest.identifier)
  }

  /// ARKit keeps refining plane anchors, so prefer the anchor's latest geometry.
  private func currentTablePlane(for shots: [RimShot]) -> Plane {
    let anchors = preview.session.currentFrame?.anchors.compactMap { $0 as? ARPlaneAnchor } ?? []
    for shot in shots.reversed() {
      if let id = shot.tableAnchor, let anchor = anchors.first(where: { $0.identifier == id }) { return Plane(anchor) }
    }
    return shots.last!.tablePlane
  }

  /// Keeps each measurement (photos, poses, taps, results) as a future eval fixture.
  private func saveRimTest(shots: [RimShot], observations: [RimObservation], table: Plane,
                           rulerDiameterMm: Double?, result: [String: Any]) throws -> URL {
    let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
    let folder = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("RimTests/" + stamp, isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    var views: [[String: Any]] = []
    for (shot, observation) in zip(shots, observations) {
      let name = "view\(shot.keyframe.id).jpg"
      try? FileManager.default.copyItem(at: shot.imageURL, to: folder.appendingPathComponent(name))
      let k = shot.keyframe.intrinsics, m = shot.keyframe.worldFromCamera
      views.append(["image": name, "image_orientation": "portrait (sensor rotated 90° clockwise)",
        "intrinsics_sensor_px": ["fx": k.fxPx, "fy": k.fyPx, "cx": k.cxPx, "cy": k.cyPx, "width": k.widthPx, "height": k.heightPx],
        "world_from_camera_column_major": (0..<4).flatMap { c in (0..<4).map { r in m[c][r] } },
        "rim_taps_sensor_px": observation.keypointsPx.map { [$0.x, $0.y] },
        "has_depth": shot.keyframe.depth != nil])
    }
    let session: [String: Any] = ["schema": "rim_tap_test_v1", "camera_axes": "+X right, +Y up, -Z forward",
      "table_plane": ["point": [table.pointWorld.x, table.pointWorld.y, table.pointWorld.z],
                      "normal": [table.normalWorld.x, table.normalWorld.y, table.normalWorld.z]],
      "ruler_diameter_mm": rulerDiameterMm ?? NSNull(), "views": views, "result": result]
    // NaN or infinity would raise an Objective-C exception inside JSONSerialization.
    guard JSONSerialization.isValidJSONObject(session) else { throw RimTapException("Result is not valid JSON.") }
    let data = try JSONSerialization.data(withJSONObject: session, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: folder.appendingPathComponent("rim.json"), options: .atomic)
    return folder
  }
}
