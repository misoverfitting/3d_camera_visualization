import ExpoModulesCore
import ARKit
import SceneKit
import AVFoundation
import CoreImage
import ImageIO
import ZIPFoundation

public class TareCaptureModule: Module {
  public func definition() -> ModuleDefinition {
    Name("TareCapture")
    AsyncFunction("latestRecording") { () -> String? in
      let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("TareScans", isDirectory: true)
      let scans = (try? FileManager.default.contentsOfDirectory(at: root,
        includingPropertiesForKeys: [.creationDateKey])) ?? []
      return scans.filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("manifest.json").path) }
        .sorted { ((try? $0.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast)
          > ((try? $1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast) }.first?.path
    }
    AsyncFunction("shareRecording") { (path: String) in
      let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("TareScans", isDirectory: true).standardizedFileURL
      let source = URL(fileURLWithPath: path).standardizedFileURL
      guard source.deletingLastPathComponent() == root,
            FileManager.default.fileExists(atPath: source.appendingPathComponent("manifest.json").path) else {
        throw CaptureError.invalidRecording
      }
      let zip = FileManager.default.temporaryDirectory.appendingPathComponent(source.lastPathComponent + ".zip")
      if FileManager.default.fileExists(atPath: zip.path) { try FileManager.default.removeItem(at: zip) }
      try FileManager.default.zipItem(at: source, to: zip)
      DispatchQueue.main.async {
        guard var presenter = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene })
          .flatMap({ $0.windows }).first(where: { $0.isKeyWindow })?.rootViewController else { return }
        while let next = presenter.presentedViewController { presenter = next }
        let sheet = UIActivityViewController(activityItems: [zip], applicationActivities: nil)
        sheet.popoverPresentationController?.sourceView = presenter.view
        sheet.completionWithItemsHandler = { _, _, _, _ in try? FileManager.default.removeItem(at: zip) }
        presenter.present(sheet, animated: true)
      }
    }
    View(TareCaptureView.self) {
      Events("onStatus")
      Prop("recording") { (view: TareCaptureView, value: Bool) in view.setRecording(value) }
      Prop("regionSize") { (view: TareCaptureView, value: Double) in view.setSize(value) }
      Prop("captureWithoutBox") { (view: TareCaptureView, value: Bool) in view.captureWithoutBox = value }
      Prop("resetToken") { (view: TareCaptureView, value: Int) in view.resetTracking(value) }
    }
  }
}

enum CaptureError: Error { case invalidRecording, imageEncoding }

final class TareCaptureView: ExpoView, ARSessionDelegate, ARSCNViewDelegate {
  let onStatus = EventDispatcher()
  private let preview = ARSCNView(frame: .zero)
  private let coaching = ARCoachingOverlayView(frame: .zero)
  private let writer = DispatchQueue(label: "tare.recording.writer", qos: .utility)
  private let imageContext = CIContext()
  private var notification: NSObjectProtocol?
  private var foregroundNotification: NSObjectProtocol?
  private var recording = false
  private var finishing = false
  private var busy = false
  private var region: simd_float4x4?
  private var size: Float = 1
  private var box: SCNNode?
  var captureWithoutBox = false
  private var surfaces: [UUID: SCNNode] = [:]
  private var trackingDescription = "initializing"
  private var horizontalPlanes = 0
  private var verticalPlanes = 0
  private var featurePoints = 0
  private var resetToken = 0
  private var placementHint: String?
  private var hintUntil: TimeInterval = 0
  private var folder: URL?
  private var startTime: TimeInterval = 0
  private var lastSample: TimeInterval = -.infinity
  private var lastStatus: TimeInterval = 0
  private var count = 0
  private var dropped = 0
  private var trackingSkipped = 0
  // Accessed only on writer queue, and reset only after that queue drains.
  private var frames: [[String: Any]] = []
  private var manifest: [String: Any] = [:]

  required init(appContext: AppContext? = nil) {
    super.init(appContext: appContext)
    preview.scene = SCNScene()
    addSubview(preview)
    preview.session.delegate = self
    preview.session.delegateQueue = .main
    preview.delegate = self
    preview.isPlaying = true
    preview.rendersContinuously = true
    preview.automaticallyUpdatesLighting = true
    preview.debugOptions = [.showFeaturePoints]
    preview.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(selectRegion(_:))))
    coaching.session = preview.session
    coaching.goal = .horizontalPlane
    // Drive coaching from the same live anchor snapshot as the surface counter.
    // Keep taps available for estimated-plane raycasts while coaching is visible.
    coaching.activatesAutomatically = false
    coaching.isUserInteractionEnabled = false
    addSubview(coaching)
    notification = NotificationCenter.default.addObserver(forName: UIApplication.willResignActiveNotification,
      object: nil, queue: .main) { [weak self] _ in
        self?.finish(reason: "App interrupted; recording saved")
        self?.preview.session.pause()
      }
    foregroundNotification = NotificationCenter.default.addObserver(forName: UIApplication.didBecomeActiveNotification,
      object: nil, queue: .main) { [weak self] _ in
        guard let self, self.window != nil, AVCaptureDevice.authorizationStatus(for: .video) == .authorized else { return }
        self.region = nil; self.box?.removeFromParentNode()
        self.runSession(); self.emit("Place a new scan box after resuming.")
      }
  }
  deinit {
    if let notification { NotificationCenter.default.removeObserver(notification) }
    if let foregroundNotification { NotificationCenter.default.removeObserver(foregroundNotification) }
    preview.session.pause()
  }
  override func layoutSubviews() {
    super.layoutSubviews(); preview.frame = bounds; coaching.frame = bounds
  }
  override func didMoveToWindow() {
    super.didMoveToWindow()
    guard window != nil else { finish(reason: "Camera closed; recording saved"); preview.session.pause(); return }
    guard ARWorldTrackingConfiguration.isSupported else { emit("AR world tracking is unavailable on this device"); return }
    AVCaptureDevice.requestAccess(for: .video) { [weak self] allowed in
      DispatchQueue.main.async {
        guard let self, self.window != nil else { return }
        guard allowed else { self.emit("Camera access denied. Enable it in Settings."); return }
        self.runSession()
      }
    }
  }
  private func runSession(reset: Bool = false) {
    let config = ARWorldTrackingConfiguration()
    config.planeDetection = [.horizontal, .vertical]
    preview.session.run(config, options: reset ? [.resetTracking, .removeExistingAnchors] : [])
  }
  func resetTracking(_ token: Int) {
    guard token != resetToken else { return }
    resetToken = token
    guard !recording, !finishing, window != nil,
      AVCaptureDevice.authorizationStatus(for: .video) == .authorized else { return }
    region = nil; box?.removeFromParentNode(); box = nil
    surfaces.values.forEach { $0.removeFromParentNode() }; surfaces.removeAll()
    horizontalPlanes = 0; verticalPlanes = 0; featurePoints = 0; trackingDescription = "initializing"
    placementHint = nil; hintUntil = 0; lastStatus = 0
    runSession(reset: true)
    emit("Tracking restarted. Point down at a textured table and move slowly sideways.")
  }
  func setSize(_ value: Double) {
    guard !recording, !finishing, value.isFinite else { return }
    size = Float(min(3, max(0.25, value))); drawBox()
  }
  @objc private func selectRegion(_ tap: UITapGestureRecognizer) {
    guard !recording, !finishing else { return }
    guard let frame = preview.session.currentFrame, case .normal = frame.camera.trackingState else {
      showPlacementHint("Tracking is not ready. Move slowly sideways over a well-lit, textured table.")
      return
    }
    var result: ARRaycastResult?
    for target in [ARRaycastQuery.Target.existingPlaneGeometry, .estimatedPlane] {
      if let query = preview.raycastQuery(from: tap.location(in: preview), allowing: target, alignment: .horizontal),
         let hit = preview.session.raycast(query).first { result = hit; break }
    }
    guard let hit = result else {
      showPlacementHint("No horizontal surface at that tap. Look downward, move sideways, then tap a cyan surface. You can also skip the box.")
      return
    }
    // World-aligned box starts at the selected support surface and extends upward.
    var transform = matrix_identity_float4x4
    transform.columns.3 = hit.worldTransform.columns.3
    region = transform
    drawBox(); showPlacementHint("Scan box placed. Adjust size, then start.")
  }
  private func showPlacementHint(_ message: String) {
    placementHint = message
    hintUntil = (preview.session.currentFrame?.timestamp ?? 0) + 6
    emit(message)
  }
  // SceneKit supplies the anchor node and maintains its world transform. Serialize
  // our overlay bookkeeping with the ARSession delegate and React props on main.
  func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
    scheduleSurfaceUpdate(anchor, node: node)
  }
  func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
    scheduleSurfaceUpdate(anchor, node: node)
  }
  private func scheduleSurfaceUpdate(_ anchor: ARAnchor, node: SCNNode) {
    guard let plane = anchor as? ARPlaneAnchor, plane.alignment == .horizontal else { return }
    DispatchQueue.main.async { [weak self] in
      guard let self,
        let current = self.preview.session.currentFrame?.anchors.first(where: { $0.identifier == plane.identifier }) as? ARPlaneAnchor,
        self.preview.node(for: current) === node else { return }
      self.updateSurface(current, anchorNode: node)
    }
  }
  private func surfaceMaterial(alpha: CGFloat) -> SCNMaterial {
    let material = SCNMaterial()
    material.diffuse.contents = UIColor.cyan.withAlphaComponent(alpha)
    material.lightingModel = .constant
    material.isDoubleSided = true
    material.blendMode = .alpha
    material.readsFromDepthBuffer = false
    material.writesToDepthBuffer = false
    return material
  }
  private func updateSurface(_ plane: ARPlaneAnchor, anchorNode: SCNNode) {
    guard let device = preview.device else { return }
    let overlay: SCNNode
    let geometry: ARSCNPlaneGeometry
    if let existing = surfaces[plane.identifier], let mesh = existing.geometry as? ARSCNPlaneGeometry {
      overlay = existing; geometry = mesh
    } else {
      guard let mesh = ARSCNPlaneGeometry(device: device) else { return }
      geometry = mesh
      geometry.materials = [surfaceMaterial(alpha: 0.5)]
      overlay = SCNNode(geometry: geometry)
      overlay.name = "tare.detected-plane"
      overlay.renderingOrder = 10
      // ARSCNPlaneGeometry already uses the anchor's x/z coordinates. Applying
      // plane.center or rotating it again would displace/rotate the actual mesh.
      overlay.position.y = 0.003
      surfaces[plane.identifier] = overlay
    }
    geometry.update(from: plane.geometry)
    if overlay.parent !== anchorNode {
      overlay.removeFromParentNode(); anchorNode.addChildNode(overlay)
    }
    overlay.childNodes.forEach { $0.removeFromParentNode() }
    let vertices = plane.geometry.boundaryVertices.map { SCNVector3($0.x, $0.y + 0.002, $0.z) }
    if vertices.count >= 3 {
      let indices: [Int32] = (0..<vertices.count).flatMap { [Int32($0), Int32(($0 + 1) % vertices.count)] }
      let border = SCNGeometry(sources: [SCNGeometrySource(vertices: vertices)],
        elements: [SCNGeometryElement(indices: indices, primitiveType: .line)])
      border.materials = [surfaceMaterial(alpha: 1)]
      let edge = SCNNode(geometry: border); edge.renderingOrder = 11
      overlay.addChildNode(edge)
    }
  }
  private func drawBox() {
    box?.removeFromParentNode()
    guard let region else { return }
    let geometry = SCNBox(width: CGFloat(size), height: CGFloat(size), length: CGFloat(size), chamferRadius: 0)
    let material = surfaceMaterial(alpha: 1)
    material.fillMode = .lines
    geometry.materials = [material]
    let node = SCNNode(geometry: geometry)
    node.simdTransform = region
    node.position.y += size / 2
    preview.scene.rootNode.addChildNode(node); box = node
  }
  func setRecording(_ value: Bool) {
    if !value { finish(reason: "Recording saved"); return }
    guard !recording else { return }
    guard !finishing else { emit("Wait for the recording to finish saving.", extra: ["recording": false]); return }
    guard let frame = preview.session.currentFrame, case .normal = frame.camera.trackingState,
      region != nil || captureWithoutBox else {
      emit("Wait for normal tracking, then place a box or choose Skip box.", extra: ["recording": false]); return
    }
    do {
      let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
      let target = documents.appendingPathComponent("TareScans/" + UUID().uuidString, isDirectory: true)
      try FileManager.default.createDirectory(at: target.appendingPathComponent("images"), withIntermediateDirectories: true)
      folder = target; count = 0; dropped = 0; trackingSkipped = 0; frames = []
      startTime = frame.timestamp; lastSample = -.infinity
      manifest = ["schema_version": 1, "source": "arkit_rgb", "created_at": ISO8601DateFormatter().string(from: Date()),
        "device": UIDevice.current.model, "system_version": UIDevice.current.systemVersion,
        "world_units": "meters", "matrix_layout": "column_major", "pose": "camera_to_world",
        "camera_axes": "+X right, +Y up, -Z forward", "image_orientation": "native_sensor_unrotated",
        "depth_available": false,
        "max_frames": 240, "sample_interval_seconds": 0.5]
      if let region {
        manifest["region"] = ["origin": [region.columns.3.x, region.columns.3.y, region.columns.3.z],
          "size_m": size, "convention": "world aligned; origin at bottom center"]
      } else { manifest["region"] = NSNull() }
      recording = true; UIApplication.shared.isIdleTimerDisabled = true
      emit("Recording. Orbit slowly and keep the object stationary.", extra: ["recording": true])
    } catch { emit("Cannot create recording: \(error)", extra: ["recording": false]) }
  }
  func session(_ session: ARSession, didUpdate frame: ARFrame) {
    let tracking: String
    switch frame.camera.trackingState {
    case .normal: tracking = "normal"
    case .notAvailable: tracking = "unavailable"
    case .limited(let reason): tracking = "limited: \(reason)"
    }
    trackingDescription = tracking
    if frame.timestamp - lastStatus > 1 {
      lastStatus = frame.timestamp
      // Reconcile the complete live anchor list, including anchors whose initial
      // callback may have arrived before this view was ready. Zero now means the
      // frame contains no horizontal planes, rather than no callbacks received.
      let planes = frame.anchors.compactMap { $0 as? ARPlaneAnchor }
      let horizontalIDs = Set(planes.filter { $0.alignment == .horizontal }.map { $0.identifier })
      for id in Array(surfaces.keys) where !horizontalIDs.contains(id) {
        surfaces.removeValue(forKey: id)?.removeFromParentNode()
      }
      for plane in planes where plane.alignment == .horizontal {
        if let anchorNode = preview.node(for: plane) { updateSurface(plane, anchorNode: anchorNode) }
      }
      horizontalPlanes = horizontalIDs.count
      verticalPlanes = planes.filter { $0.alignment == .vertical }.count
      featurePoints = frame.rawFeaturePoints?.points.count ?? 0
      let needsCoaching = horizontalIDs.isEmpty && region == nil && !captureWithoutBox && !recording && !finishing
      if coaching.isActive != needsCoaching { coaching.setActive(needsCoaching, animated: true) }
      let message: String
      if recording { message = "Recording — \(count) frames saved" }
      else if finishing { message = "Saving recording…" }
      else if let placementHint, frame.timestamp < hintUntil { message = placementHint }
      else if tracking != "normal" { message = "Move slowly sideways over a well-lit, textured table to initialize tracking." }
      else if region != nil { message = "Scan box placed. Adjust size, then start." }
      else if captureWithoutBox { message = "Ready to capture without a scan box. No region will be saved." }
      else if horizontalPlanes == 0 { message = "Finding a table or floor. Look downward and move sideways; cyan patches mark detected surfaces." }
      else { message = "Tap a cyan surface near the object to place the scan box." }
      emit(message)
    }
    guard recording else { return }
    if frame.timestamp - startTime >= 120 || count >= 240 { finish(reason: "Capture limit reached; recording saved"); return }
    guard frame.timestamp - lastSample >= 0.5 else { return }
    guard case .normal = frame.camera.trackingState else { trackingSkipped += 1; lastSample = frame.timestamp; return }
    guard !busy, let folder else { dropped += 1; return }
    busy = true; lastSample = frame.timestamp
    let filename = String(format: "images/%05d.jpg", count)
    let buffer = frame.capturedImage // Retained until this single queued write finishes.
    let metadata: [String: Any] = ["image": filename, "timestamp_seconds": frame.timestamp,
      "width": CVPixelBufferGetWidth(buffer), "height": CVPixelBufferGetHeight(buffer),
      "intrinsics": floats(frame.camera.intrinsics), "camera_to_world": floats(frame.camera.transform),
      "tracking": "normal"]
    writer.async { [self] in
      do {
        try autoreleasepool {
          let image = CIImage(cvPixelBuffer: buffer)
          guard let color = CGColorSpace(name: CGColorSpace.sRGB),
            let jpeg = imageContext.jpegRepresentation(of: image, colorSpace: color,
              options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.9]) else { throw CaptureError.imageEncoding }
          try jpeg.write(to: folder.appendingPathComponent(filename), options: .atomic)
          frames.append(metadata)
        }
        DispatchQueue.main.async { [self] in count += 1; busy = false }
      } catch {
        DispatchQueue.main.async { [self] in busy = false; finish(reason: "Image write failed: \(error)") }
      }
    }
  }
  private func finish(reason: String) {
    guard recording, let folder else { return }
    recording = false; finishing = true
    UIApplication.shared.isIdleTimerDisabled = false
    let droppedCount = dropped; let skippedCount = trackingSkipped
    emit("Saving recording…")
    writer.async { [self] in
      var saved = manifest
      saved["frames"] = frames
      saved["diagnostics"] = ["saved_frames": frames.count, "writer_busy_skips": droppedCount,
        "tracking_skips": skippedCount, "stop_reason": reason]
      do {
        let data = try JSONSerialization.data(withJSONObject: saved, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: folder.appendingPathComponent("manifest.json"), options: .atomic)
        DispatchQueue.main.async { [self] in
          finishing = false; emit(reason, extra: ["recording": false, "path": folder.path])
        }
      } catch {
        DispatchQueue.main.async { [self] in finishing = false; emit("Save failed: \(error)", extra: ["recording": false]) }
      }
    }
  }
  func sessionWasInterrupted(_ session: ARSession) { finish(reason: "AR session interrupted; recording saved") }
  func sessionInterruptionEnded(_ session: ARSession) { region = nil; box?.removeFromParentNode(); runSession(); emit("Tracking interrupted. Place a new scan box.") }
  func session(_ session: ARSession, didFailWithError error: Error) { finish(reason: "AR session failed: \(error)"); emit("AR session error: \(error)") }
  private func emit(_ message: String, extra: [String: Any] = [:]) {
    let inView: Int
    if let camera = preview.pointOfView {
      inView = surfaces.values.filter { $0.parent != nil && preview.isNode($0, insideFrustumOf: camera) }.count
    } else { inView = 0 }
    var event: [String: Any] = ["message": message, "frames": count, "region": region != nil,
      "tracking": trackingDescription, "surfaces": horizontalPlanes,
      "overlays": surfaces.count, "overlaysInView": inView,
      "build": Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown",
      "walls": verticalPlanes, "points": featurePoints, "saving": finishing]
    event.merge(extra) { _, new in new }; onStatus(event)
  }
  private func floats(_ m: simd_float4x4) -> [Float] {
    (0..<4).flatMap { c in (0..<4).map { r in m[c][r] } }
  }
  private func floats(_ m: simd_float3x3) -> [Float] {
    (0..<3).flatMap { c in (0..<3).map { r in m[c][r] } }
  }
}
