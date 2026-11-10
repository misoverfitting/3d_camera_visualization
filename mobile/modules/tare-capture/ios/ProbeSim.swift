import ARKit
import SceneKit
import simd

// Probe simulation for the plate demo: once points are on the plate, the prototype probe
// pen (Assets/probe-pen.glb) visits each one: slider forward, blade into the food,
// confirm button, LED blink, a ring marks the spot measured, then on to the next.
// Geometry and stroke poses come from MealScan (GLBModel, ProbePenRig, ProbeStroke).

final class ProbeSim {
  /// Lights in this category light only the pen; the demo's markers are unlit.
  static let category = 1 << 3

  let node = SCNNode()
  private(set) var running = false
  private(set) var done = 0
  private(set) var total = 0
  /// Label of the spot being probed.
  private(set) var current: String?
  private var generation = 0

  private static var loaded: (template: SCNNode, rig: ProbePenRig)?? = nil

  static func load() -> (template: SCNNode, rig: ProbePenRig)? {
    if let loaded { return loaded }
    let bundle = Bundle.main.url(forResource: "TareCaptureAssets", withExtension: "bundle").flatMap(Bundle.init(url:))
    let url = bundle?.url(forResource: "probe-pen", withExtension: "glb") ?? Bundle.main.url(forResource: "probe-pen", withExtension: "glb")
    let result = url.flatMap { try? Data(contentsOf: $0) }.flatMap { try? GLBModel(data: $0) }.flatMap { model -> (SCNNode, ProbePenRig)? in
      guard let rig = try? ProbePenRig(model: model) else { return nil }
      return (sceneNode(model), rig)
    }
    loaded = .some(result)
    return result
  }

  var status: [String: Any] { ["running": running, "done": done, "total": total] }

  func stop() {
    generation += 1
    running = false; done = 0; total = 0; current = nil
    node.removeAllActions()
    node.childNodes.forEach { $0.removeFromParentNode() }
  }

  struct Stop {
    var target: ProbeStroke.Target
    var color: UIColor
    var label: String
  }

  /// Runs one stroke per stop, in nearest-neighbor order from the viewer. `progress` is
  /// called on the main thread with (done, total, label of the spot being probed or nil).
  func run(_ stops: [Stop], upWorld: SIMD3<Double>, viewerWorld: SIMD3<Double>,
           progress: @escaping (Int, Int, String?) -> Void) -> Bool {
    stop()
    guard let (template, rig) = Self.load(), !stops.isEmpty else { return false }
    running = true; total = stops.count
    let generation = self.generation
    let center = stops.map(\.target.positionWorld).reduce(.zero, +) / Double(stops.count)
    let rotation = ProbeStroke.orientation(rig: rig, upWorld: upWorld, viewerWorld: viewerWorld, targetWorld: center)
    let order = ProbeStroke.order(stops.map(\.target.positionWorld), fromWorld: viewerWorld)

    addLights(upWorld: upWorld)
    let pen = template.clone()  // Shares geometry; transforms are per clone.
    pen.simdOrientation = simd_quatf(simd_float3x3(SIMD3<Float>(rotation.columns.0), SIMD3<Float>(rotation.columns.1), SIMD3<Float>(rotation.columns.2)))
    pen.opacity = 0
    node.addChildNode(pen)
    guard let carriage = pen.childNode(withName: "carriage", recursively: true) else { stop(); return false }
    let button = pen.childNode(withName: "confirm_button", recursively: true)
    let led = pen.childNode(withName: "status_led", recursively: true)
    let ledMaterial = led?.geometry?.firstMaterial.map { $0.copy() as! SCNMaterial }
    if let ledMaterial { led?.geometry = (led?.geometry?.copy() as? SCNGeometry).map { $0.materials = [ledMaterial]; return $0 } }

    func vector(_ v: SIMD3<Double>) -> SCNVector3 { SCNVector3(Float(v.x), Float(v.y), Float(v.z)) }
    func origin(_ tip: SIMD3<Double>) -> SCNVector3 { vector(ProbeStroke.modelOrigin(tipWorld: tip, rotation: rotation, rig: rig)) }
    func move(_ tip: SIMD3<Double>, _ duration: TimeInterval) -> SCNAction {
      let action = SCNAction.move(to: origin(tip), duration: duration)
      action.timingMode = .easeInEaseOut
      return action
    }
    func slide(_ fraction: Double, _ duration: TimeInterval) -> SCNAction {
      let rest = rig.carriageRest.columns.3
      let action = SCNAction.move(to: vector(SIMD3(rest.x, rest.y, rest.z) + fraction * rig.carriageTravelLocal), duration: duration)
      action.timingMode = .easeInEaseOut
      return action
    }
    let alive = { [weak self] in self?.generation == generation }
    // Action blocks run on SceneKit's render thread; scene edits go back to main.
    func onMain(_ block: @escaping () -> Void) -> SCNAction { .run { _ in DispatchQueue.main.async { if alive() { block() } } } }

    var steps: [SCNAction] = []
    let first = ProbeStroke.tips(for: stops[order[0]].target, rotation: rotation, rig: rig, upWorld: upWorld)
    pen.position = origin(first.hover)
    steps.append(.fadeIn(duration: 0.3))
    steps.append(onMain { carriage.runAction(slide(1, 0.6)) })
    steps.append(.wait(duration: 0.7))
    for (k, index) in order.enumerated() {
      let stop = stops[index]
      let tips = ProbeStroke.tips(for: stop.target, rotation: rotation, rig: rig, upWorld: upWorld)
      let occluder = Self.occluder(at: stop.target.positionWorld, up: upWorld)
      steps.append(.run { _ in DispatchQueue.main.async { [weak self] in if alive() { self?.current = stop.label; progress(k, stops.count, stop.label) } } })
      if k > 0 { steps.append(move(tips.hover, 0.5)) }
      steps.append(onMain { [weak self] in self?.node.addChildNode(occluder) })
      steps.append(move(tips.inserted, 0.45))
      steps.append(onMain { [weak self] in
        button.map { b in
          let rest = SCNVector3(Float(rig.buttonRest?.columns.3.x ?? 0), Float(rig.buttonRest?.columns.3.y ?? 0), Float(rig.buttonRest?.columns.3.z ?? 0))
          let pressed = SCNVector3(rest.x + Float(rig.buttonPressLocal.x), rest.y + Float(rig.buttonPressLocal.y), rest.z + Float(rig.buttonPressLocal.z))
          b.runAction(.sequence([.move(to: pressed, duration: 0.08), .wait(duration: 0.12), .move(to: rest, duration: 0.1)]))
        }
        if let ledMaterial {
          led?.runAction(.sequence([.wait(duration: 0.25), .customAction(duration: 0.7) { _, t in
            // Two blinks, then back to the idle glow.
            ledMaterial.emission.intensity = 1 + 5 * max(0, sin(CGFloat(t) / 0.7 * 4 * .pi))
          }, .run { _ in ledMaterial.emission.intensity = 1 }]))
        }
        self?.mark(stop, up: upWorld, after: 0.9)
      })
      steps.append(.wait(duration: 1.1))
      steps.append(move(tips.hover, 0.35))
      steps.append(onMain { occluder.removeFromParentNode() })
      steps.append(.run { _ in DispatchQueue.main.async { [weak self] in
        guard let self, alive() else { return }
        self.done = k + 1; progress(k + 1, stops.count, nil)
      } })
    }
    steps.append(onMain { carriage.runAction(slide(0, 0.45)) })
    steps.append(.wait(duration: 0.5))
    let lift = SCNAction.move(by: vector(0.05 * upWorld), duration: 0.4)
    steps.append(.group([lift, .fadeOut(duration: 0.4)]))
    steps.append(.run { _ in DispatchQueue.main.async { [weak self] in
      guard let self, alive() else { return }
      pen.removeFromParentNode(); self.running = false; self.current = nil; progress(stops.count, stops.count, nil)
    } })
    pen.runAction(.sequence(steps))
    return true
  }

  /// A white ring around the spot once its reading is in, after a short pulse.
  private func mark(_ stop: Stop, up: SIMD3<Double>, after delay: TimeInterval) {
    let orientation = simd_quatf(from: SIMD3(0, 1, 0), to: SIMD3<Float>(simd_normalize(up)))
    let position = SIMD3<Float>(stop.target.positionWorld + 0.0008 * simd_normalize(up))
    let pulse = SCNNode(geometry: SCNTorus(ringRadius: 0.009, pipeRadius: 0.0008))
    pulse.geometry?.materials = [Self.flat(stop.color)]
    pulse.simdPosition = position; pulse.simdOrientation = orientation
    pulse.opacity = 0
    node.addChildNode(pulse)
    pulse.runAction(.sequence([.wait(duration: delay * 0.3), .fadeIn(duration: 0.05),
                               .group([.scale(to: 2.6, duration: 0.6), .fadeOut(duration: 0.6)]), .removeFromParentNode()]))
    let ring = SCNNode(geometry: SCNTorus(ringRadius: 0.0085, pipeRadius: 0.0011))
    ring.geometry?.materials = [Self.flat(.white)]
    ring.simdPosition = position; ring.simdOrientation = orientation
    ring.opacity = 0
    node.addChildNode(ring)
    ring.runAction(.sequence([.wait(duration: delay), .fadeIn(duration: 0.2)]))
  }

  /// Depth-only disc on the surface around the spot, so the part of the blade that is
  /// "in the food" is hidden when seen from above.
  private static func occluder(at position: SIMD3<Double>, up: SIMD3<Double>) -> SCNNode {
    let material = SCNMaterial()
    material.colorBufferWriteMask = []
    material.writesToDepthBuffer = true
    material.isDoubleSided = true
    let disc = SCNNode(geometry: SCNCylinder(radius: 0.03, height: 0.0002))
    disc.geometry?.materials = [material]
    disc.simdPosition = SIMD3<Float>(position)
    disc.simdOrientation = simd_quatf(from: SIMD3(0, 1, 0), to: SIMD3<Float>(simd_normalize(up)))
    disc.renderingOrder = -10  // Before the pen, so the blade below it fails the depth test.
    return disc
  }

  private func addLights(upWorld: SIMD3<Double>) {
    let ambient = SCNLight()
    ambient.type = .ambient; ambient.intensity = 450; ambient.categoryBitMask = Self.category
    let key = SCNLight()
    key.type = .directional; key.intensity = 1100; key.categoryBitMask = Self.category
    let ambientNode = SCNNode(), keyNode = SCNNode()
    ambientNode.light = ambient; keyNode.light = key
    // Directional lights shine along their node's -Z: point it mostly down.
    keyNode.simdOrientation = simd_quatf(from: SIMD3(0, 0, -1), to: simd_normalize(SIMD3<Float>(-simd_normalize(upWorld)) + SIMD3(0.3, 0, 0.2)))
    node.addChildNode(ambientNode); node.addChildNode(keyNode)
  }

  private static func flat(_ color: UIColor) -> SCNMaterial {
    let material = SCNMaterial()
    material.diffuse.contents = color
    material.lightingModel = .constant
    material.readsFromDepthBuffer = false
    return material
  }

  // MARK: glTF → SceneKit

  private static func sceneNode(_ model: GLBModel) -> SCNNode {
    let materials = model.materials.map(material)
    func build(_ index: Int) -> SCNNode {
      let source = model.nodes[index]
      let node = SCNNode()
      node.name = source.name
      let m = source.matrix
      node.simdTransform = simd_float4x4(SIMD4<Float>(m.columns.0), SIMD4<Float>(m.columns.1), SIMD4<Float>(m.columns.2), SIMD4<Float>(m.columns.3))
      node.categoryBitMask = category
      node.renderingOrder = 10
      let geometries = source.primitives.map { geometry($0, materials: materials) }
      if geometries.count == 1 {
        node.geometry = geometries[0]
      } else {
        for g in geometries { let child = SCNNode(geometry: g); child.categoryBitMask = category; child.renderingOrder = 10; node.addChildNode(child) }
      }
      source.children.forEach { node.addChildNode(build($0)) }
      return node
    }
    let root = SCNNode()
    root.name = "probe-pen"
    model.roots.forEach { root.addChildNode(build($0)) }
    return root
  }

  private static func geometry(_ primitive: GLBModel.Primitive, materials: [SCNMaterial]) -> SCNGeometry {
    var sources = [SCNGeometrySource(vertices: primitive.positions.map { SCNVector3($0.x, $0.y, $0.z) })]
    if !primitive.normals.isEmpty { sources.append(SCNGeometrySource(normals: primitive.normals.map { SCNVector3($0.x, $0.y, $0.z) })) }
    let element = SCNGeometryElement(indices: primitive.indices, primitiveType: .triangles)
    let geometry = SCNGeometry(sources: sources, elements: [element])
    geometry.materials = [primitive.material.flatMap { $0 < materials.count ? materials[$0] : nil } ?? SCNMaterial()]
    return geometry
  }

  private static func material(_ source: GLBModel.Material) -> SCNMaterial {
    func srgb(_ c: Double) -> CGFloat { CGFloat(c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1 / 2.4) - 0.055) }
    let material = SCNMaterial()
    material.name = source.name
    material.lightingModel = .physicallyBased
    material.diffuse.contents = UIColor(red: srgb(source.baseColor.x), green: srgb(source.baseColor.y),
                                        blue: srgb(source.baseColor.z), alpha: CGFloat(source.baseColor.w))
    material.metalness.contents = NSNumber(value: source.metallic)
    material.roughness.contents = NSNumber(value: source.roughness)
    if source.emissive != .zero {
      let peak = max(1, source.emissive.max())
      material.emission.contents = UIColor(red: srgb(source.emissive.x / peak), green: srgb(source.emissive.y / peak),
                                           blue: srgb(source.emissive.z / peak), alpha: 1)
      material.emission.intensity = 1
    }
    return material
  }
}

extension TareCaptureView {
  /// Probes the food points when Claude labeled the plate, else the ring of probe spots.
  func demoProbe() {
    guard demo.stage == .done, demo.fit != nil, let table = demo.table else { return emit("Finish the plate scan first.") }
    let up = table.normalWorld
    let lift = (demo.heightOffsetM + 0.0005) * up
    var stops: [ProbeSim.Stop] = []
    if !demo.food.items.isEmpty {
      let plate = Numerics.percentile(demo.spots.map(\.surfaceHeightM), 0.5) ?? PipelineConfig.Probe().fallbackSurfaceHeightM
      for item in demo.food.items {
        let color = FoodState.palette[item.colorIndex % FoodState.palette.count].0
        for point in item.points {
          let depth = ProbeStroke.foodDepthM(heightAboveTableM: table.signedDistanceM(point.positionWorld), plateSurfaceM: plate)
          stops.append(.init(target: .init(positionWorld: point.positionWorld + lift, depthM: depth), color: color, label: item.label))
        }
      }
    } else {
      // Ring spots may be bare plate: touch, don't stab.
      stops = demo.spots.map { .init(target: .init(positionWorld: $0.positionWorld + lift, depthM: 0), color: .systemOrange, label: "spot \($0.index + 1)") }
    }
    guard !stops.isEmpty else { return emit("No points to probe yet.") }
    guard let camera = preview.session.currentFrame?.camera.transform.columns.3 else { return }
    let started = demo.probe.run(stops, upWorld: up, viewerWorld: SIMD3(Double(camera.x), Double(camera.y), Double(camera.z))) { [weak self] done, total, label in
      guard let self else { return }
      if let label { self.emit("Probing \(label) (\(done + 1) of \(total))…") }
      else if done == total, !self.demo.probe.running { self.emit("Probed \(total) spot\(total == 1 ? "" : "s").") }
    }
    if !started { emit("The probe model is missing from the app bundle.") }
  }
}
