import Foundation
import simd

/// The moving parts of the prototype probe pen (Assets/probe-pen.glb), read from its model:
/// the slider carriage that pushes the blade out of the nose, the confirm button and the
/// status LED. All vectors are in model coordinates (meters, the scene root's frame).
public struct ProbePenRig: Sendable {
  public enum Failure: Error, Equatable { case missingPart(String) }

  /// Unit direction the blade travels out of the nose.
  public var axis: SIMD3<Double>
  /// Unit direction from the blade axis toward the button side, perpendicular to `axis`.
  public var top: SIMD3<Double>
  /// Blade tip with the slider back (blade inside the shell).
  public var tipRetracted: SIMD3<Double>
  /// Slider travel between its two detents.
  public var travelM: Double
  public var carriage: Int
  public var button: Int?
  public var led: Int?
  /// `carriage`'s rest transform and full extension, in its parent's coordinates.
  public var carriageRest: simd_double4x4
  public var carriageTravelLocal: SIMD3<Double>
  /// `button`'s rest transform and a 0.8 mm press, in its parent's coordinates.
  public var buttonRest: simd_double4x4?
  public var buttonPressLocal: SIMD3<Double>

  public init(model: GLBModel) throws {
    func part(_ name: String) throws -> Int {
      guard let index = model.index(named: name) else { throw Failure.missingPart(name) }
      return index
    }
    let matrices = model.modelMatrices()
    func origin(_ index: Int) -> SIMD3<Double> { let c = (matrices[index] ?? matrix_identity_double4x4).columns.3; return SIMD3(c.x, c.y, c.z) }
    func parent(of index: Int) -> Int? { model.nodes.indices.first { model.nodes[$0].children.contains(index) } }
    func local(_ direction: SIMD3<Double>, under index: Int) -> SIMD3<Double> {
      let m = parent(of: index).flatMap { matrices[$0] } ?? matrix_identity_double4x4
      let v = m.inverse * SIMD4(direction, 0)
      return SIMD3(v.x, v.y, v.z)
    }

    carriage = try part("carriage")
    let blade = try part("probe_blade")
    let retracted = origin(try part("detent_retracted")), extended = origin(try part("detent_extended"))
    let aperture = origin(try part("blade_aperture"))
    travelM = simd_distance(retracted, extended)
    let axis = simd_normalize(extended - retracted)
    self.axis = axis
    let bladePoints = model.modelPositions(of: blade)
    guard let tipAlong = bladePoints.map({ simd_dot($0 - aperture, axis) }).max() else { throw Failure.missingPart("probe_blade mesh") }
    tipRetracted = aperture + tipAlong * axis

    button = model.index(named: "confirm_button")
    led = model.index(named: "status_led")
    let buttonPosition = button.map(origin) ?? (aperture + SIMD3(0, 0, 0.004))
    let offset = buttonPosition - aperture
    let top = simd_normalize(offset - simd_dot(offset, axis) * axis)
    self.top = top

    carriageRest = model.nodes[carriage].matrix
    carriageTravelLocal = local(simd_distance(retracted, extended) * axis, under: carriage)
    buttonRest = button.map { model.nodes[$0].matrix }
    buttonPressLocal = button.map { local(-0.0008 * top, under: $0) } ?? .zero
  }

  public func tip(extension fraction: Double) -> SIMD3<Double> { tipRetracted + fraction * travelM * axis }
}

/// Poses for animating one probe stroke per target: hover above the spot with the blade
/// out, push the tip in along the pen's axis, hold for the reading, pull back.
public enum ProbeStroke {
  public struct Target: Sendable {
    public var positionWorld: SIMD3<Double>
    /// How far below the surface the tip goes (0 = touch).
    public var depthM: Double
    public init(positionWorld: SIMD3<Double>, depthM: Double) { self.positionWorld = positionWorld; self.depthM = depthM }
  }

  /// World-from-model rotation: the blade points down at `tiltRad` from vertical, leaning
  /// sideways to the viewer so the pen is seen in profile, with the button facing them.
  public static func orientation(rig: ProbePenRig, upWorld: SIMD3<Double>, viewerWorld: SIMD3<Double>,
                                 targetWorld: SIMD3<Double>, tiltRad: Double = 30 * .pi / 180) -> simd_double3x3 {
    let up = simd_normalize(upWorld)
    var toViewer = viewerWorld - targetWorld
    var flat = toViewer - simd_dot(toViewer, up) * up
    if simd_length(flat) < 1e-6 { flat = abs(up.x) < 0.9 ? simd_cross(up, SIMD3(1, 0, 0)) : simd_cross(up, SIMD3(0, 0, 1)) }
    flat = simd_normalize(flat)
    let right = simd_cross(up, flat)  // Viewer's right, on the table.
    let axisWorld = simd_normalize(-cos(tiltRad) * up - sin(tiltRad) * right)
    if simd_length(toViewer) < 1e-6 { toViewer = up }
    var topWorld = simd_normalize(toViewer) - simd_dot(simd_normalize(toViewer), axisWorld) * axisWorld
    if simd_length(topWorld) < 1e-6 { topWorld = up - simd_dot(up, axisWorld) * axisWorld }
    topWorld = simd_normalize(topWorld)
    let world = simd_double3x3(axisWorld, topWorld, simd_cross(axisWorld, topWorld))
    let model = simd_double3x3(rig.axis, rig.top, simd_cross(rig.axis, rig.top))
    return world * model.transpose
  }

  /// World position of the model origin that puts the tip (at `extension` of the slider
  /// travel) on `tipWorld`.
  public static func modelOrigin(tipWorld: SIMD3<Double>, rotation: simd_double3x3, rig: ProbePenRig,
                                 extension fraction: Double = 1) -> SIMD3<Double> {
    tipWorld - rotation * rig.tip(extension: fraction)
  }

  /// Tip positions for a target: hovering `hoverM` back along the axis, and inserted.
  public static func tips(for target: Target, rotation: simd_double3x3, rig: ProbePenRig,
                          upWorld: SIMD3<Double>, hoverM: Double = 0.035) -> (hover: SIMD3<Double>, inserted: SIMD3<Double>) {
    let axisWorld = rotation * rig.axis
    // Along the tilted axis, so the tip ends `depthM` below the surface vertically.
    let along = target.depthM / max(0.2, -simd_dot(axisWorld, simd_normalize(upWorld)))
    return (target.positionWorld - hoverM * axisWorld, target.positionWorld + along * axisWorld)
  }

  /// Visiting order: start nearest `fromWorld`, then always the nearest unvisited target.
  public static func order(_ targets: [SIMD3<Double>], fromWorld: SIMD3<Double>) -> [Int] {
    var left = Array(targets.indices), out: [Int] = [], at = fromWorld
    while let next = left.min(by: { simd_distance(targets[$0], at) < simd_distance(targets[$1], at) }) {
      out.append(next); left.removeAll { $0 == next }; at = targets[next]
    }
    return out
  }

  /// Insertion depth into food: 60% of its height above the plate surface, 3–15 mm.
  public static func foodDepthM(heightAboveTableM: Double, plateSurfaceM: Double) -> Double {
    min(0.015, max(0.003, 0.6 * (heightAboveTableM - plateSurfaceM)))
  }
}
