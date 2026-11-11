import XCTest
import simd
@testable import MealScan

final class ProbeStrokeTests: XCTestCase {
  static let modelURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Assets/probe-pen.glb")

  func model() throws -> GLBModel { try GLBModel(data: Data(contentsOf: Self.modelURL)) }

  func testReadsThePrototypeModel() throws {
    let model = try model()
    XCTAssertEqual(model.nodes.count, 25)
    XCTAssertEqual(model.materials.map(\.name).first, "shell_pc_abs")
    XCTAssertEqual(model.materials.first { $0.name == "led_lens" }?.emissive.x ?? 0, 0.55, accuracy: 1e-6)
    let shell = try XCTUnwrap(model.index(named: "shell"))
    let xs = model.modelPositions(of: shell).map(\.x)
    // About 150 mm long once the root's mm → m scale is applied.
    XCTAssertEqual(try XCTUnwrap(xs.max()) - XCTUnwrap(xs.min()), 0.150, accuracy: 0.002)
    for node in model.nodes { for p in node.primitives { XCTAssertEqual(p.indices.count % 3, 0) } }
  }

  func testRigFindsTheBladeAndSlider() throws {
    let rig = try ProbePenRig(model: model())
    XCTAssertEqual(simd_distance(rig.axis, SIMD3(1, 0, 0)), 0, accuracy: 1e-9)
    XCTAssertEqual(simd_distance(rig.top, SIMD3(0, 0, 1)), 0, accuracy: 1e-6)
    XCTAssertEqual(rig.travelM, 0.1394, accuracy: 1e-4)
    XCTAssertEqual(rig.tipRetracted.x, 0.08115, accuracy: 1e-4)  // Just inside the nose aperture.
    XCTAssertEqual(simd_length(rig.carriageTravelLocal), 139.4, accuracy: 0.1)  // Parent is in millimeters.
    XCTAssertNotNil(rig.button); XCTAssertNotNil(rig.led)
  }

  func testStrokeGeometry() throws {
    let rig = try ProbePenRig(model: model())
    let up = simd_normalize(SIMD3(0.05, 1, -0.02)), target = SIMD3(0.1, -0.3, -0.4)
    let rotation = ProbeStroke.orientation(rig: rig, upWorld: up, viewerWorld: target + SIMD3(0, 0.3, 0.3), targetWorld: target)
    XCTAssertEqual(simd_determinant(rotation), 1, accuracy: 1e-9)
    let axis = rotation * rig.axis
    XCTAssertEqual(acos(-simd_dot(axis, up)) * 180 / .pi, 30, accuracy: 1e-6)
    XCTAssertGreaterThan(simd_dot(rotation * rig.top, SIMD3(0, 0.3, 0.3)), 0)  // Button faces the viewer.

    let tips = ProbeStroke.tips(for: .init(positionWorld: target, depthM: 0.01), rotation: rotation, rig: rig, upWorld: up)
    XCTAssertEqual(simd_dot(tips.inserted - target, up), -0.01, accuracy: 1e-9)
    XCTAssertGreaterThan(simd_dot(tips.hover - target, up), 0.025)
    let origin = ProbeStroke.modelOrigin(tipWorld: tips.inserted, rotation: rotation, rig: rig)
    XCTAssertEqual(simd_distance(origin + rotation * rig.tip(extension: 1), tips.inserted), 0, accuracy: 1e-12)
    // The pen body stays well above the surface while the blade is in.
    XCTAssertGreaterThan(simd_dot(origin - target, up), 0.08)
  }

  func testOrderAndDepth() {
    let points = [SIMD3(0.1, 0, 0), SIMD3(0.0, 0, 0), SIMD3(0.05, 0, 0)]
    XCTAssertEqual(ProbeStroke.order(points, fromWorld: SIMD3(-1, 0, 0)), [1, 2, 0])
    XCTAssertEqual(ProbeStroke.foodDepthM(heightAboveTableM: 0.03, plateSurfaceM: 0.01), 0.012, accuracy: 1e-12)
    XCTAssertEqual(ProbeStroke.foodDepthM(heightAboveTableM: 0.011, plateSurfaceM: 0.01), 0.003)
    XCTAssertEqual(ProbeStroke.foodDepthM(heightAboveTableM: 0.08, plateSurfaceM: 0.01), 0.015)
  }

  func testRejectsNonGLB() {
    XCTAssertThrowsError(try GLBModel(data: Data("not a model at all".utf8)))
  }
}
