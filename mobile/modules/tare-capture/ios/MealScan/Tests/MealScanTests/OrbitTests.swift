import XCTest
import simd
@testable import MealScan

/// Rendered grayscale images of a dish on a textured table, seen from an orbit.
/// The dish is a cylinder: a flat top at the rim height (lip, then a darker well and
/// optional food inside it) over a darker side wall.
struct DishImageScene {
  var radiusM: Double
  var rimHeightM: Double
  var wellRadiusM: Double
  var food: (center: SIMD2<Double>, radiusM: Double)? = nil

  func intensity(_ ray: Ray, frame: PlaneFrame, center: SIMD2<Double>) -> Float {
    let o = frame.coordinates(ray.originWorld) - center, oh = frame.height(ray.originWorld)
    let d = SIMD2(simd_dot(ray.directionWorld, frame.x), simd_dot(ray.directionWorld, frame.z))
    let dh = simd_dot(ray.directionWorld, frame.normal)
    guard dh < 0 else { return 0.5 }
    let top = o + (rimHeightM - oh) / dh * d
    if simd_length(top) <= radiusM {
      if let food, simd_distance(top, food.center) <= food.radiusM { return 0.35 }
      return simd_length(top) <= wellRadiusM ? 0.78 : 0.93
    }
    // Side wall: first hit of the cylinder below the rim.
    let a = simd_dot(d, d), b = 2 * simd_dot(o, d), c = simd_dot(o, o) - radiusM * radiusM
    let disc = b * b - 4 * a * c
    if disc >= 0 {
      let t = (-b - disc.squareRoot()) / (2 * a)
      let h = oh + t * dh
      if t > 0, h >= 0, h <= rimHeightM { return 0.6 }
    }
    let table = o + (0 - oh) / dh * d + center
    // Blocky texture: plenty of straight clutter edges at table height.
    let cell = SIMD2<Int>(Int((table.x / 0.017).rounded(.down)), Int((table.y / 0.023).rounded(.down)))
    var hash = UInt64(bitPattern: Int64(cell.x &* 73_856_093 ^ cell.y &* 19_349_663))
    hash = (hash ^ (hash >> 13)) &* 0x5bd1e995
    return 0.22 + 0.2 * Float(hash % 1000) / 1000
  }

  func render(_ keyframe: Keyframe, noise: inout GaussianNoise) -> GrayImage {
    let k = keyframe.intrinsics, frame = PlaneFrame(Synthetic.table)
    let center = frame.coordinates(Synthetic.dishCenter)
    var pixels = [Float](repeating: 0, count: k.widthPx * k.heightPx)
    for y in 0..<k.heightPx {
      for x in 0..<k.widthPx {
        let value = intensity(keyframe.rayWorld(throughPx: SIMD2(Double(x), Double(y))), frame: frame, center: center)
        pixels[y * k.widthPx + x] = 255 * (value + Float(noise.next(sigma: 0.01)))
      }
    }
    return GrayImage(widthPx: k.widthPx, heightPx: k.heightPx, pixels: pixels)
  }
}

final class OrbitTests: XCTestCase {
  static let small = Synthetic.rgb.scaled(toWidthPx: 480, heightPx: 360)

  /// Orbit keyframes looking at the dish: an overhead view plus rings at two elevations.
  static func orbit(distanceM: Double = 0.4) -> [Keyframe] {
    var eyes: [SIMD3<Double>] = [SIMD3(0.01, distanceM, 0.0)]
    for (elevationDeg, count) in [(60.0, 5), (35.0, 6)] {
      for i in 0..<count {
        let azimuth = 2 * .pi * Double(i) / Double(count) + elevationDeg
        let e = elevationDeg * .pi / 180
        eyes.append(distanceM * SIMD3(cos(e) * cos(azimuth), sin(e), cos(e) * sin(azimuth)))
      }
    }
    return eyes.enumerated().map { i, offset in
      Keyframe(id: i, timestampS: Double(i), intrinsics: small,
               worldFromCamera: Synthetic.pose(eye: Synthetic.dishCenter + offset, target: Synthetic.dishCenter))
    }
  }

  func fit(_ scene: DishImageScene, seedOffset: SIMD3<Double>) throws -> EdgeRimFit {
    var noise = GaussianNoise(seed: 7)
    let views = Self.orbit().map { EdgeView(keyframe: $0, image: scene.render($0, noise: &noise)) }
    return try EdgeRimFitter().fit(views, tablePlane: Synthetic.table, seedWorld: Synthetic.dishCenter + seedOffset).get()
  }

  func assertRim(_ fit: EdgeRimFit, radiusM: Double, heightM: Double, file: StaticString = #filePath, line: UInt = #line) {
    let rim = fit.rim
    XCTAssertEqual(rim.diameterM, 2 * radiusM, accuracy: 0.004, "diameter", file: file, line: line)
    XCTAssertEqual(rim.heightM, heightM, accuracy: 0.004, "height", file: file, line: line)
    XCTAssertLessThan(simd_distance(Synthetic.table.coordinates2D(rim.centerWorld),
                                    Synthetic.table.coordinates2D(Synthetic.dishCenter)), 0.004, "center", file: file, line: line)
    XCTAssertEqual(rim.method, .orbitEdges, file: file, line: line)
  }

  func testEdgeFitFindsPlateRimNotWellOrFood() throws {
    let plate = DishImageScene(radiusM: 0.13, rimHeightM: 0.02, wellRadiusM: 0.09, food: (SIMD2(0.02, -0.01), 0.045))
    let fit = try fit(plate, seedOffset: SIMD3(0.03, 0, -0.02))
    assertRim(fit, radiusM: 0.13, heightM: 0.02)
    XCTAssertGreaterThan(fit.edgeSupport, 0.3)
  }

  func testEdgeFitMeasuresDeepBowlHeight() throws {
    let bowl = DishImageScene(radiusM: 0.08, rimHeightM: 0.06, wellRadiusM: 0.07)
    assertRim(try fit(bowl, seedOffset: SIMD3(-0.02, 0, 0.02)), radiusM: 0.08, heightM: 0.06)
  }

  func testEdgeFitNeedsViews() {
    let views = Self.orbit().prefix(2).map { EdgeView(keyframe: $0, image: GrayImage(widthPx: 480, heightPx: 360, pixels: .init(repeating: 0, count: 480 * 360))) }
    guard case .failure(.insufficientViews) = EdgeRimFitter().fit(Array(views), tablePlane: Synthetic.table, seedWorld: Synthetic.dishCenter) else {
      return XCTFail("expected insufficientViews")
    }
  }

  func testOrbitSelectorSpreadsKeyframesAndTracksCoverage() {
    var selector = OrbitSelector(targetWorld: Synthetic.dishCenter, tablePlane: Synthetic.table)
    let views = Self.orbit()
    XCTAssertNil(selector.consider(views[1], angularSpeedDegS: 5, speedMS: 0.05))
    XCTAssertEqual(selector.consider(views[1], angularSpeedDegS: 5, speedMS: 0.05), .redundant)
    XCTAssertEqual(selector.consider(views[2], angularSpeedDegS: 90, speedMS: 0.05), .moving)
    for view in views.dropFirst(2) { XCTAssertNil(selector.consider(view, angularSpeedDegS: 5, speedMS: 0.05)) }
    XCTAssertEqual(selector.coverage, 11.0 / 24, accuracy: 1e-9)
    // A camera looking away from the dish is rejected.
    let away = Keyframe(id: 99, timestampS: 0, intrinsics: Self.small, worldFromCamera: Synthetic.pose(
      eye: Synthetic.dishCenter + SIMD3(0, 0.4, 0.2), target: Synthetic.dishCenter + SIMD3(0, 0.4, 1)))
    XCTAssertEqual(selector.consider(away, angularSpeedDegS: 0, speedMS: 0), .outOfFrame)
  }

  func testProbeSpotsSitInsideRimOnTheSurface() {
    let rim = RimEstimate(centerWorld: Synthetic.dishCenter + SIMD3(0, 0.02, 0), radiusM: 0.13, heightM: 0.02,
                          radiusUncertaintyM: nil, heightUncertaintyM: nil, rmsResidualPx: 0,
                          method: .orbitEdges, viewCount: 12, confidence: 1)
    // Surface points 15 mm up, only on the +X half of the plate; unmeasured spots borrow their height.
    let center = Synthetic.table.coordinates2D(Synthetic.dishCenter)
    let points = stride(from: 0.0, to: 0.12, by: 0.004).flatMap { x in
      stride(from: -0.12, to: 0.12, by: 0.004).map { z in Synthetic.table.pointWorld(at: center + SIMD2(x, z), heightM: 0.015) }
    }
    let spots = ProbeRing.spots(rim: rim, tablePlane: Synthetic.table, surfacePointsWorld: points)
    XCTAssertEqual(spots.count, 8)
    for spot in spots {
      let offset = Synthetic.table.coordinates2D(spot.positionWorld) - center
      XCTAssertEqual(simd_length(offset), 0.13 - 0.0254, accuracy: 1e-9)
      if offset.x > 0.01 {
        XCTAssertTrue(spot.measuredSurface); XCTAssertEqual(spot.surfaceHeightM, 0.015, accuracy: 1e-9)
      } else if offset.x < -0.02 {
        XCTAssertFalse(spot.measuredSurface); XCTAssertEqual(spot.surfaceHeightM, 0.015, accuracy: 1e-9)
      }
      XCTAssertEqual(Synthetic.table.signedDistanceM(spot.positionWorld), spot.surfaceHeightM, accuracy: 1e-9)
    }
    // With no points at all, the fallback plate-floor height is used.
    XCTAssertEqual(ProbeRing.spots(rim: rim, tablePlane: Synthetic.table).map(\.surfaceHeightM), Array(repeating: 0.008, count: 8))
  }
}
