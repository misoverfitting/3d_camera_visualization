import XCTest
import simd
@testable import MealScan

/// Geometry modules must recover synthetic rims to 0.1 mm (tare_impl.md §13).
final class RimGeometryTests: XCTestCase {
  let toleranceM = 0.0001

  /// Keyframe A near overhead; keyframe B after a 12 cm sideways slide (§6.1).
  func views() -> (Keyframe, Keyframe) {
    (Synthetic.keyframe(id: 0, offset: SIMD3(0.0, 0, 0.08)), Synthetic.keyframe(id: 1, offset: SIMD3(0.12, 0, 0.08)))
  }

  func assertRim(_ result: Result<RimEstimate, RimFailure>, radiusM: Double, heightM: Double,
                 file: StaticString = #filePath, line: UInt = #line) {
    guard case .success(let rim) = result else { return XCTFail("\(result)", file: file, line: line) }
    XCTAssertEqual(rim.radiusM, radiusM, accuracy: toleranceM, file: file, line: line)
    XCTAssertEqual(rim.heightM, heightM, accuracy: toleranceM, file: file, line: line)
    XCTAssertLessThan(simd_distance(rim.centerWorld, Synthetic.dishCenter + SIMD3(0, heightM, 0)), toleranceM, file: file, line: line)
  }

  func observations(radiusM: Double, heightM: Double) -> [RimObservation] {
    let (a, b) = views()
    return [RimObservation(keyframe: a, keypointsPx: Synthetic.rimKeypoints(radiusM: radiusM, heightM: heightM, in: a)),
            RimObservation(keyframe: b, keypointsPx: Synthetic.rimKeypoints(radiusM: radiusM, heightM: heightM, in: b, phase: 0.1))]
  }

  func testTriangulatesDinnerPlate() {
    let result = RimTriangulator().triangulate(observations(radiusM: 0.132, heightM: 0.021), tablePlane: Synthetic.table)
    assertRim(result, radiusM: 0.132, heightM: 0.021)
    XCTAssertEqual(try result.get().method, .twoView)
  }

  func testTriangulatesDeepBowl() {
    assertRim(RimTriangulator().triangulate(observations(radiusM: 0.075, heightM: 0.072), tablePlane: Synthetic.table),
              radiusM: 0.075, heightM: 0.072)
  }

  func testTriangulatesFlatRimAtTableHeight() {
    assertRim(RimTriangulator().triangulate(observations(radiusM: 0.10, heightM: 0.0), tablePlane: Synthetic.table),
              radiusM: 0.10, heightM: 0.0)
  }

  /// Food covering most of the rim: occluded keypoints carry zero weight.
  func testIgnoresOccludedKeypoints() {
    var obs = observations(radiusM: 0.132, heightM: 0.021)
    obs[0].weights = obs[0].weights.indices.map { $0 % 3 == 0 ? 1 : 0 }
    obs[1].weights = obs[1].weights.indices.map { $0 < 12 ? 1 : 0 }
    // Corrupt the occluded pixels: they must not influence the fit.
    obs[0].keypointsPx = obs[0].keypointsPx.enumerated().map { $0.offset % 3 == 0 ? $0.element : $0.element + SIMD2(40, -25) }
    let result = RimTriangulator().triangulate(obs, tablePlane: Synthetic.table)
    assertRim(result, radiusM: 0.132, heightM: 0.021)
    XCTAssertLessThan(try result.get().confidence, 1)  // Visible fraction is under 60%.
  }

  /// Not a 0.1 mm gate: realistic keypoint noise must stay within the M2 bar (median ≤ 3%).
  func testHalfPixelNoiseStaysWithinThreePercent() throws {
    var noise = GaussianNoise(seed: 42)
    var obs = observations(radiusM: 0.132, heightM: 0.021)
    for i in obs.indices {
      obs[i].keypointsPx = obs[i].keypointsPx.map { $0 + SIMD2(noise.next(sigma: 0.5), noise.next(sigma: 0.5)) }
    }
    let rim = try RimTriangulator().triangulate(obs, tablePlane: Synthetic.table).get()
    XCTAssertEqual(rim.diameterM, 0.264, accuracy: 0.264 * 0.03)
    XCTAssertEqual(rim.heightM, 0.021, accuracy: 0.005)
    let sigma = try XCTUnwrap(rim.radiusUncertaintyM)
    XCTAssertGreaterThan(sigma, 0)
    XCTAssertLessThan(abs(rim.radiusM - 0.132), 4 * sigma + 1e-4, "error should be consistent with reported uncertainty")
  }

  func testRejectsDegenerateCaptures() {
    let (a, _) = views()
    let single = RimObservation(keyframe: a, keypointsPx: Synthetic.rimKeypoints(radiusM: 0.1, heightM: 0.02, in: a))
    XCTAssertEqual(RimTriangulator().triangulate([single], tablePlane: Synthetic.table).failure, .insufficientViews)
    let nearby = Synthetic.keyframe(id: 2, offset: SIMD3(0.003, 0, 0.08))
    let second = RimObservation(keyframe: nearby, keypointsPx: Synthetic.rimKeypoints(radiusM: 0.1, heightM: 0.02, in: nearby))
    XCTAssertEqual(RimTriangulator().triangulate([single, second], tablePlane: Synthetic.table).failure, .insufficientBaseline)
  }

  func testSingleViewFitIsExactWithKnownHeight() {
    let (a, _) = views()
    let pixels = Synthetic.rimKeypoints(radiusM: 0.132, heightM: 0.021, in: a, count: 64)
    let result = ContourRimFitter().fit(rimPx: pixels, keyframe: a, tablePlane: Synthetic.table, rimHeightM: 0.021)
    assertRim(result, radiusM: 0.132, heightM: 0.021)
    XCTAssertEqual(try result.get().method, .singleView)
  }

  /// Without a height, projecting onto the table overstates by D / (D − h) (§3).
  func testSingleViewWithoutHeightOverstatesAsPredicted() throws {
    let overhead = Synthetic.keyframe(id: 0, offset: SIMD3(0, 0, 1e-4), heightM: 0.40)
    let pixels = Synthetic.rimKeypoints(radiusM: 0.075, heightM: 0.072, in: overhead, count: 64)
    let rim = try ContourRimFitter().fit(rimPx: pixels, keyframe: overhead, tablePlane: Synthetic.table, rimHeightM: nil).get()
    XCTAssertEqual(rim.radiusM, 0.075 * 0.40 / (0.40 - 0.072), accuracy: toleranceM)
    XCTAssertLessThanOrEqual(rim.confidence, 0.5)
  }

  func testSingleViewFitRejectsContourOutliers() {
    let (a, _) = views()
    var pixels = Synthetic.rimKeypoints(radiusM: 0.132, heightM: 0.021, in: a, count: 64)
    var rng = SplitMix64(seed: 3)
    // 30% spurious contour pixels, e.g. food spilling over the rim.
    pixels += (0..<27).map { _ in SIMD2(Double.random(in: 600...1300, using: &rng), Double.random(in: 300...1100, using: &rng)) }
    assertRim(ContourRimFitter().fit(rimPx: pixels, keyframe: a, tablePlane: Synthetic.table, rimHeightM: 0.021),
              radiusM: 0.132, heightM: 0.021)
  }

  func testPlaneFrameRoundTrips() {
    let tilted = Plane(pointWorld: SIMD3(0.2, -0.5, 0.1), normalWorld: SIMD3(0.1, 1, -0.2))
    let point = tilted.pointWorld(at: SIMD2(0.3, -0.07), heightM: 0.04)
    XCTAssertEqual(tilted.signedDistanceM(point), 0.04, accuracy: 1e-12)
    XCTAssertLessThan(simd_distance(tilted.coordinates2D(point), SIMD2(0.3, -0.07)), 1e-12)
  }
}

extension Result {
  var failure: Failure? { if case .failure(let error) = self { return error } else { return nil } }
}
