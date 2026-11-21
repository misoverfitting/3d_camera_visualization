import XCTest
import simd
@testable import MealScan

final class SupportPlacementTests: XCTestCase {
  func testPlateCenterUsesSelectedTableInsteadOfElevatedHit() throws {
    let table = Plane(pointWorld: SIMD3(0, 0.7, 0), normalWorld: SIMD3(0, 1, 0))
    let ray = Ray(originWorld: SIMD3(0.1, 1.1, 0.2), directionWorld: SIMD3(0, -1, 0))
    let target = try XCTUnwrap(SupportPlacement.target(ray: ray, tablePlane: table))
    XCTAssertEqual(target.y, 0.7, accuracy: 1e-9)
    XCTAssertEqual(target.x, 0.1, accuracy: 1e-9)
    XCTAssertEqual(target.z, 0.2, accuracy: 1e-9)
    // A separate estimated plane on food would put the seed six cm too high.
    let food = table.offset(byM: 0.06)
    XCTAssertEqual(try XCTUnwrap(food.intersection(ray)).y - target.y, 0.06, accuracy: 1e-9)
  }

  func testHeightAdjustmentMovesTargetOnSameCameraRay() throws {
    let table = Plane(pointWorld: SIMD3(0, 0.7, 0), normalWorld: SIMD3(0, 1, 0))
    let ray = Ray(originWorld: SIMD3(0, 1.1, 0.2), directionWorld: SIMD3(0.2, -1, -0.3))
    let original = try XCTUnwrap(SupportPlacement.target(ray: ray, tablePlane: table))
    let adjusted = try XCTUnwrap(SupportPlacement.target(ray: ray, tablePlane: table.offset(byM: -0.005)))
    XCTAssertEqual(adjusted.y, original.y - 0.005, accuracy: 1e-9)
    XCTAssertEqual(simd_length(simd_cross(adjusted - ray.originWorld, ray.directionWorld)), 0, accuracy: 1e-9)
  }

  func testWrongFacingAndDistantTableCannotStartCapture() {
    let table = Plane(pointWorld: .zero, normalWorld: SIMD3(0, 1, 0))
    XCTAssertNil(SupportPlacement.target(ray: Ray(originWorld: SIMD3(0, 0.4, 0), directionWorld: SIMD3(0, 1, 0)), tablePlane: table))
    XCTAssertNil(SupportPlacement.target(ray: Ray(originWorld: SIMD3(0, 2, 0), directionWorld: SIMD3(0, -1, 0)), tablePlane: table))
  }

  func testFoodPileDoesNotRaiseUnmeasuredPlateSpots() {
    let table = Plane(pointWorld: .zero, normalWorld: SIMD3(0, 1, 0))
    let rim = RimEstimate(centerWorld: SIMD3(0, 0.02, 0), radiusM: 0.13, heightM: 0.02,
                          radiusUncertaintyM: nil, heightUncertaintyM: nil, rmsResidualPx: 0,
                          method: .orbitEdges, viewCount: 6, confidence: 1)
    let r = rim.radiusM - PipelineConfig.Probe().insetM
    let food = [SIMD3(r, 0.06, 0), SIMD3(r + 0.001, 0.06, 0), SIMD3(r, 0.06, 0.001)]
    let spots = ProbeRing.spots(rim: rim, tablePlane: table, surfacePointsWorld: food)
    XCTAssertTrue(spots.contains { $0.measuredSurface && $0.surfaceHeightM > rim.heightM })
    XCTAssertTrue(spots.filter { !$0.measuredSurface }.allSatisfy { $0.surfaceHeightM <= rim.heightM })
    let noDepth = ProbeRing.spots(rim: rim, tablePlane: table)
    XCTAssertTrue(noDepth.allSatisfy { !$0.measuredSurface && abs($0.surfaceHeightM - 0.008) < 1e-9 })
  }
}
