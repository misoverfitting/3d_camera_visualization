import XCTest
import simd
@testable import MealScan

/// LiDAR modules are limited by depth sampling (~2 mm per depth pixel at 40 cm), so
/// their tolerances are millimetres, not the 0.1 mm of the analytic geometry tests.
final class LidarTests: XCTestCase {
  // Dinner plate with a 35 mm-radius, 25 mm-tall food puck on its floor.
  static let puckCenter = SIMD2(0.03, 0.0), puckRadiusM = 0.035, puckTopM = 0.006 + 0.025
  static let plateScene = Synthetic.plate(radiusM: 0.132, rimHeightM: 0.021) { p in
    simd_distance(p, puckCenter) <= puckRadiusM ? puckTopM : nil
  }
  static let bowlScene = Synthetic.bowl(radiusM: 0.075, rimHeightM: 0.072, floorM: 0.012)

  static let plateKeyframes = [
    Synthetic.keyframe(id: 0, offset: SIMD3(0.0, 0, 0.06), depth: plateScene),
    Synthetic.keyframe(id: 1, offset: SIMD3(0.11, 0, 0.06), depth: plateScene),
  ]
  static let bowlKeyframes = [
    Synthetic.keyframe(id: 0, offset: SIMD3(0.0, 0, 0.03), depth: bowlScene),
    Synthetic.keyframe(id: 1, offset: SIMD3(0.10, 0, 0.03), depth: bowlScene),
  ]

  func container(radiusM: Double, heightM: Double, type: ContainerType) -> ContainerGeometry {
    ContainerGeometry(type: type, tablePlane: Synthetic.table, rim: RimEstimate(
      centerWorld: Synthetic.dishCenter + SIMD3(0, heightM, 0), radiusM: radiusM, heightM: heightM,
      radiusUncertaintyM: nil, heightUncertaintyM: nil, rmsResidualPx: 0, method: .twoView, viewCount: 2, confidence: 1))
  }

  func testDepthUnprojectionLandsOnScene() throws {
    let depth = try XCTUnwrap(Self.plateKeyframes[0].depth)
    let points = depth.pointsWorld(minConfidence: 2)
    XCTAssertGreaterThan(points.count, depth.widthPx * depth.heightPx * 9 / 10)
    let center = Synthetic.table.coordinates2D(Synthetic.dishCenter)
    let worst = points.map { abs(Synthetic.table.signedDistanceM($0) - Self.plateScene.heightM(Synthetic.table.coordinates2D($0) - center)) }
    // Only rim-edge samples may straddle the vertical drop; the bulk must be exact.
    XCTAssertLessThan(try XCTUnwrap(Numerics.percentile(worst, 0.98)), 1e-5)
  }

  func testRimReferenceMeasuresPlate() throws {
    let reference = try XCTUnwrap(LidarRimReference().estimate(
      keyframes: Self.plateKeyframes, tablePlane: Synthetic.table,
      approximateCenterWorld: Synthetic.dishCenter + SIMD3(0.01, 0, -0.01), searchRadiusM: 0.25))
    XCTAssertEqual(reference.radiusM, 0.132, accuracy: 0.003)
    XCTAssertEqual(reference.heightM, 0.021, accuracy: 0.002)
    XCTAssertLessThan(simd_distance(Synthetic.table.coordinates2D(reference.centerWorld),
                                    Synthetic.table.coordinates2D(Synthetic.dishCenter)), 0.002)
  }

  func testRimReferenceMeasuresBowlAndScoresAnEstimate() throws {
    let reference = try XCTUnwrap(LidarRimReference().estimate(
      keyframes: Self.bowlKeyframes, tablePlane: Synthetic.table,
      approximateCenterWorld: Synthetic.dishCenter, searchRadiusM: 0.2))
    XCTAssertEqual(reference.radiusM, 0.075, accuracy: 0.003)
    XCTAssertEqual(reference.heightM, 0.072, accuracy: 0.002)
    let estimate = container(radiusM: 0.078, heightM: 0.07, type: .bowl).rim
    let comparison = reference.compare(estimate)
    XCTAssertEqual(comparison.diameterErrorM, 0.156 - 2 * reference.radiusM, accuracy: 1e-12)
    XCTAssertEqual(comparison.heightErrorM, 0.07 - reference.heightM, accuracy: 1e-12)
  }

  func puckItem(in keyframe: Keyframe) -> ItemSegment {
    // Item mask at a model-like 480 x 360: pixels whose ray meets the puck's top disc.
    let intrinsics = keyframe.intrinsics.scaled(toWidthPx: 480, heightPx: 360)
    let maskFrame = Keyframe(id: keyframe.id, timestampS: 0, intrinsics: intrinsics, worldFromCamera: keyframe.worldFromCamera)
    let top = Synthetic.table.offset(byM: Self.puckTopM)
    let center = Synthetic.table.coordinates2D(Synthetic.dishCenter) + Self.puckCenter
    let mask = ImageMask(widthPx: 480, heightPx: 360) { x, y in
      guard let hit = top.intersection(maskFrame.rayWorld(throughPx: SIMD2(Double(x), Double(y)))) else { return false }
      return simd_distance(Synthetic.table.coordinates2D(hit), center) <= Self.puckRadiusM
    }
    return ItemSegment(id: 1, keyframeId: keyframe.id, mask: mask, label: "puck")
  }

  func testItemHeightAboveRimPlaneAndTable() throws {
    let keyframe = Self.plateKeyframes[0]
    let item = puckItem(in: keyframe)
    let plate = container(radiusM: 0.132, heightM: 0.021, type: .plate)
    let rimHeight = try XCTUnwrap(LidarItemHeightProvider().height(for: item, container: plate, keyframes: Self.plateKeyframes))
    XCTAssertEqual(rimHeight.reference, .rimPlane)
    XCTAssertEqual(rimHeight.source, .lidar)
    XCTAssertEqual(rimHeight.valueM, Self.puckTopM - 0.021, accuracy: 0.001)
    XCTAssertGreaterThan(rimHeight.sampleCount, 100)
    let tableHeight = try XCTUnwrap(LidarItemHeightProvider(reference: .table)
      .height(for: item, container: plate, keyframes: Self.plateKeyframes))
    XCTAssertEqual(tableHeight.valueM, Self.puckTopM, accuracy: 0.001)
    XCTAssertEqual(try XCTUnwrap(tableHeight.meanSurfaceM), Self.puckTopM, accuracy: 0.001)
  }

  func testItemHeightNeedsDepthInTheSegmentedKeyframe() {
    var keyframes = Self.plateKeyframes
    let item = puckItem(in: keyframes[0])
    keyframes[0].depth = nil
    XCTAssertNil(LidarItemHeightProvider().height(for: item, container: container(radiusM: 0.132, heightM: 0.021, type: .plate),
                                                  keyframes: keyframes))
  }

  func testLowConfidenceDepthIsExcluded() throws {
    var keyframes = Self.plateKeyframes
    var depth = try XCTUnwrap(keyframes[0].depth)
    depth.confidence = [UInt8](repeating: 1, count: depth.depthM.count)  // All medium.
    keyframes[0].depth = depth
    let item = puckItem(in: keyframes[0])
    XCTAssertNil(LidarItemHeightProvider().height(for: item, container: container(radiusM: 0.132, heightM: 0.021, type: .plate),
                                                  keyframes: keyframes))
  }

  func testBowlProfileRecoversInteriorShape() throws {
    let bowl = container(radiusM: 0.075, heightM: 0.072, type: .bowl)
    let profile = try XCTUnwrap(LidarContainerProfileProvider().profile(for: bowl, keyframes: Self.bowlKeyframes))
    XCTAssertEqual(profile.source, .lidar)
    XCTAssertTrue(profile.measuresVisibleSurface)
    // Deepest bin is near the center; the analytic floor is 60 mm below the rim.
    XCTAssertEqual(profile.innerDepthM, 0.072 - 0.012, accuracy: 0.003)
    for sample in profile.samples {
      let expected = 0.072 - (0.012 + 0.06 * sample.radiusM * sample.radiusM / (0.075 * 0.075))
      XCTAssertEqual(sample.depthBelowRimM, expected, accuracy: 0.003, "radius \(sample.radiusM)")
    }
    XCTAssertGreaterThanOrEqual(profile.samples.count, 10)
  }
}
