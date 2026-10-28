import XCTest
import simd
@testable import MealScan

final class FoodPointsTests: XCTestCase {
  let rim = RimEstimate(centerWorld: Synthetic.dishCenter + SIMD3(0, 0.02, 0), radiusM: 0.13, heightM: 0.02,
                        radiusUncertaintyM: nil, heightUncertaintyM: nil, rmsResidualPx: 0,
                        method: .orbitEdges, viewCount: 12, confidence: 1)

  func testRotatedCropMapsBackToSensorPixels() {
    let crop = ImageCrop(x0Px: 400, y0Px: 200, widthPx: 1000, heightPx: 800, rotatedClockwise: true)
    // Upright top-left = sensor crop's bottom-left; upright top-right = sensor top-left.
    XCTAssertEqual(crop.sensorPixel(normalized: SIMD2(0, 0)), SIMD2(399.5, 999.5))
    XCTAssertEqual(crop.sensorPixel(normalized: SIMD2(1, 0)), SIMD2(399.5, 199.5))
    XCTAssertEqual(crop.sensorPixel(normalized: SIMD2(0, 1)), SIMD2(1399.5, 999.5))
    let plain = ImageCrop(x0Px: 400, y0Px: 200, widthPx: 1000, heightPx: 800, rotatedClockwise: false)
    XCTAssertEqual(plain.sensorPixel(normalized: SIMD2(0.5, 0.5)), SIMD2(899.5, 599.5))
  }

  func testCropContainsTheRim() throws {
    let keyframe = Synthetic.keyframe(id: 0, offset: SIMD3(0.02, 0, 0.03))
    let crop = try XCTUnwrap(ImageCrop.around(rim, tablePlane: Synthetic.table, in: keyframe, margin: 1.15, rotatedClockwise: true))
    for pixel in Synthetic.rimKeypoints(radiusM: 0.13, heightM: 0.02, in: keyframe) {
      XCTAssertTrue(pixel.x >= crop.x0Px - 0.5 && pixel.x <= crop.x0Px + crop.widthPx)
      XCTAssertTrue(pixel.y >= crop.y0Px - 0.5 && pixel.y <= crop.y0Px + crop.heightPx)
    }
  }

  func testLiftUsesDepthOnTheDishAndThePlaneOtherwise() throws {
    let scene = Synthetic.plate(radiusM: 0.13, rimHeightM: 0.02) { p in simd_distance(p, SIMD2(0.03, 0)) <= 0.03 ? 0.04 : nil }
    var keyframe = Synthetic.keyframe(id: 0, offset: SIMD3(0.02, 0, 0.03), depth: scene)
    let target = Synthetic.table.pointWorld(at: Synthetic.table.coordinates2D(Synthetic.dishCenter) + SIMD2(0.03, 0), heightM: 0.04)
    let pixel = try XCTUnwrap(keyframe.projectPx(target))
    let lifted = try XCTUnwrap(FoodPointLifter.lift(pixel, keyframe: keyframe, rim: rim, tablePlane: Synthetic.table, surfaceHeightM: 0.008))
    XCTAssertTrue(lifted.fromDepth)
    XCTAssertLessThan(simd_distance(lifted.positionWorld, target), 0.003)

    keyframe.depth = nil
    let planar = try XCTUnwrap(FoodPointLifter.lift(pixel, keyframe: keyframe, rim: rim, tablePlane: Synthetic.table, surfaceHeightM: 0.008))
    XCTAssertFalse(planar.fromDepth)
    XCTAssertEqual(Synthetic.table.signedDistanceM(planar.positionWorld), 0.008, accuracy: 1e-9)

    // A point off the plate is dropped.
    let off = try XCTUnwrap(keyframe.projectPx(Synthetic.dishCenter + SIMD3(0.25, 0, 0)))
    XCTAssertNil(FoodPointLifter.lift(off, keyframe: keyframe, rim: rim, tablePlane: Synthetic.table, surfaceHeightM: 0.008))
  }
}
