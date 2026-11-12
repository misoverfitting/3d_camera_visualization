import XCTest
import simd
@testable import MealScan

final class KeyframeSelectorTests: XCTestCase {
  func testSelectsOverheadThenSideways() {
    var selector = KeyframeSelector()
    // ~45° oblique view is too tilted for keyframe A (limit 30°).
    XCTAssertEqual(selector.consider(Synthetic.keyframe(id: 0, offset: SIMD3(0.4, 0, 0))), .rejected(.tooTilted))
    var limited = Synthetic.keyframe(id: 1, offset: SIMD3(0, 0, 0.05))
    limited.tracking = .limited
    XCTAssertEqual(selector.consider(limited), .rejected(.trackingLimited))
    var blurry = Synthetic.keyframe(id: 2, offset: SIMD3(0, 0, 0.05))
    blurry.sharpness = 10
    XCTAssertEqual(selector.consider(blurry), .rejected(.tooBlurry))
    XCTAssertEqual(selector.consider(Synthetic.keyframe(id: 3, offset: SIMD3(0, 0, 0.05))), .acceptedA)
    XCTAssertEqual(selector.consider(Synthetic.keyframe(id: 4, offset: SIMD3(0.06, 0, 0.05))), .rejected(.insufficientBaseline))
    XCTAssertFalse(selector.isComplete)
    XCTAssertEqual(selector.consider(Synthetic.keyframe(id: 5, offset: SIMD3(0.11, 0, 0.05))), .acceptedB)
    XCTAssertTrue(selector.isComplete)
    XCTAssertEqual(selector.keyframeA?.id, 3)
    XCTAssertEqual(selector.keyframeB?.id, 5)
  }

  func testOverheadTilt() {
    XCTAssertEqual(KeyframeSelector.overheadTiltDeg(Synthetic.keyframe(id: 0, offset: SIMD3(0, 0, 1e-6))), 0, accuracy: 0.01)
    XCTAssertEqual(KeyframeSelector.overheadTiltDeg(Synthetic.keyframe(id: 0, offset: SIMD3(0.4, 0, 0), heightM: 0.4)), 45, accuracy: 0.01)
  }
}
