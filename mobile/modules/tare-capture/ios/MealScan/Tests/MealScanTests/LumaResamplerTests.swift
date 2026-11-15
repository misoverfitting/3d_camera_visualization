import XCTest
@testable import MealScan

final class LumaResamplerTests: XCTestCase {
  func testPaddedRowsAndFractionalScalePreserveMean() throws {
    let bytes: [UInt8] = [0, 10, 20, 255, 30, 40, 50, 255]
    let image = try XCTUnwrap(bytes.withUnsafeBufferPointer {
      LumaResampler.downsample($0, width: 3, height: 2, bytesPerRow: 4, targetWidth: 2)
    })
    XCTAssertEqual(image.widthPx, 2)
    XCTAssertEqual(image.heightPx, 1)
    XCTAssertEqual(image.pixels[0], 18.333333, accuracy: 0.0001)
    XCTAssertEqual(image.pixels[1], 31.666667, accuracy: 0.0001)
  }

  func testCommonSensorFormatsDoNotDropFrames() throws {
    for (w, h) in [(1920, 1440), (1280, 720), (640, 480)] {
      let bytes = [UInt8](repeating: 73, count: w * h)
      let image = try XCTUnwrap(bytes.withUnsafeBufferPointer {
        LumaResampler.downsample($0, width: w, height: h, bytesPerRow: w, targetWidth: 480)
      })
      XCTAssertEqual(image.widthPx, 480)
      XCTAssertEqual(image.heightPx, Int((Double(h) * 480 / Double(w)).rounded()))
      XCTAssertTrue(image.pixels.allSatisfy { abs($0 - 73) < 0.0001 })
      XCTAssertEqual(image.sharpness, 0, accuracy: 0.0001)
    }
  }

  func testInvalidBufferIsRejected() {
    let bytes: [UInt8] = [0, 1]
    XCTAssertNil(bytes.withUnsafeBufferPointer {
      LumaResampler.downsample($0, width: 3, height: 2, bytesPerRow: 3, targetWidth: 2)
    })
  }
}
