import XCTest
import simd
@testable import MealScan

final class PlateItemsTests: XCTestCase {
  /// 200x200 overhead plate: white disc (radius 90) on a wood-colored table, with a
  /// brown "chicken" ellipse, a textured green "broccoli" blob and white textured "rice".
  func meal() -> (RGBImage, [[Bool]]) {
    let w = 200, h = 200
    var rgba = [UInt8](repeating: 255, count: w * h * 4)
    var truth = [[Bool]](repeating: [Bool](repeating: false, count: w * h), count: 3)
    var seed: UInt64 = 7
    func noise() -> Double { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Double(seed >> 33) / Double(1 << 31) - 0.5 }
    for y in 0..<h { for x in 0..<w {
      let p = SIMD2(Double(x) + 0.5, Double(y) + 0.5)
      var c = SIMD3<Double>(150, 110, 70)  // Table.
      if simd_distance(p, SIMD2(100, 100)) < 90 { c = SIMD3(236, 236, 232) + 3 * noise() }  // Plate, nearly smooth.
      let chicken = pow((p.x - 70) / 28, 2) + pow((p.y - 80) / 18, 2) < 1
      let broccoli = simd_distance(p, SIMD2(130, 75)) < 20
      let rice = simd_distance(p, SIMD2(105, 135)) < 24
      if chicken { c = SIMD3(170, 120, 70) + 8 * noise(); truth[0][y * w + x] = true }
      if broccoli { c = SIMD3(60, 120, 50) + 50 * noise(); truth[1][y * w + x] = true }
      if rice { c = SIMD3(240, 238, 228) + 70 * noise(); truth[2][y * w + x] = true }
      for k in 0..<3 { rgba[4 * (y * w + x) + k] = UInt8(max(0, min(255, c[k]))) }
    } }
    return (RGBImage(widthPx: w, heightPx: h, rgba: rgba), truth)
  }

  func circle(_ r: Double) -> [SIMD2<Double>] {
    (0..<64).map { i in let a = 2 * .pi * Double(i) / 64; return SIMD2(0.5 + r * cos(a) / 200, 0.5 + r * sin(a) / 200) }
  }

  func testFindsEachItemAndPutsPointsInside() {
    let (image, truth) = meal()
    let interior = PlateItems.rasterize(circle(84), widthPx: 200, heightPx: 200)
    let inner = PlateItems.rasterize(circle(72), widthPx: 200, heightPx: 200)
    let ring = zip(interior, inner).map { $0 && !$1 }
    let items = PlateItems.segment(image, interior: interior, ring: ring)
    XCTAssertEqual(items.count, 3, "areas: \(items.map(\.areaPx))")
    for t in truth {
      let best = items.map { item -> Double in
        let inter = zip(item.mask, t).filter { $0 && $1 }.count, union = zip(item.mask, t).filter { $0 || $1 }.count
        return Double(inter) / Double(max(1, union))
      }.max() ?? 0
      XCTAssertGreaterThan(best, 0.8)
    }
    for item in items {
      XCTAssertGreaterThanOrEqual(item.points.count, 2)
      for p in item.points {
        let i = Int(p.y * 200) * 200 + Int(p.x * 200)
        XCTAssertTrue(item.mask[i])
        XCTAssertTrue(truth.contains { $0[i] })
      }
    }
  }

  func testEmptyPlateHasNoItems() {
    var (image, _) = meal()
    for y in 0..<200 { for x in 0..<200 where simd_distance(SIMD2(Double(x), Double(y)), SIMD2(100, 100)) < 90 {
      for k in 0..<3 { image.rgba[4 * (y * 200 + x) + k] = 236 }
    } }
    let interior = PlateItems.rasterize(circle(84), widthPx: 200, heightPx: 200)
    let inner = PlateItems.rasterize(circle(72), widthPx: 200, heightPx: 200)
    XCTAssertTrue(PlateItems.segment(image, interior: interior, ring: zip(interior, inner).map { $0 && !$1 }).isEmpty)
  }

  func testPointsSpreadAcrossALongItem() {
    var mask = [Bool](repeating: false, count: 100 * 40)
    for y in 10..<30 { for x in 5..<95 { mask[y * 100 + x] = true } }
    let points = PlateItems.points(in: mask, widthPx: 100, heightPx: 40, count: 3)
    XCTAssertEqual(points.count, 3)
    let xs = points.map(\.x).sorted()
    XCTAssertGreaterThan(xs[2] - xs[0], 0.5)
  }

  func testCropOutlineRoundTrips() {
    let crop = ImageCrop(x0Px: 400, y0Px: 200, widthPx: 1000, heightPx: 800, rotatedClockwise: true)
    let p = SIMD2(0.3, 0.7)
    XCTAssertLessThan(simd_distance(crop.normalized(sensorPixel: crop.sensorPixel(normalized: p)), p), 1e-12)
  }
}
