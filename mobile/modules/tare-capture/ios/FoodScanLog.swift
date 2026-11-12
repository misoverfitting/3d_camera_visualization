import CoreGraphics
import Foundation
import ImageIO
import UIKit
import UniformTypeIdentifiers
import ZIPFoundation

// Field log for on-device food ID: every identification saves what the model saw to
// Documents/FoodScans/<time>/, so failures can be sorted into segmentation vs. classifier
// mistakes, and the frames reused for training without labeling anything by hand.
//   frame.jpg      full camera frame (sensor orientation) used for the photo
//   plate.jpg      upright plate crop the segmenter and classifier ran on
//   overlay.jpg    plate.jpg with region i tinted palette color i, the bare-plate ring
//                  blue and everything outside the searched interior darkened
//   plate-input.jpg  whole-plate classifier input
//   region-<i>.jpg classifier input for region i
//   scan.json      predictions, boxes, points, model version
// Visible in the Files app (UIFileSharingEnabled) and shared as one zip from the app.

enum FoodScanLog {
  static var root: URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("FoodScans", isDirectory: true)
  }

  static var count: Int {
    (try? FileManager.default.contentsOfDirectory(atPath: root.path).filter { !$0.hasPrefix(".") }.count) ?? 0
  }

  /// Returns the folder name, or nil when saving failed (never fails identification).
  @discardableResult
  static func save(frameJPEG: Data, plate: CGImage, scan: LocalFoodScan, info: [String: Any]) -> String? {
    let stamp = DateFormatter()
    stamp.locale = Locale(identifier: "en_US_POSIX")
    stamp.dateFormat = "yyyyMMdd-HHmmss"
    let name = stamp.string(from: Date())
    let folder = root.appendingPathComponent(name, isDirectory: true)
    do {
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      try frameJPEG.write(to: folder.appendingPathComponent("frame.jpg"))
      write(plate, to: folder.appendingPathComponent("plate.jpg"))
      if let overlay = scan.overlay { write(overlay, to: folder.appendingPathComponent("overlay.jpg")) }
      if let input = scan.plateInput { write(input, to: folder.appendingPathComponent("plate-input.jpg")) }
      for (i, region) in scan.regions.enumerated() { write(region.crop, to: folder.appendingPathComponent("region-\(i).jpg")) }

      func ranked(_ list: [(String, Double)]) -> [[String: Any]] { list.map { ["label": $0.0, "p": $0.1] } }
      var json = info
      json["time"] = ISO8601DateFormatter().string(from: Date())
      json["dish"] = ranked(scan.dish)
      json["wholePlate"] = ranked(scan.wholePlate)
      json["regions"] = scan.regions.enumerated().map { i, region -> [String: Any] in
        ["index": i, "color": FoodState.palette[i % FoodState.palette.count].1,
         "box": [region.box.x, region.box.y, region.box.z, region.box.w], "areaFraction": region.areaFraction,
         "top": ranked(region.topK), "points": region.points.map { [$0.x, $0.y] }]
      }
      json["items"] = scan.items.map { ["label": $0.label, "category": $0.category, "confidence": $0.confidence, "points": $0.points.count] }
      let data = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
      try data.write(to: folder.appendingPathComponent("scan.json"))
      return name
    } catch {
      return nil
    }
  }

  /// All saved scans as one zip in the temporary directory.
  static func zip() throws -> URL {
    guard count > 0 else { throw RimTapException("No food scans saved yet.") }
    let out = FileManager.default.temporaryDirectory.appendingPathComponent("FoodScans-\(UIDevice.current.name.filter { $0.isLetter || $0.isNumber }).zip")
    if FileManager.default.fileExists(atPath: out.path) { try FileManager.default.removeItem(at: out) }
    try FileManager.default.zipItem(at: root, to: out)
    return out
  }

  private static func write(_ image: CGImage, to url: URL) {
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
    CGImageDestinationFinalize(destination)
  }
}
