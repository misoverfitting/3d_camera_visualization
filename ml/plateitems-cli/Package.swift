// swift-tools-version:5.9
// Runs the app's plate item segmentation (MealScan.PlateItems) on photo folders, so
// training data gets the same item proposals the phone makes. macOS only (uses Vision
// foreground masks to find the plate when no rim is known).
import PackageDescription

let package = Package(
  name: "plateitems",
  platforms: [.macOS(.v14)],
  dependencies: [.package(path: "../../mobile/modules/tare-capture/ios/MealScan")],
  targets: [.executableTarget(name: "plateitems", dependencies: [.product(name: "MealScan", package: "MealScan")])]
)
