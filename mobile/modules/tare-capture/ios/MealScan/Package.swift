// swift-tools-version:5.9
// Pure-Swift meal-scan pipeline core (tare_impl.md §5): no ARKit, so every module
// runs from plain value types on macOS (`swift test`) as well as inside the iOS pod,
// which compiles these same sources directly.
import PackageDescription

let package = Package(
  name: "MealScan",
  platforms: [.macOS(.v13), .iOS(.v15)],
  products: [.library(name: "MealScan", targets: ["MealScan"])],
  targets: [
    .target(name: "MealScan"),
    .testTarget(name: "MealScanTests", dependencies: ["MealScan"]),
  ]
)
