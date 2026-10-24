import Foundation
import simd

/// `ItemHeightProvider` from LiDAR depth (tare_impl.md §6.15). Measures the visible
/// top surface inside the item's mask, in the keyframe where it was segmented.
/// Top-side only: item sides and undersides are never assumed visible.
public struct LidarItemHeightProvider: ItemHeightProvider {
  public var config: PipelineConfig.Lidar
  public var reference: HeightReference

  public init(config: PipelineConfig.Lidar = .init(), reference: HeightReference = .rimPlane) {
    self.config = config; self.reference = reference
  }

  public func height(for item: ItemSegment, container: ContainerGeometry, keyframes: [Keyframe]) -> ItemHeight? {
    guard let depth = keyframes.first(where: { $0.id == item.keyframeId })?.depth else { return nil }
    let points = depth.pointsWorld(insideMask: item.mask, erosionPx: config.maskErosionPx, minConfidence: config.minConfidence)
    let referenceM = reference == .rimPlane ? container.rim.heightM : 0
    let heightsM = points.map { container.tablePlane.signedDistanceM($0) - referenceM }
    guard heightsM.count >= 10,
          let topM = Numerics.percentile(heightsM, config.topSurfacePercentile),
          let q1 = Numerics.percentile(heightsM, 0.25), let q3 = Numerics.percentile(heightsM, 0.75)
    else { return nil }
    // Sensor noise plus the surface spread's standard error (IQR / 1.349 ≈ σ). Initial
    // guess at an uncertainty model; replace once golden-set residuals exist.
    let spreadM = (q3 - q1) / 1.349
    let uncertaintyM = sqrt(config.depthNoiseM * config.depthNoiseM + spreadM * spreadM / Double(heightsM.count))
    return ItemHeight(valueM: topM, uncertaintyM: uncertaintyM, reference: reference, source: .lidar,
                      meanSurfaceM: heightsM.reduce(0, +) / Double(heightsM.count), sampleCount: heightsM.count)
  }
}
