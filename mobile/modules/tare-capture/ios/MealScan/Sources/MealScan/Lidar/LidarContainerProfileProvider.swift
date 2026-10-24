import Foundation
import simd

/// `ContainerProfileProvider` from LiDAR depth (tare_impl.md §6.8). Bins depth points
/// inside the rim by distance from the rim center and takes a low height percentile
/// per bin, i.e. the visible interior surface. With food present that surface is
/// partly food (`measuresVisibleSurface`); an empty-container capture gives the
/// true cross-section.
public struct LidarContainerProfileProvider: ContainerProfileProvider {
  public var config: PipelineConfig.Lidar

  public init(config: PipelineConfig.Lidar = .init()) { self.config = config }

  public func profile(for container: ContainerGeometry, keyframes: [Keyframe]) -> ContainerProfile? {
    let table = container.tablePlane, rim = container.rim
    let center = table.coordinates2D(rim.centerWorld)
    // Stay inside the rim lip so the rim's own top does not enter the outer bin.
    let maxRadiusM = rim.radiusM - config.rimAnnulusM
    guard maxRadiusM > 0 else { return nil }
    let binWidthM = maxRadiusM / Double(config.profileBins)
    var bins = [[Double]](repeating: [], count: config.profileBins)
    for keyframe in keyframes {
      guard let depth = keyframe.depth else { continue }
      for p in depth.pointsWorld(minConfidence: config.minConfidence) {
        let radiusM = simd_distance(table.coordinates2D(p), center)
        guard radiusM < maxRadiusM else { continue }
        bins[min(config.profileBins - 1, Int(radiusM / binWidthM))].append(table.signedDistanceM(p))
      }
    }
    let samples: [ContainerProfile.Sample] = bins.enumerated().compactMap { index, heightsM in
      guard heightsM.count >= 5, let surfaceM = Numerics.percentile(heightsM, config.profilePercentile) else { return nil }
      return .init(radiusM: (Double(index) + 0.5) * binWidthM, depthBelowRimM: rim.heightM - surfaceM, sampleCount: heightsM.count)
    }
    guard let innerDepthM = samples.map(\.depthBelowRimM).max() else { return nil }
    return ContainerProfile(samples: samples, innerDepthM: innerDepthM, source: .lidar, measuresVisibleSurface: true)
  }
}
