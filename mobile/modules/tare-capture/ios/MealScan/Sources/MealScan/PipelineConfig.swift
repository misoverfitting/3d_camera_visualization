/// Every pipeline tunable (tare_impl.md §13). Each value cites its source: a
/// section of tare_impl.md, an eval run ID, or "initial guess". Replace guesses
/// with eval-backed values and update the citation.
public struct PipelineConfig: Sendable {
  public struct Keyframes: Sendable {
    /// Max angle between keyframe A's view direction and straight down. Source: §6.1.
    public var maxOverheadTiltDeg = 30.0
    /// Min camera travel between keyframes A and B. Source: §6.1.
    public var minBaselineM = 0.10
    /// Min sharpness (variance of Laplacian on 8-bit luma). Source: initial guess; calibrate per device.
    public var minSharpness = 60.0

    public init() {}
  }

  public struct Rim: Sendable {
    /// Rim height search range for the triangulator's initialization. Source: initial guess (deep bowls ~10 cm).
    public var maxRimHeightM = 0.15
    /// Initialization grid step. Source: initial guess.
    public var heightSearchStepM = 0.005
    /// Levenberg-Marquardt iteration cap. Source: initial guess.
    public var maxIterations = 100
    /// Min keypoints with nonzero weight per view. Source: initial guess.
    public var minKeypointsPerView = 6
    /// RMS reprojection residual at which confidence halves. Source: initial guess.
    public var confidenceHalfResidualPx = 1.5
    /// Visible rim fraction below which confidence drops. Source: §11 ("visible arc < 60%").
    public var minVisibleFraction = 0.6
    /// RANSAC inlier distance for single-view circle fitting. Source: initial guess.
    public var ransacInlierM = 0.004
    public var ransacIterations = 200

    public init() {}
  }

  public struct Lidar: Sendable {
    /// ARConfidenceLevel raw value required for depth samples (2 = high). Source: initial guess.
    public var minConfidence: UInt8 = 2
    /// Depth pixels removed inside mask edges, avoiding mixed/flying pixels. Source: initial guess.
    public var maskErosionPx = 1
    /// Height above the table that counts as part of the dish. Source: initial guess; LiDAR noise ~ few mm.
    public var elevatedThresholdM = 0.004
    /// Angular bins for the ground-truth rim boundary. Source: initial guess.
    public var rimBoundaryBins = 72
    /// Width of the annulus inside the rim used for rim height. Source: initial guess.
    public var rimAnnulusM = 0.012
    /// Upper height percentile for "top surface" (robust to spikes). Source: initial guess.
    public var topSurfacePercentile = 0.9
    /// Lower percentile for the visible interior surface in profiles. Source: initial guess.
    public var profilePercentile = 0.1
    /// Radial bins in container profiles. Source: initial guess.
    public var profileBins = 12
    /// Per-sample depth noise used in uncertainties. Source: initial guess; measure on the golden set.
    public var depthNoiseM = 0.003

    public init() {}
  }

  public var keyframes = Keyframes()
  public var rim = Rim()
  public var lidar = Lidar()

  public init() {}
}
