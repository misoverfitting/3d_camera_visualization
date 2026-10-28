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

  /// Keyframes sampled while the user orbits the dish (demo capture flow).
  public struct Orbit: Sendable {
    /// Max keyframes kept; each costs ~0.7 MB at the fitting resolution. Source: initial guess.
    public var maxKeyframes = 32
    /// Min angle between view directions (seen from the dish) of accepted keyframes. Source: initial guess.
    public var minAngleDeg = 8.0
    /// Min camera elevation above the table, seen from the dish. Source: initial guess.
    public var minElevationDeg = 15.0
    /// Motion-blur guards from consecutive ARKit poses. Source: initial guess.
    public var maxAngularSpeedDegS = 45.0
    public var maxSpeedMS = 0.35
    /// Camera distance to the dish. Source: initial guess (arm's length).
    public var minDistanceM = 0.15
    public var maxDistanceM = 1.0
    /// The dish must project inside this centered fraction of the image. Source: initial guess.
    public var maxCenterOffsetFraction = 0.8
    /// Coverage grid over the hemisphere: azimuth bins x elevation bands (degrees). Source: initial guess.
    public var azimuthBins = 12
    public var elevationBandsDeg: [Double] = [15, 45, 90]

    public init() {}
  }

  /// Automatic rim fitting from image edges across an orbit. Stands in for the learned
  /// RimDetector (§6.4) and, like the triangulator, never reads depth.
  public struct EdgeRim: Sendable {
    /// Center search half-width around the seed (a tap on the dish). Source: initial guess.
    public var centerSearchM = 0.06
    /// Rim radius range: saucer to large platter. Source: initial guess.
    public var minRadiusM = 0.05
    public var maxRadiusM = 0.18
    /// Rim height search range and step. Source: initial guess (deep bowls ~8 cm).
    public var maxHeightM = 0.08
    public var heightStepM = 0.004
    /// Vote histogram bin in the rim plane. Source: initial guess (~1 px at 480 px over 40 cm).
    public var binM = 0.002
    /// Coarse search steps. Source: initial guess.
    public var coarseCenterStepM = 0.006
    public var coarseRadiusStepM = 0.004
    /// Gaussian blur before gradients, in fitting-image pixels. Source: initial guess.
    public var blurSigmaPx = 1.2
    /// Fraction of a view's pixels kept as edge votes (strongest gradients). Source: initial guess.
    public var edgeFraction = 0.06
    /// The rim is the outermost ring scoring at least this fraction of the best ring. Source: initial guess.
    public var outermostScoreFraction = 0.55
    /// Ring samples per view when scoring. Source: initial guess.
    public var ringSamples = 96
    /// Refinement is limited to this window around the coarse estimate. Source: initial guess.
    public var refineWindowM = 0.01
    /// Min views for a fit. Source: initial guess.
    public var minViews = 4

    public init() {}
  }

  /// Spectrometer probe spots on a ring inside the rim (demo).
  public struct Probe: Sendable {
    /// Inset from the rim: about one inch. Source: product request (2026-10-28).
    public var insetM = 0.0254
    public var count = 8
    /// Surface height used when no 3D points exist near a spot (typical plate floor). Source: initial guess.
    public var fallbackSurfaceHeightM = 0.008
    /// Horizontal radius for collecting surface points around a spot. Source: initial guess.
    public var surfaceSearchRadiusM = 0.012

    public init() {}
  }

  public var keyframes = Keyframes()
  public var rim = Rim()
  public var lidar = Lidar()
  public var orbit = Orbit()
  public var edgeRim = EdgeRim()
  public var probe = Probe()

  public init() {}
}
