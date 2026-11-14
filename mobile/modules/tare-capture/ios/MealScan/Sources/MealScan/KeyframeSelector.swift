import Foundation
import simd

/// Chooses keyframes A (near-overhead) and B (after a sideways move) from the live
/// stream (tare_impl.md §6.1). Not yet covered: "dish fully in frame", which needs
/// the container segmenter's output.
public struct KeyframeSelector {
  public enum Rejection: String, Sendable { case trackingLimited, tooBlurry, tooTilted, insufficientBaseline }
  public enum Decision: Equatable, Sendable { case acceptedA, acceptedB, rejected(Rejection) }

  public var config: PipelineConfig.Keyframes
  public private(set) var keyframeA: Keyframe?
  public private(set) var keyframeB: Keyframe?

  public init(config: PipelineConfig.Keyframes = .init()) { self.config = config }

  public var isComplete: Bool { keyframeB != nil }

  public mutating func reset() { keyframeA = nil; keyframeB = nil }

  /// Angle between the camera's view direction and straight down (world -Y).
  public static func overheadTiltDeg(_ keyframe: Keyframe) -> Double {
    acos(min(1, max(-1, simd_dot(keyframe.forwardWorld, SIMD3(0, -1, 0))))) * 180 / .pi
  }

  public mutating func consider(_ keyframe: Keyframe) -> Decision {
    guard keyframe.tracking == .normal else { return .rejected(.trackingLimited) }
    if let sharpness = keyframe.sharpness, sharpness < config.minSharpness { return .rejected(.tooBlurry) }
    guard let a = keyframeA else {
      guard Self.overheadTiltDeg(keyframe) <= config.maxOverheadTiltDeg else { return .rejected(.tooTilted) }
      keyframeA = keyframe
      return .acceptedA
    }
    guard simd_distance(a.cameraCenterWorld, keyframe.cameraCenterWorld) >= config.minBaselineM else {
      return .rejected(.insufficientBaseline)
    }
    keyframeB = keyframe
    return .acceptedB
  }
}
