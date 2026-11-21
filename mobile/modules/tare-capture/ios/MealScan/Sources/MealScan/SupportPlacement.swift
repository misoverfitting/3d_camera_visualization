import simd

/// A plate tap selects a ray on an already selected support plane. It must not
/// introduce another estimated plane through food, the plate lip, or the floor.
public enum SupportPlacement {
  public static func target(ray: Ray, tablePlane: Plane,
                            minDistanceM: Double = 0.15, maxDistanceM: Double = 1.0) -> SIMD3<Double>? {
    guard let point = tablePlane.intersection(ray) else { return nil }
    let distance = simd_distance(point, ray.originWorld)
    guard distance.isFinite, distance >= minDistanceM, distance <= maxDistanceM else { return nil }
    return point
  }
}
