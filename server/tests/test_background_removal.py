"""Fast, deterministic unit tests for background_removal.py against
synthetic point clouds - no COLMAP/ffmpeg needed. Exercises the actual
real-world failure mode directly: a floor/tabletop that dominates the
point count, which a fixed "don't remove more than X% of points" safety
cap would (and once did - see git history) refuse to remove even though
that's exactly the case where removing it matters most.
"""
import numpy as np
import open3d as o3d

from app.pipeline import background_removal


def _floor_and_object_cloud(seed=0, floor_points=4000, object_points=400, stray_points=50):
    rng = np.random.default_rng(seed)
    floor_xy = rng.uniform(-3, 3, size=(floor_points, 2))
    floor_pts = np.column_stack([floor_xy, rng.normal(0, 0.002, size=floor_points)])
    object_pts = rng.normal(loc=[0, 0, 0.4], scale=0.15, size=(object_points, 3))
    stray_pts = rng.uniform(-3, 3, size=(stray_points, 3)) + np.array([0, 0, 2.0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([floor_pts, object_pts, stray_pts]))
    return pcd


def test_removes_dominant_floor_even_when_it_is_most_of_the_cloud():
    # The floor here is ~90% of all points - a realistic ratio for a small
    # object shot with a lot of visible tabletop, and the exact case a
    # naive "don't remove more than 85%" cap would refuse to touch.
    pcd = _floor_and_object_cloud()
    filtered, stats = background_removal.isolate_foreground(pcd)

    assert stats["plane_removed"] is True
    assert stats["plane_point_fraction"] > 0.8

    pts = np.asarray(filtered.points)
    assert len(pts) < len(pcd.points) * 0.2, "floor should have been removed, not kept"
    # Almost everything left should be in the object's z-band, not the
    # floor (z~0) or the stray cluster (z~2).
    in_object_band = (pts[:, 2] > 0.1) & (pts[:, 2] < 0.7)
    assert np.mean(in_object_band) > 0.9


def test_plane_tolerance_scales_with_extent_not_density():
    # A curved *surface* (a sphere shell, sampled uniformly over its area
    # via normalize-a-Gaussian-vector - not a solid blob, since a filled 3D
    # Gaussian volume has way more RANSAC-plane-friendly structure through
    # its middle than a real hollow surface would) is densely sampled but
    # not flat. The regression this guards: an earlier version scaled
    # RANSAC's flatness tolerance to mean point *spacing*, so this densely-
    # sampled curved surface got a tight-seeming absolute tolerance that
    # was actually generous relative to its small radius, and a slice
    # through it got misidentified as a plane. Scaling tolerance to the
    # cloud's overall *extent* instead fixes that: curvature over a small
    # object is large relative to the object's own size, however densely
    # it's sampled.
    rng = np.random.default_rng(1)
    directions = rng.normal(size=(2000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radius = 0.3 + rng.normal(0, 0.01, size=(2000, 1))
    sphere_pts = directions * radius

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sphere_pts)

    filtered, stats = background_removal.isolate_foreground(pcd)
    assert stats["plane_removed"] is False
    assert stats["plane_point_fraction"] < background_removal.MIN_PLANE_FRACTION
    # No plane removed, and the whole sphere is one connected surface, so
    # almost nothing should be discarded (DBSCAN may still drop a handful
    # of points in sparser regions as noise - that's expected, not a bug).
    assert len(filtered.points) > len(pcd.points) * 0.98


def test_bails_out_safely_on_tiny_point_clouds():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.default_rng(2).normal(size=(30, 3)))
    filtered, stats = background_removal.isolate_foreground(pcd)
    assert len(filtered.points) == 30
    assert stats["plane_removed"] is False
