"""Removes the dominant background plane (the floor/tabletop the object was
captured on) and any other disconnected clutter from a reconstructed point
cloud, so the final mesh is the object alone rather than object + floor.

There is no foreground/background distinction anywhere upstream of this -
SfM/MVS reconstructs everything visible in every frame with equal priority,
and deliberately so: a bare object against a blank background is a *harder*
SfM target than one with some surrounding texture to anchor camera-pose
estimation on (see docs/CAPTURE_TIPS.md's "Background" tip). The tradeoff is
that the floor or tabletop - large, flat, and visible in nearly every frame
- ends up heavily represented in the point cloud, often more so than the
object itself. This module strips it back out afterward instead.

Two-step heuristic, both standard techniques for exactly this problem:
  1. RANSAC plane segmentation removes the single largest planar surface,
     if (and only if) it's large enough to plausibly be the floor rather
     than a flat part of the object itself.
  2. DBSCAN clustering then keeps only the largest remaining connected
     cluster, on the assumption that the object is the largest contiguous
     piece of geometry left once the floor is gone.

Both steps are self-calibrating rather than using a fixed metric
threshold, since sparse-SfM and dense-MVS point clouds differ in scale by
orders of magnitude and there's no fixed real-world unit at this stage
(see the scale-calibration tool for that) - but the two steps calibrate to
different things on purpose. RANSAC's flatness tolerance is scaled to the
point cloud's overall *extent* (bounding-box diagonal), because "how flat
counts as flat" is a question of scale, not sampling density: a densely
sampled curved surface (a smoothly-scanned round object) has *tighter*
point spacing than a coarsely-sampled floor, so scaling flatness tolerance
to spacing would make curved surfaces look "flatter" than the actual floor
and get misidentified as background. DBSCAN's neighbor radius, by
contrast, genuinely is a question of sampling density (how far apart
neighboring points on the same surface typically are), so it's scaled to
the point cloud's mean nearest-neighbor spacing as usual.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

# Only treat the largest detected plane as "the floor" if it accounts for a
# clear majority-ish share of all points - evidence it's something the
# camera saw constantly (a floor/tabletop visible in nearly every frame),
# not an incidental flat patch on the object itself. Deliberately no upper
# cap: a wide, well-textured floor legitimately can be 90%+ of all points
# when the object is small relative to how much floor the camera saw, and
# that's exactly when removing it matters most. The real safety net is
# MIN_POINTS_TO_KEEP below - if too few points would survive, we bail out
# and keep everything rather than risk deleting the object along with it.
MIN_PLANE_FRACTION = 0.25
PLANE_TOLERANCE_FRACTION = 0.01  # of the point cloud's bounding-box diagonal
MIN_POINTS_TO_ATTEMPT = 100
MIN_POINTS_TO_KEEP = 50


def isolate_foreground(pcd: o3d.geometry.PointCloud) -> tuple[o3d.geometry.PointCloud, dict]:
    """Returns (filtered_point_cloud, stats). Falls back to returning the
    input unchanged (with stats explaining why) whenever the heuristics
    can't confidently identify background to remove - erring toward
    keeping everything rather than risking deleting real object geometry.
    """
    stats: dict = {
        "input_points": len(pcd.points),
        "plane_removed": False,
        "cluster_kept": False,
        "output_points": len(pcd.points),
    }
    if len(pcd.points) < MIN_POINTS_TO_ATTEMPT:
        return pcd, stats

    diagonal = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))
    plane_threshold = max(diagonal * PLANE_TOLERANCE_FRACTION, 1e-9)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=plane_threshold, ransac_n=3, num_iterations=1000
    )
    inlier_fraction = len(inliers) / len(pcd.points)

    working = pcd
    if inlier_fraction >= MIN_PLANE_FRACTION:
        candidate = pcd.select_by_index(inliers, invert=True)
        if len(candidate.points) >= MIN_POINTS_TO_KEEP:
            working = candidate
            stats["plane_removed"] = True
            stats["plane_point_fraction"] = inlier_fraction
    else:
        stats["plane_point_fraction"] = inlier_fraction

    distances = working.compute_nearest_neighbor_distance()
    avg_spacing = float(np.mean(distances)) if len(distances) else 0.01
    cluster_eps = max(avg_spacing * 3.0, 1e-9)
    labels = np.array(working.cluster_dbscan(eps=cluster_eps, min_points=10))
    if labels.size and labels.max() >= 0:
        largest_label = np.bincount(labels[labels >= 0]).argmax()
        keep_mask = labels == largest_label
        if keep_mask.sum() >= MIN_POINTS_TO_KEEP:
            working = working.select_by_index(np.nonzero(keep_mask)[0])
            stats["cluster_kept"] = True
            stats["cluster_point_fraction"] = float(keep_mask.sum()) / len(labels)

    stats["output_points"] = len(working.points)
    return working, stats
