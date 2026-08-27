"""CPU-only meshing helpers, used when COLMAP's CUDA-based dense stereo is
unavailable. Builds a surface directly from the sparse SfM point cloud via
Open3D's Poisson reconstruction. This is intentionally a fallback: sparse
SfM points are much less dense than true multi-view stereo output, so the
resulting mesh is smoother/coarser than what a GPU-equipped deployment
would produce (see README for the GPU-recommended path).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from . import background_removal


def mesh_from_point_cloud(ply_path: Path, out_path: Path, poisson_depth: int = 9) -> tuple[Path, dict]:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) < 50:
        raise RuntimeError(
            f"Only {len(pcd.points)} 3D points reconstructed - too sparse to mesh. "
            "Capture more overlapping photos of a well-textured, matte surface."
        )

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd, bg_stats = background_removal.isolate_foreground(pcd)
    if len(pcd.points) < 50:
        raise RuntimeError(
            f"Only {len(pcd.points)} points left after removing background - too "
            "sparse to mesh. Capture more overlapping photos of the object."
        )

    distances = pcd.compute_nearest_neighbor_distance()
    avg_spacing = float(np.mean(distances)) if len(distances) else 0.01
    radius = max(avg_spacing * 3.0, 1e-6)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )

    # Poisson extrapolates a watertight surface even far from any sample;
    # trim the lowest-density (least-supported) vertices to cut that away.
    densities = np.asarray(densities)
    low_density_threshold = np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(densities < low_density_threshold)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()

    o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)
    return out_path, bg_stats


def convert_to_obj(ply_path: Path, out_path: Path) -> Path | None:
    """Best-effort OBJ export for tools that don't read PLY (vertex colors
    are dropped since plain OBJ has no standard way to carry them)."""
    try:
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        if len(mesh.vertices) == 0:
            return None
        o3d.io.write_triangle_mesh(str(out_path), mesh)
        return out_path
    except Exception:
        return None
