"""Scale calibration: photogrammetry has no inherent sense of scale, so we
let the user pick two points on the reconstructed model corresponding to a
reference object of known real-world size, then uniformly rescale the mesh
so distances become metric (see report section 4, "Scale").
"""
from __future__ import annotations

import math
from pathlib import Path

import open3d as o3d


def apply_scale_calibration(
    mesh_path: Path,
    out_path: Path,
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
    real_world_distance_m: float,
) -> float:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0:
        raise ValueError("Mesh has no vertices to scale")

    model_distance = math.dist(point_a, point_b)
    if model_distance <= 1e-9:
        raise ValueError("Selected points are coincident; pick two distinct points")

    scale_factor = real_world_distance_m / model_distance
    mesh.scale(scale_factor, center=(0.0, 0.0, 0.0))
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)
    return scale_factor
