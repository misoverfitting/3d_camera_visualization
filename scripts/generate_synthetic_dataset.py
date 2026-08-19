#!/usr/bin/env python3
"""Generates a synthetic multi-view photo set of a procedurally textured
convex object, orbiting the camera at multiple heights - mimicking a good
phone capture session (see docs/CAPTURE_TIPS.md). Useful for:

  - Trying the app end-to-end without a phone/object on hand (demo mode)
  - Automated testing of the reconstruction pipeline (no OpenGL/GPU needed;
    this is a small pure-numpy rasterizer, not a real renderer)

The object is a subdivided icosphere with position-hashed vertex colors
(deliberately non-repeating, unlike a checkerboard) so structure-from-motion
has plenty of distinctive local texture to match across views. Being convex,
simple backface culling is sufficient for correct visibility - no z-buffer
needed.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def _diffused_noise(mesh: trimesh.Trimesh, channels: int, iters: int, seed: int) -> np.ndarray:
    """Random per-vertex values, spatially smoothed along the mesh graph.
    Unlike a sine-wave field (which repeats/aliases and gives SIFT many
    look-alike patches), this is aperiodic - closer to a natural mottled
    surface (granite, marble) that photogrammetry handles well."""
    rng = np.random.default_rng(seed)
    n = len(mesh.vertices)
    values = rng.random((n, channels))
    neighbors = mesh.vertex_neighbors
    for _ in range(iters):
        new_values = values.copy()
        for i, nbrs in enumerate(neighbors):
            if nbrs:
                new_values[i] = 0.4 * values[i] + 0.6 * values[nbrs].mean(axis=0)
        values = new_values
    return values


def make_object() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.5)

    # Mildly irregular (non-spherical) shape - aperiodic diffusion noise
    # rather than sine bumps, so no two regions of the surface look alike.
    bump = _diffused_noise(mesh, channels=1, iters=25, seed=1)[:, 0]
    bump = (bump - bump.mean()) * 0.18
    mesh.vertices = mesh.vertices * (1.0 + bump[:, None])
    mesh.vertices -= mesh.vertices.mean(axis=0)

    # Mottled, non-repeating vertex-color texture: a lightly-smoothed base
    # (for broad blotches) plus strong per-vertex speckle (for the sharp,
    # high-contrast local detail SIFT keypoints anchor to).
    base = _diffused_noise(mesh, channels=3, iters=1, seed=2)
    speckle = _diffused_noise(mesh, channels=3, iters=0, seed=3)
    colors = np.clip(0.45 * base + 0.55 * speckle, 0.08, 0.98)
    mesh.visual.vertex_colors = np.hstack([
        (colors * 255).astype(np.uint8), np.full((len(mesh.vertices), 1), 255, dtype=np.uint8)
    ])
    return mesh


def make_ground_plane(y: float, half_extent: float, grid: int, seed: int = 7) -> trimesh.Trimesh:
    """A textured backdrop plane (like a tabletop) under the object. Real
    phone captures always have *some* background texture in frame, which
    gives structure-from-motion extra distant features to stabilize camera
    pose estimation - an isolated object against a blank void is a harder,
    less realistic case."""
    xs = np.linspace(-half_extent, half_extent, grid)
    zs = np.linspace(-half_extent, half_extent, grid)
    xx, zz = np.meshgrid(xs, zs)
    vertices = np.stack([xx.ravel(), np.full(xx.size, y), zz.ravel()], axis=1)

    faces = []
    for r in range(grid - 1):
        for c in range(grid - 1):
            i0 = r * grid + c
            i1 = r * grid + c + 1
            i2 = (r + 1) * grid + c
            i3 = (r + 1) * grid + c + 1
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])

    plane = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)
    rng = np.random.default_rng(seed)
    base = _diffused_noise(plane, channels=3, iters=1, seed=seed)
    speckle = _diffused_noise(plane, channels=3, iters=0, seed=seed + 1)
    colors = np.clip(0.5 * base + 0.5 * speckle, 0.08, 0.9)
    plane.visual.vertex_colors = np.hstack([
        (colors * 255).astype(np.uint8), np.full((len(vertices), 1), 255, dtype=np.uint8)
    ])
    return plane


def orbit_cameras(
    distance: float, elevations_deg: list[float], steps_per_ring: int, jitter_seed: int = 42
) -> list[np.ndarray]:
    """Returns a list of 4x4 camera-to-world matrices orbiting the origin,
    with small handheld-style jitter in distance and look-at point (a
    perfectly regular geometric orbit is an easy but unrealistic special
    case; real phone capture always wobbles a little)."""
    rng = np.random.default_rng(jitter_seed)
    poses = []
    for elev in elevations_deg:
        elev_r = math.radians(elev)
        for i in range(steps_per_ring):
            az = 2 * math.pi * i / steps_per_ring
            jittered_distance = distance * (1.0 + rng.uniform(-0.12, 0.12))
            eye = jittered_distance * np.array([
                math.cos(elev_r) * math.cos(az),
                math.sin(elev_r),
                math.cos(elev_r) * math.sin(az),
            ])
            look_at = rng.uniform(-0.05, 0.05, size=3)
            forward = (look_at - eye)
            forward /= np.linalg.norm(forward)
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
            if np.linalg.norm(right) < 1e-6:
                world_up = np.array([0.0, 0.0, 1.0])
                right = np.cross(forward, world_up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)

            cam_to_world = np.eye(4)
            cam_to_world[:3, 0] = right
            cam_to_world[:3, 1] = up
            cam_to_world[:3, 2] = -forward  # camera looks down -Z
            cam_to_world[:3, 3] = eye
            poses.append(cam_to_world)
    return poses


def render_view(
    mesh: trimesh.Trimesh, cam_to_world: np.ndarray, width: int, height: int, focal: float,
    light_dir_world: np.ndarray,
) -> Image.Image:
    """Gouraud-shaded (per-pixel, barycentric-interpolated vertex color)
    software rasterizer. Smooth interpolation - rather than flat per-face
    fill - keeps the apparent texture tied to true 3D surface points and
    stable under small viewpoint changes, which is what real photographs
    (and SfM feature matching) look like."""
    world_to_cam = np.linalg.inv(cam_to_world)
    verts_h = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
    verts_cam = (world_to_cam @ verts_h.T).T[:, :3]

    cx, cy = width / 2.0, height / 2.0
    z = -verts_cam[:, 2]  # camera looks down -Z; positive depth in front
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = focal * verts_cam[:, 0] / z + cx
        sy = -focal * verts_cam[:, 1] / z + cy
    screen = np.stack([sx, sy], axis=1)

    canvas = np.full((height, width, 3), 245.0, dtype=np.float64)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float64)

    faces = mesh.faces
    normals = mesh.face_normals
    light_dir_world = light_dir_world / np.linalg.norm(light_dir_world)
    vertex_colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0

    z_faces = z[faces]
    visible = np.all(z_faces > 1e-3, axis=1)
    cam_pos_world = cam_to_world[:3, 3]
    face_centers = mesh.vertices[faces].mean(axis=1)
    to_cam = cam_pos_world[None, :] - face_centers
    to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-9
    front_facing = np.einsum("ij,ij->i", normals, to_cam) > 0
    shade = np.clip(normals @ light_dir_world, 0.55, 1.0)

    # Backface culling + a real z-buffer below: correct hidden-surface
    # removal for any (not necessarily convex) combination of meshes, e.g.
    # an object sitting in front of a background/ground plane.
    for face_idx in np.nonzero(visible & front_facing)[0]:
        tri = faces[face_idx]
        p0, p1, p2 = screen[tri]
        z0, z1, z2 = z_faces[face_idx]
        x_min = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
        x_max = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), width - 1)
        y_min = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
        y_max = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), height - 1)
        if x_max <= x_min or y_max <= y_min:
            continue

        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(denom) < 1e-9:
            continue

        xs, ys = np.meshgrid(
            np.arange(x_min, x_max + 1) + 0.5, np.arange(y_min, y_max + 1) + 0.5
        )
        w0 = ((p1[1] - p2[1]) * (xs - p2[0]) + (p2[0] - p1[0]) * (ys - p2[1])) / denom
        w1 = ((p2[1] - p0[1]) * (xs - p2[0]) + (p0[0] - p2[0]) * (ys - p2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not np.any(inside):
            continue

        depth = w0 * z0 + w1 * z1 + w2 * z2
        region_depth = depth_buffer[y_min : y_max + 1, x_min : x_max + 1]
        nearer = inside & (depth < region_depth)
        if not np.any(nearer):
            continue

        tri_colors = vertex_colors[tri] * shade[face_idx]
        pixel_color = (
            w0[..., None] * tri_colors[0]
            + w1[..., None] * tri_colors[1]
            + w2[..., None] * tri_colors[2]
        ) * 255.0

        region_depth[nearer] = depth[nearer]
        region_canvas = canvas[y_min : y_max + 1, x_min : x_max + 1]
        region_canvas[nearer] = pixel_color[nearer]

    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def generate(out_dir: Path, num_elevations: int, steps_per_ring: int, width: int, height: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = make_object()
    obj_min_y = obj.vertices[:, 1].min()
    ground = make_ground_plane(y=obj_min_y - 0.02, half_extent=4.0, grid=24)
    mesh = trimesh.util.concatenate([obj, ground])

    fov_deg = 55.0
    focal = width / (2 * math.tan(math.radians(fov_deg) / 2))
    distance = 2.2

    # Stay above the ground plane (you can't put a phone below the table).
    elevations = np.linspace(5, 60, num_elevations).tolist()
    poses = orbit_cameras(distance, elevations, steps_per_ring)
    light_dir_world = np.array([0.4, 0.8, 0.5])  # fixed light: even, non-changing (per capture tips)

    count = 0
    for i, pose in enumerate(poses):
        img = render_view(mesh, pose, width, height, focal, light_dir_world)
        img.save(out_dir / f"synthetic_{i:04d}.jpg", quality=92)
        count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--elevations", type=int, default=4)
    parser.add_argument("--steps-per-ring", type=int, default=22)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    n = generate(args.out_dir, args.elevations, args.steps_per_ring, args.width, args.height)
    print(f"Wrote {n} synthetic photos to {args.out_dir}")
