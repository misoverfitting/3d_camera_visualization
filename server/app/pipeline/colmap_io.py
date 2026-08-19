"""Minimal reader for COLMAP's TXT sparse-model export (cameras.txt,
images.txt, points3D.txt), used to feed camera poses / intrinsics / an
initial point cloud into the Gaussian splatting trainer without pulling in
a pycolmap dependency."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraModel:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]

    def intrinsics(self) -> np.ndarray:
        """Returns the 3x3 K matrix. Assumes PINHOLE (fx, fy, cx, fy) params,
        which is what COLMAP's image_undistorter produces."""
        if self.model != "PINHOLE":
            raise ValueError(f"Expected PINHOLE camera model, got {self.model}")
        fx, fy, cx, cy = self.params
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


@dataclass
class ImagePose:
    image_id: int
    qvec: np.ndarray  # w, x, y, z
    tvec: np.ndarray
    camera_id: int
    name: str

    def world_to_camera(self) -> np.ndarray:
        """4x4 view matrix (world -> camera), COLMAP convention."""
        w, x, y, z = self.qvec
        R = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ])
        view = np.eye(4)
        view[:3, :3] = R
        view[:3, 3] = self.tvec
        return view


def read_cameras_txt(path: Path) -> dict[int, CameraModel]:
    cameras = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        params = [float(p) for p in parts[4:]]
        cameras[camera_id] = CameraModel(camera_id, model, width, height, params)
    return cameras


def read_images_txt(path: Path) -> dict[int, ImagePose]:
    images = {}
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    # images.txt alternates: pose line, then a POINTS2D line, for each image.
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = np.array([float(p) for p in parts[1:5]])
        tvec = np.array([float(p) for p in parts[5:8]])
        camera_id = int(parts[8])
        name = parts[9]
        images[image_id] = ImagePose(image_id, qvec, tvec, camera_id, name)
    return images


def read_points3d_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (xyz [N,3], rgb [N,3] in 0..1)."""
    xyz, rgb = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb.append([int(parts[4]) / 255.0, int(parts[5]) / 255.0, int(parts[6]) / 255.0])
    return np.array(xyz, dtype=np.float64), np.array(rgb, dtype=np.float64)
