"""Pose-diverse, sharp keyframes without modifying imported images/calibration."""
from __future__ import annotations
import numpy as np


def select_frames(data, photos, limit):
    frames = data['frames']
    if not limit or len(frames) <= limit:
        return frames
    from PIL import Image
    from scipy.ndimage import laplace
    poses = np.array([np.array(f['camera_to_world']).reshape(4, 4, order='F') for f in frames])
    positions, directions = poses[:, :3, 3], -poses[:, :3, 2]
    span = max(float(np.linalg.norm(np.ptp(positions, axis=0))), 0.01)
    distances = (np.linalg.norm(positions[:, None] - positions[None, :], axis=2) / span +
                 np.arccos(np.clip(directions @ directions.T, -1, 1)) / np.pi)
    sharpness = []
    for frame in frames:
        with Image.open(photos / frame['image']) as source:
            image = source.convert('L')
            image.thumbnail((320, 320))
            sharpness.append(float(laplace(np.asarray(image, dtype=float)).var()))
    sharpness = np.array(sharpness)
    quality = np.clip(sharpness / max(float(np.median(sharpness)), 1e-6), 0.1, 2)
    # Keep endpoints for sequence coverage and orbit seam evidence.
    selected = [0, len(frames) - 1]
    while len(selected) < limit:
        score = distances[:, selected].min(axis=1) * np.sqrt(quality)
        score[selected] = -1
        selected.append(int(np.argmax(score)))
    return [frames[i] for i in sorted(selected)]
