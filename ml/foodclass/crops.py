"""Item crops exactly like the app's (LocalFoodID.itemCrop), plus augmentations that make
training crops look like what the app's segmenter actually produces.

The app pads the region's box by 12.5% per side and dims everything outside the mask to
35%. Its masks are noisy: they bleed onto the plate, miss parts of the food, swallow a
neighbouring food or cover only part of a pile. Training on clean ground-truth masks
only teaches the clean case, so `perturb` roughens them first.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter

PADDING = 0.125
DIM = 0.35


def box_of(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]


def crop(image: Image.Image, box: list[float], mask: np.ndarray | None, padding: float = PADDING,
         dim: float = DIM) -> Image.Image:
    """Padded crop around the item, background dimmed (not blacked out)."""
    w, h = image.size
    x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    pad_x, pad_y = (x1 - x0) * padding, (y1 - y0) * padding
    left, top = max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y))
    right, bottom = min(w, int(x1 + pad_x + 1)), min(h, int(y1 + pad_y + 1))
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    if mask is None:
        mask = np.zeros((h, w), dtype=bool)
        mask[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True
    dimmed = np.where(mask[..., None], pixels, pixels * dim)
    return Image.fromarray(dimmed[top:bottom, left:right].astype(np.uint8))


def _morph(mask: np.ndarray, radius: int, grow: bool) -> np.ndarray:
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.filter(ImageFilter.MaxFilter(size) if grow else ImageFilter.MinFilter(size))
    return np.asarray(image) > 127


def _blob(shape: tuple[int, int], rng: random.Random, scale: float) -> np.ndarray:
    """A random smooth blob (low-frequency noise thresholded)."""
    h, w = shape
    coarse = np.asarray([[rng.random() for _ in range(6)] for _ in range(6)], dtype=np.float32)
    field = np.asarray(Image.fromarray((coarse * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    return field > 1 - scale


def perturb(mask: np.ndarray, others: np.ndarray | None, rng: random.Random) -> np.ndarray:
    """Roughens a clean item mask the way the app's color/texture segmenter gets it wrong.

    `others` marks other food on the same plate (for merges with a neighbour)."""
    out = mask.copy()
    h, w = mask.shape
    side = max(1, int(np.sqrt(mask.sum())))
    r = rng.random()
    if r < 0.25:  # Bleeds onto the plate around the food.
        out = _morph(out, max(1, int(side * rng.uniform(0.03, 0.12))), grow=True)
    elif r < 0.45:  # Misses the food's edges (shading, sauce).
        eroded = _morph(out, max(1, int(side * rng.uniform(0.03, 0.1))), grow=False)
        out = eroded if eroded.sum() > 0.3 * mask.sum() else out
    if others is not None and others.any() and rng.random() < 0.15:  # Swallows a touching neighbour.
        near = _morph(out, max(2, int(side * 0.15)), grow=True) & others
        out |= _morph(near, max(2, int(side * 0.2)), grow=True) & others
    if rng.random() < 0.2:  # Only part of the pile (color clusters split it).
        part = out & _blob((h, w), rng, rng.uniform(0.4, 0.7))
        out = part if part.sum() > 0.25 * mask.sum() else out
    if rng.random() < 0.15:  # Holes (highlights, garnish).
        out &= ~_blob((h, w), rng, rng.uniform(0.05, 0.15))
    return out if out.any() else mask


def app_crop(image: Image.Image, mask: np.ndarray, others: np.ndarray | None, rng: random.Random) -> Image.Image:
    """Training crop: perturbed mask, slightly varied padding and dimming."""
    noisy = perturb(mask, others, rng)
    box = box_of(noisy) or box_of(mask)
    return crop(image, box, noisy, padding=rng.uniform(0.08, 0.2), dim=rng.uniform(0.28, 0.45))


def plate_dim(image: Image.Image, rng: random.Random) -> Image.Image:
    """Whole-dish photo made to look like the app's whole-plate input: an elliptical
    'plate interior' kept bright, the rest dimmed, cropped to the ellipse plus padding."""
    w, h = image.size
    cx, cy = w * rng.uniform(0.4, 0.6), h * rng.uniform(0.4, 0.6)
    rx, ry = w * rng.uniform(0.3, 0.5), h * rng.uniform(0.3, 0.5)
    ys, xs = np.mgrid[0:h, 0:w]
    mask = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1
    box = box_of(mask)
    return crop(image, box, mask, padding=PADDING, dim=rng.uniform(0.28, 0.45))
