"""Turns a short orbit video into a well-chosen set of sharp still frames
for photogrammetry - this is what actually gets fed to COLMAP.

Tapping a shutter button 60-150 times turned out to be the biggest friction
point in the app, and manual taps also tend to have uneven, gappy coverage.
Recording a slow ~20-40s orbit is much easier to do well, but raw video
frames are frequently motion-blurred (phones moving during capture) and
give you far more frames than you need, many near-duplicates.

Approach: extract candidate frames at a moderate fixed rate via ffmpeg, then
divide them into evenly spaced time buckets covering the whole video and
keep only the sharpest frame (by variance-of-Laplacian, a standard blur
metric) in each bucket. This directly fights motion blur, spaces the final
set evenly across the whole orbit, and keeps frame count in the range SfM
wants without the user having to count anything.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .. import config

CANDIDATE_FPS = 6.0  # raw extraction rate, before sharpness-based thinning
MIN_CANDIDATE_FRAMES = 16


class VideoExtractError(RuntimeError):
    pass


def _run(cmd: list[str], step: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoExtractError(f"{step} failed:\n{proc.stderr.strip()[-2000:]}")
    return proc


def _probe_duration_seconds(video_path: Path) -> float:
    proc = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(video_path),
    ], "Video probing")
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise VideoExtractError(f"Could not read video duration: {exc}") from exc


def _sharpness(image_path: Path) -> float:
    """Variance of the Laplacian - low for blurry/flat images, high for
    sharp, detailed ones. Cheap, well-established blur metric."""
    with Image.open(image_path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.float64)
    return float(ndimage.laplace(gray).var())


def extract_frames(
    video_path: Path,
    out_dir: Path,
    target_frames: int | None = None,
) -> int:
    """Extracts up to `target_frames` sharp, evenly-spaced frames from
    `video_path` into `out_dir` (cleared first), named so alphabetical order
    matches capture order. Returns the number of frames written.

    If `target_frames` is omitted, it scales with video length (~3
    frames/second of footage) clamped to the recommended photo-count range,
    so a longer, more thorough orbit yields proportionally more frames
    instead of always capping at the same number.
    """
    duration = _probe_duration_seconds(video_path)
    if duration < 2.0:
        raise VideoExtractError(
            f"Video is only {duration:.1f}s long - record at least a few seconds "
            "of slow orbiting around the object."
        )

    if target_frames is None:
        target_frames = int(np.clip(
            round(duration * 3), config.MIN_RECOMMENDED_PHOTOS, config.MAX_RECOMMENDED_PHOTOS
        ))

    for existing in out_dir.glob("*"):
        existing.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"fps={CANDIDATE_FPS}",
            "-qscale:v", "2",
            str(tmp_dir / "candidate_%05d.jpg"),
        ], "Frame extraction")

        candidates = sorted(tmp_dir.glob("candidate_*.jpg"))
        if len(candidates) < MIN_CANDIDATE_FRAMES:
            raise VideoExtractError(
                f"Only extracted {len(candidates)} candidate frames from a "
                f"{duration:.1f}s video - try recording a longer, slower orbit."
            )

        num_buckets = min(target_frames, len(candidates))
        buckets = np.array_split(np.arange(len(candidates)), num_buckets)

        written = 0
        for i, bucket in enumerate(buckets):
            if len(bucket) == 0:
                continue
            best_idx = max(bucket, key=lambda idx: _sharpness(candidates[idx]))
            dest = out_dir / f"frame_{written:05d}.jpg"
            shutil.copy(candidates[best_idx], dest)
            written += 1

    return written
