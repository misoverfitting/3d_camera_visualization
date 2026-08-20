"""Shared COLMAP stages used by both the 'accurate' (mesh) and 'compelling'
(Gaussian splat) pipelines: feature extraction, matching, sparse SfM, and
undistortion to a clean PINHOLE camera model."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..jobs import ProgressReporter
from ..sessions import Session

MIN_PHOTOS_FOR_SFM = 8


class PipelineError(RuntimeError):
    pass


def run(cmd: list[str], step: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PipelineError(f"{step} failed:\n{proc.stderr.strip()[-4000:]}")
    return proc


def dense_stereo_available() -> bool:
    proc = subprocess.run([config.COLMAP_BIN, "-h"], capture_output=True, text=True)
    return "without CUDA" not in (proc.stdout + proc.stderr)


@dataclass
class SfmResult:
    undistorted_images_dir: Path
    undistorted_model_dir: Path  # PINHOLE sparse model, TXT format
    dense_workspace_dir: Path


def run_sfm_and_undistort(session: Session, reporter: ProgressReporter) -> SfmResult:
    images_dir = session.photos_dir
    photo_count = session.photo_count()
    if photo_count < MIN_PHOTOS_FOR_SFM:
        raise PipelineError(
            f"Only {photo_count} photos uploaded; need at least {MIN_PHOTOS_FOR_SFM} "
            f"(60-150+ recommended) for reliable reconstruction."
        )

    work = session.work_dir
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    db_path = work / "database.db"
    sparse_dir = work / "sparse"
    sparse_dir.mkdir()

    num_threads = str(config.COLMAP_NUM_THREADS)

    reporter.update("feature_extraction", "Extracting features from photos", 5)
    run([
        config.COLMAP_BIN, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.num_threads", num_threads,
    ], "Feature extraction")

    reporter.update("matching", "Matching features across photos", 20)
    run([
        config.COLMAP_BIN, "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "0",
        "--SiftMatching.num_threads", num_threads,
    ], "Feature matching")

    reporter.update("sparse_reconstruction", "Estimating camera poses and sparse geometry (SfM)", 35)
    run([
        config.COLMAP_BIN, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.num_threads", num_threads,
    ], "Sparse reconstruction")

    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        raise PipelineError(
            "Structure-from-motion failed to register any images. Try photos with "
            "more overlap, better lighting, and a more textured/matte object surface."
        )

    dense_dir = work / "dense"
    reporter.update("undistort", "Undistorting images to a clean pinhole camera model", 50)
    run([
        config.COLMAP_BIN, "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(model_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ], "Image undistortion")

    undistorted_model_dir = dense_dir / "sparse"
    undistorted_model_txt_dir = dense_dir / "sparse_txt"
    undistorted_model_txt_dir.mkdir(exist_ok=True)
    run([
        config.COLMAP_BIN, "model_converter",
        "--input_path", str(undistorted_model_dir),
        "--output_path", str(undistorted_model_txt_dir),
        "--output_type", "TXT",
    ], "Model conversion to TXT")

    return SfmResult(
        undistorted_images_dir=dense_dir / "images",
        undistorted_model_dir=undistorted_model_txt_dir,
        dense_workspace_dir=dense_dir,
    )
