"""App-wide configuration and paths."""
from __future__ import annotations

import os
from pathlib import Path

# Root where all session data (uploaded photos, work dirs, results) lives.
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[2] / "data")).resolve()
SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Capture guidance defaults, taken from the practical capture-tip research
# (report: "Phone-Based 3D Capture: The 2026 Landscape", section 4).
MIN_RECOMMENDED_PHOTOS = 60
MAX_RECOMMENDED_PHOTOS = 150

# Binaries / feature toggles, overridable via env for deployment flexibility.
COLMAP_BIN = os.environ.get("COLMAP_BIN", "colmap")
GSPLAT_TRAIN_SCRIPT = os.environ.get("GSPLAT_TRAIN_SCRIPT")  # optional external trainer
CUDA_EXPECTED = os.environ.get("CUDA_EXPECTED", "0") == "1"

# COLMAP defaults its *.num_threads options to -1 ("auto", i.e. one thread
# per core reported by the OS). In a lot of container platforms (Railway
# included) that reports the underlying host's full core count rather than
# the container's actual cgroup CPU quota, so COLMAP spins up hundreds of
# threads on a container that may only have 1-2 real vCPUs - it thrashes
# and typically fails outright. Cap it to a small, explicit number instead;
# raise COLMAP_NUM_THREADS if you're running on a beefier host.
COLMAP_NUM_THREADS = int(os.environ.get("COLMAP_NUM_THREADS", "4"))

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB per photo, generous for phone JPEGs
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic"}

MAX_VIDEO_UPLOAD_BYTES = 500 * 1024 * 1024  # a 20-40s phone orbit video easily runs 100MB+
ALLOWED_VIDEO_EXTS = {".webm", ".mp4", ".mov", ".m4v"}

# "Reprocess" workflow: download a session's photos to run the GPU-dependent
# pipeline on another machine (see Dockerfile.gpu), then upload the result
# back into this session so it's viewable through the same link. Result
# files are matched by exact filename against what the pipelines themselves
# produce (colmap_pipeline.py / splat_pipeline.py), so an upload here is
# indistinguishable from one this server produced itself.
ALLOWED_RESULT_FILENAMES = {"model.ply", "model.obj", "splat.ply"}
MAX_RESULT_UPLOAD_BYTES = 1024 * 1024 * 1024  # dense meshes / splats can be large
