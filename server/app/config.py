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

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB per photo, generous for phone JPEGs
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic"}
