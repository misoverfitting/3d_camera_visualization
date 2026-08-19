"""End-to-end test of the 'accurate' (COLMAP) pipeline against a synthetic,
procedurally rendered multi-view photo set (no phone/GPU/OpenGL required -
see scripts/generate_synthetic_dataset.py). Skipped automatically if the
`colmap` binary isn't on PATH.

This is intentionally the one slow/integration-style test in the suite
(runs real COLMAP feature extraction, matching, and SfM); everything else
in test_api.py runs fast, without COLMAP.
"""
import importlib.util
import shutil
from pathlib import Path

import pytest

COLMAP_AVAILABLE = shutil.which("colmap") is not None
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_synthetic_dataset", REPO_ROOT / "scripts" / "generate_synthetic_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not COLMAP_AVAILABLE, reason="colmap binary not installed")
def test_accurate_pipeline_reconstructs_synthetic_object(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    import sys
    for mod in list(sys.modules):
        if mod.startswith("app"):
            del sys.modules[mod]
    from app import sessions
    from app.pipeline import colmap_pipeline

    gen = _load_generator()
    photos_dir = tmp_path / "synthetic_photos"
    num_photos = gen.generate(photos_dir, num_elevations=3, steps_per_ring=16, width=640, height=480)
    assert num_photos == 48

    session = sessions.create_session("synthetic pytest object")
    for photo in sorted(photos_dir.glob("*.jpg")):
        shutil.copy(photo, session.photos_dir)

    class _Reporter:
        def update(self, stage, message, percent):
            pass

    result_files = colmap_pipeline.run_accurate_pipeline(session, _Reporter())

    assert "model.ply" in result_files
    mesh_path = session.output_dir / "model.ply"
    assert mesh_path.exists() and mesh_path.stat().st_size > 0

    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    assert len(mesh.vertices) > 100, "reconstruction produced a degenerate/near-empty mesh"

    import json
    meta = json.loads((session.output_dir / "reconstruction.json").read_text())
    assert meta["photo_count"] == num_photos
