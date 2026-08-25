"""End-to-end test of the video-capture path: synthetic photos -> encoded
video -> ffmpeg frame extraction -> COLMAP reconstruction. Skipped
automatically if `colmap` or `ffmpeg` isn't on PATH.
"""
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

COLMAP_AVAILABLE = shutil.which("colmap") is not None
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_synthetic_dataset", REPO_ROOT / "scripts" / "generate_synthetic_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not COLMAP_AVAILABLE, reason="colmap binary not installed")
@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg binary not installed")
def test_video_capture_reconstructs_synthetic_object(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    import sys
    for mod in list(sys.modules):
        if mod.startswith("app"):
            del sys.modules[mod]
    from app import sessions
    from app.pipeline import colmap_pipeline, video_extract

    gen = _load_generator()
    frames_dir = tmp_path / "synthetic_frames"
    num_frames = gen.generate(frames_dir, num_elevations=3, steps_per_ring=16, width=640, height=480)
    assert num_frames == 48

    # Encode the frames into a video at a realistic orbit pacing (a phone
    # recording at ~2 frames/sec of *distinct* viewpoints while orbiting
    # slowly), simulating what actually reaches the server.
    video_path = tmp_path / "orbit.mp4"
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", "2",
            "-i", str(frames_dir / "synthetic_%04d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    session = sessions.create_session("video pytest object")
    extracted = video_extract.extract_frames(video_path, session.photos_dir)
    assert extracted > 30, "too few frames survived extraction"

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
    assert meta["photo_count"] == extracted
