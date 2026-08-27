"""Fast tests for the session/photo/job HTTP API, independent of COLMAP."""
import io
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # config.SESSIONS_DIR is computed at import time, so make sure a fresh
    # import picks up the patched DATA_DIR.
    import sys
    for mod in list(sys.modules):
        if mod.startswith("app"):
            del sys.modules[mod]
    from app.main import app
    return TestClient(app)


def _tiny_jpeg_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def test_session_lifecycle(client):
    resp = client.post("/api/sessions", json={"name": "My Object"})
    assert resp.status_code == 200
    session = resp.json()
    assert session["name"] == "My Object"
    assert session["photo_count"] == 0
    session_id = session["id"]

    resp = client.get(f"/api/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == session_id

    resp = client.get("/api/sessions")
    assert any(s["id"] == session_id for s in resp.json())

    resp = client.delete(f"/api/sessions/{session_id}")
    assert resp.status_code == 200
    resp = client.get(f"/api/sessions/{session_id}")
    assert resp.status_code == 404


def test_photo_upload_and_delete(client):
    session_id = client.post("/api/sessions", json={"name": "t"}).json()["id"]

    jpeg = _tiny_jpeg_bytes()
    resp = client.post(
        f"/api/sessions/{session_id}/photos",
        files={"file": ("photo1.jpg", jpeg, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["photo_count"] == 1

    resp = client.post(
        f"/api/sessions/{session_id}/photos",
        files={"file": ("photo2.jpg", jpeg, "image/jpeg")},
    )
    assert resp.json()["photo_count"] == 2

    resp = client.get(f"/api/sessions/{session_id}/photos")
    photos = resp.json()["photos"]
    assert len(photos) == 2

    resp = client.delete(f"/api/sessions/{session_id}/photos/{photos[0]}")
    assert resp.status_code == 200
    assert resp.json()["photo_count"] == 1


def test_reprocess_download_and_upload_round_trip(client):
    # Download half: bundle a session's photos for reprocessing elsewhere.
    session_id = client.post("/api/sessions", json={"name": "t"}).json()["id"]
    jpeg = _tiny_jpeg_bytes()
    for name in ("a.jpg", "b.jpg"):
        client.post(f"/api/sessions/{session_id}/photos", files={"file": (name, jpeg, "image/jpeg")})

    resp = client.get(f"/api/sessions/{session_id}/photos.zip")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    import zipfile
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert sorted(zf.namelist()) == ["00000.jpg", "00001.jpg"]

    # Upload half: a locally-reprocessed result comes back and the session
    # is marked done with it, as if this server had produced it.
    fake_ply = b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"
    resp = client.post(
        f"/api/sessions/{session_id}/result",
        files={"file": ("model.ply", fake_ply, "application/octet-stream")},
    )
    assert resp.status_code == 200
    assert resp.json()["result_files"] == ["model.ply"]

    job = client.get(f"/api/sessions/{session_id}/job").json()
    assert job["status"] == "done"
    assert job["result_files"] == ["model.ply"]

    result = client.get(f"/api/sessions/{session_id}/result/model.ply")
    assert result.status_code == 200
    assert result.content == fake_ply


def test_reprocess_zip_rejects_empty_session(client):
    session_id = client.post("/api/sessions", json={"name": "t"}).json()["id"]
    resp = client.get(f"/api/sessions/{session_id}/photos.zip")
    assert resp.status_code == 400


def test_reprocess_upload_rejects_unexpected_filename(client):
    session_id = client.post("/api/sessions", json={"name": "t"}).json()["id"]
    resp = client.post(
        f"/api/sessions/{session_id}/result",
        files={"file": ("not_a_real_result.ply", b"whatever", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_upload_rejects_bad_extension(client):
    session_id = client.post("/api/sessions", json={"name": "t"}).json()["id"]
    resp = client.post(
        f"/api/sessions/{session_id}/photos",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_video_upload_rejects_bad_extension(client):
    session_id = client.post("/api/sessions", json={"name": "t"}).json()["id"]
    resp = client.post(
        f"/api/sessions/{session_id}/video",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg binary not installed")
def test_video_upload_extracts_frames(client, tmp_path):
    video_path = tmp_path / "test.mp4"
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=8:size=320x240:rate=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    session_id = client.post("/api/sessions", json={"name": "t"}).json()["id"]
    with open(video_path, "rb") as f:
        resp = client.post(
            f"/api/sessions/{session_id}/video",
            files={"file": ("orbit.mp4", f, "video/mp4")},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["photo_count"] > 0

    photos = client.get(f"/api/sessions/{session_id}/photos").json()["photos"]
    assert len(photos) == resp.json()["photo_count"]


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg binary not installed")
def test_video_upload_handles_webm_with_no_duration_metadata(client, tmp_path):
    # Regression test: browser MediaRecorder output (WebM) frequently has no
    # Duration written in its Segment Info at all - ffprobe's
    # format=duration then has nothing to report. Reproduce that by muxing
    # to an unseekable pipe, which forces ffmpeg to skip the trailing
    # Duration/Cues it would otherwise backpatch into a seekable file.
    video_path = tmp_path / "no_duration.webm"
    with open(video_path, "wb") as out:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=8:size=320x240:rate=24",
                "-c:v", "libvpx", "-pix_fmt", "yuv420p", "-f", "webm", "pipe:1",
            ],
            stdout=out, stderr=subprocess.PIPE, text=True,
        )
    assert proc.returncode == 0, proc.stderr

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video_path)],
        capture_output=True, text=True,
    )
    assert '"duration"' not in probe.stdout, "test video unexpectedly has duration metadata"

    session_id = client.post("/api/sessions", json={"name": "t"}).json()["id"]
    with open(video_path, "rb") as f:
        resp = client.post(
            f"/api/sessions/{session_id}/video",
            files={"file": ("orbit.webm", f, "video/webm")},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["photo_count"] > 0


def test_video_upload_rejects_unparseable_video(client, tmp_path):
    # A file with a video extension but not actually a video (ffprobe can't
    # read a duration from it) should 400 with a clear message, not 500.
    from PIL import Image
    img_path = tmp_path / "not_really_a_video.jpg"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(img_path)
    session_id = client.post("/api/sessions", json={"name": "t"}).json()["id"]
    with open(img_path, "rb") as f:
        resp = client.post(
            f"/api/sessions/{session_id}/video",
            files={"file": ("orbit.mp4", f, "video/mp4")},
        )
    assert resp.status_code == 400


def test_reconstruct_unknown_session_404s(client):
    resp = client.post("/api/sessions/doesnotexist/reconstruct", json={"mode": "accurate"})
    assert resp.status_code == 404


def test_job_status_before_reconstruction(client):
    session_id = client.post("/api/sessions", json={"name": "t"}).json()["id"]
    resp = client.get(f"/api/sessions/{session_id}/job")
    assert resp.status_code == 200
    assert resp.json()["status"] == "new"


def test_path_traversal_blocked(client):
    session_id = client.post("/api/sessions", json={"name": "t"}).json()["id"]
    resp = client.get(f"/api/sessions/{session_id}/photos/..%2f..%2fsession.json")
    assert resp.status_code in (404, 400)
