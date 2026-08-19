"""Fast tests for the session/photo/job HTTP API, independent of COLMAP."""
import io
import shutil

import pytest
from fastapi.testclient import TestClient


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


def test_upload_rejects_bad_extension(client):
    session_id = client.post("/api/sessions", json={"name": "t"}).json()["id"]
    resp = client.post(
        f"/api/sessions/{session_id}/photos",
        files={"file": ("notes.txt", b"hello", "text/plain")},
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
