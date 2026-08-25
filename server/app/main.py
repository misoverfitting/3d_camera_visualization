from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, jobs, sessions
from .models import JobProgress, ReconstructRequest, ScaleCalibration, SessionCreate, SessionInfo
from .pipeline import colmap_pipeline, scale, splat_pipeline, video_extract

app = FastAPI(title="Phone 3D Capture", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/sessions", response_model=SessionInfo)
def create_session(body: SessionCreate):
    session = sessions.create_session(body.name)
    return session.to_info()


@app.get("/api/sessions", response_model=list[SessionInfo])
def list_sessions():
    return [s.to_info() for s in sessions.list_sessions()]


@app.get("/api/sessions/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    try:
        session = sessions.get_session(session_id)
    except sessions.SessionNotFound:
        raise HTTPException(404, "Session not found")
    return session.to_info()


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    try:
        sessions.delete_session(session_id)
    except sessions.SessionNotFound:
        raise HTTPException(404, "Session not found")
    return {"ok": True}


@app.post("/api/sessions/{session_id}/photos")
async def upload_photo(session_id: str, file: UploadFile):
    try:
        session = sessions.get_session(session_id)
    except sessions.SessionNotFound:
        raise HTTPException(404, "Session not found")

    ext = Path(file.filename or "photo.jpg").suffix.lower()
    if ext not in config.ALLOWED_IMAGE_EXTS:
        raise HTTPException(400, f"Unsupported file type {ext!r}")

    contents = await file.read()
    if len(contents) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, "Photo too large")

    dest = session.photos_dir / f"{session.photo_count():05d}{ext}"
    dest.write_bytes(contents)
    return {"filename": dest.name, "photo_count": session.photo_count()}


@app.post("/api/sessions/{session_id}/video")
def upload_video(session_id: str, file: UploadFile):
    """Accepts one orbit-video recording and replaces the session's photo
    set with sharp frames extracted from it (see pipeline/video_extract.py).
    Runs synchronously (this is a sync `def`, so FastAPI offloads it to a
    worker thread) - a 20-40s video extracts in a few seconds, not long
    enough to need the job-polling flow reconstruction uses."""
    try:
        session = sessions.get_session(session_id)
    except sessions.SessionNotFound:
        raise HTTPException(404, "Session not found")

    ext = Path(file.filename or "capture.mp4").suffix.lower()
    if ext not in config.ALLOWED_VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported video type {ext!r}")

    contents = file.file.read()
    if len(contents) > config.MAX_VIDEO_UPLOAD_BYTES:
        raise HTTPException(400, "Video too large")

    session.work_dir.mkdir(parents=True, exist_ok=True)
    video_path = session.work_dir / f"capture{ext}"
    video_path.write_bytes(contents)

    try:
        frame_count = video_extract.extract_frames(video_path, session.photos_dir)
    except video_extract.VideoExtractError as exc:
        raise HTTPException(400, str(exc))
    finally:
        video_path.unlink(missing_ok=True)

    return {"photo_count": frame_count}


@app.get("/api/sessions/{session_id}/photos")
def list_photos(session_id: str):
    try:
        session = sessions.get_session(session_id)
    except sessions.SessionNotFound:
        raise HTTPException(404, "Session not found")
    return {"photos": sorted(p.name for p in session.photos_dir.iterdir() if p.is_file())}


@app.get("/api/sessions/{session_id}/photos/{filename}")
def get_photo(session_id: str, filename: str):
    try:
        session = sessions.get_session(session_id)
    except sessions.SessionNotFound:
        raise HTTPException(404, "Session not found")
    path = (session.photos_dir / filename).resolve()
    if session.photos_dir.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "Photo not found")
    return FileResponse(path)


@app.delete("/api/sessions/{session_id}/photos/{filename}")
def delete_photo(session_id: str, filename: str):
    try:
        session = sessions.get_session(session_id)
    except sessions.SessionNotFound:
        raise HTTPException(404, "Session not found")
    path = (session.photos_dir / filename).resolve()
    if session.photos_dir.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "Photo not found")
    path.unlink()
    return {"photo_count": session.photo_count()}


def _pipeline_for(mode: str):
    if mode == "accurate":
        return colmap_pipeline.run_accurate_pipeline
    if mode == "compelling":
        return splat_pipeline.run_compelling_pipeline
    raise HTTPException(400, f"Unknown mode {mode!r}")


@app.post("/api/sessions/{session_id}/reconstruct")
def reconstruct(session_id: str, body: ReconstructRequest, background_tasks: BackgroundTasks):
    try:
        session = sessions.get_session(session_id)
    except sessions.SessionNotFound:
        raise HTTPException(404, "Session not found")

    pipeline_fn = _pipeline_for(body.mode)
    background_tasks.add_task(jobs.run_job, session, pipeline_fn)
    return {"status": "queued", "mode": body.mode}


@app.get("/api/sessions/{session_id}/job", response_model=JobProgress)
def get_job(session_id: str):
    try:
        session = sessions.get_session(session_id)
    except sessions.SessionNotFound:
        raise HTTPException(404, "Session not found")
    return jobs.read_job(session)


@app.get("/api/sessions/{session_id}/result/{filename}")
def get_result(session_id: str, filename: str):
    try:
        session = sessions.get_session(session_id)
    except sessions.SessionNotFound:
        raise HTTPException(404, "Session not found")
    path = (session.output_dir / filename).resolve()
    if session.output_dir.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "Result not found")
    return FileResponse(path, filename=filename)


@app.post("/api/sessions/{session_id}/scale")
def calibrate_scale(session_id: str, body: ScaleCalibration):
    try:
        session = sessions.get_session(session_id)
    except sessions.SessionNotFound:
        raise HTTPException(404, "Session not found")

    mesh_path = session.output_dir / "model.ply"
    if not mesh_path.exists():
        raise HTTPException(400, "No mesh available to scale yet")

    scale_factor = scale.apply_scale_calibration(
        mesh_path, mesh_path, body.point_a, body.point_b, body.real_world_distance_m
    )
    obj_path = session.output_dir / "model.obj"
    if obj_path.exists():
        scale.apply_scale_calibration(
            obj_path, obj_path, body.point_a, body.point_b, body.real_world_distance_m
        )
    return {"scale_factor": scale_factor}


# Serve the frontend as static files.
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
