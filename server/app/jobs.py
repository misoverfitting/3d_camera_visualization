"""Reconstruction job tracking.

Jobs run in a background thread (started from a FastAPI BackgroundTask) and
report progress by writing job.json in the session directory, so status can
be polled cheaply without holding anything in server memory.
"""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from typing import Callable

from .sessions import Session

_locks: dict[str, threading.Lock] = {}


def _lock_for(session_id: str) -> threading.Lock:
    return _locks.setdefault(session_id, threading.Lock())


class ProgressReporter:
    """Passed into pipeline stages so they can report progress without
    knowing about the filesystem/job format."""

    def __init__(self, job_path: Path):
        self._job_path = job_path

    def update(self, stage: str, message: str, percent: int) -> None:
        _write(self._job_path, {
            "status": "running",
            "stage": stage,
            "message": message,
            "percent": percent,
            "result_files": [],
            "error": None,
        })


def _write(job_path: Path, data: dict) -> None:
    tmp = job_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(job_path)


def mark_done(session: Session, result_files: list[str], message: str = "Reconstruction complete") -> None:
    """Directly marks a session's job as done with the given result files,
    bypassing run_job/pipeline_fn - used when a result was produced outside
    this server entirely (see the Reprocess-on-GPU-elsewhere upload path in
    main.py) rather than by a pipeline function we ran ourselves."""
    _write(session.job_path, {
        "status": "done", "stage": "complete", "message": message,
        "percent": 100, "result_files": result_files, "error": None,
    })


def read_job(session: Session) -> dict:
    if not session.job_path.exists():
        return {"status": "new", "stage": "", "message": "Not started", "percent": 0,
                "result_files": [], "error": None}
    return json.loads(session.job_path.read_text())


def run_job(session: Session, pipeline_fn: Callable[[Session, ProgressReporter], list[str]]) -> None:
    """Runs pipeline_fn in a background thread, updating job.json as it goes.

    pipeline_fn must return the list of result filenames (relative to
    session.output_dir) it produced, or raise on failure.
    """
    lock = _lock_for(session.id)
    if not lock.acquire(blocking=False):
        return  # a job is already running for this session

    def _worker():
        reporter = ProgressReporter(session.job_path)
        try:
            _write(session.job_path, {
                "status": "running", "stage": "starting", "message": "Reconstruction started",
                "percent": 0, "result_files": [], "error": None,
            })
            result_files = pipeline_fn(session, reporter)
            _write(session.job_path, {
                "status": "done", "stage": "complete", "message": "Reconstruction complete",
                "percent": 100, "result_files": result_files, "error": None,
            })
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the client
            _write(session.job_path, {
                "status": "error", "stage": "failed", "message": str(exc),
                "percent": 0, "result_files": [], "error": traceback.format_exc(),
            })
        finally:
            lock.release()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
