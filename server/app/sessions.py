"""Filesystem-backed capture session storage.

Each session is a directory under DATA_DIR/sessions/<id>/ containing:
  session.json   - metadata (name, created_at)
  photos/         - uploaded phone photos
  work/           - COLMAP / reconstruction scratch space
  output/         - final exportable model files
  job.json        - latest reconstruction job status
"""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config


class SessionNotFound(Exception):
    pass


@dataclass
class Session:
    id: str
    name: str
    created_at: str
    root: Path = field(repr=False)

    @property
    def photos_dir(self) -> Path:
        return self.root / "photos"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def job_path(self) -> Path:
        return self.root / "job.json"

    def photo_count(self) -> int:
        if not self.photos_dir.exists():
            return 0
        return sum(1 for _ in self.photos_dir.iterdir() if _.is_file())

    def to_info(self) -> dict:
        status = "new"
        if self.job_path.exists():
            try:
                status = json.loads(self.job_path.read_text()).get("status", "new")
            except (json.JSONDecodeError, OSError):
                status = "new"
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "photo_count": self.photo_count(),
            "status": status,
        }


def create_session(name: str) -> Session:
    session_id = uuid.uuid4().hex[:12]
    root = config.SESSIONS_DIR / session_id
    (root / "photos").mkdir(parents=True, exist_ok=True)
    (root / "work").mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    meta = {"id": session_id, "name": name, "created_at": created_at}
    (root / "session.json").write_text(json.dumps(meta, indent=2))
    return Session(id=session_id, name=name, created_at=created_at, root=root)


def get_session(session_id: str) -> Session:
    root = config.SESSIONS_DIR / session_id
    meta_path = root / "session.json"
    if not meta_path.exists():
        raise SessionNotFound(session_id)
    meta = json.loads(meta_path.read_text())
    return Session(id=meta["id"], name=meta["name"], created_at=meta["created_at"], root=root)


def list_sessions() -> list[Session]:
    sessions = []
    for root in sorted(config.SESSIONS_DIR.iterdir(), reverse=True):
        if not root.is_dir():
            continue
        try:
            sessions.append(get_session(root.name))
        except SessionNotFound:
            continue
    return sessions


def delete_session(session_id: str) -> None:
    root = config.SESSIONS_DIR / session_id
    if not root.exists():
        raise SessionNotFound(session_id)
    shutil.rmtree(root)
