"""On-disk workspace. Every stage reads and writes plain files here, so any stage can be
re-run, resumed after a crash, or inspected by hand.

    <root>/images/<id>.jpg          normalized upright JPEGs (long side <= MAX_SIDE_PX)
    <root>/images.jsonl             one ImageRecord per line
    <root>/labels/<id>.p<n>.json    label pass n (Claude)
    <root>/candidates/<id>.json     consensus items, plus mask stats after `segment`
    <root>/masks/<id>.npz           one boolean mask per candidate item
    <root>/verify/<id>.json         per-item verification (Claude)
    <root>/audit/<id>.json          sampled audit (stronger model)
    <root>/batches.json             submitted Message Batches still to collect
    <root>/usage.jsonl              token usage per request, for the cost report
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageRecord:
    id: str
    group: str            # Split unit: dish ID when known, else capture ID (tare_impl.md §9.1).
    source: str           # Where the image came from (path, plus frame for recordings).
    width: int
    height: int
    kind: str = "image"   # image | recording_frame | rim_test | food_upload
    meta: dict[str, Any] = field(default_factory=dict)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)
        for name in ("images", "labels", "candidates", "masks", "verify", "audit"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def image_path(self, image_id: str) -> Path:
        return self.root / "images" / f"{image_id}.jpg"

    def label_path(self, image_id: str, pass_index: int) -> Path:
        return self.root / "labels" / f"{image_id}.p{pass_index}.json"

    def candidates_path(self, image_id: str) -> Path:
        return self.root / "candidates" / f"{image_id}.json"

    def masks_path(self, image_id: str) -> Path:
        return self.root / "masks" / f"{image_id}.npz"

    def verify_path(self, image_id: str) -> Path:
        return self.root / "verify" / f"{image_id}.json"

    def audit_path(self, image_id: str) -> Path:
        return self.root / "audit" / f"{image_id}.json"

    # Image index ---------------------------------------------------------------

    def records(self) -> list[ImageRecord]:
        path = self.root / "images.jsonl"
        if not path.exists():
            return []
        return [ImageRecord(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]

    def add_records(self, records: list[ImageRecord]) -> int:
        known = {r.id for r in self.records()}
        new = [r for r in records if r.id not in known]
        with (self.root / "images.jsonl").open("a") as f:
            for record in new:
                f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        return len(new)

    # Batches and usage ----------------------------------------------------------

    def batches(self) -> dict[str, dict]:
        return read_json(self.root / "batches.json", {})

    def save_batches(self, batches: dict[str, dict]) -> None:
        write_json(self.root / "batches.json", batches)

    def log_usage(self, entry: dict) -> None:
        with (self.root / "usage.jsonl").open("a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    def usage(self) -> list[dict]:
        path = self.root / "usage.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
