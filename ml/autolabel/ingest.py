"""Collects images into the workspace: upright, RGB, long side <= MAX_SIDE_PX.

Understands the app's outputs as well as plain image folders:
- Recordings (folder or exported .zip with manifest.json + images/): a few frames per
  capture, chosen for view diversity; sensor-orientation frames are rotated upright.
- Rim tests (Documents/RimTests/<stamp>/ with rim.json): both photos.
- Food uploads saved by the backend (FOOD_SAVE_UPLOADS=1): already upright plate crops.
- Any other image: grouped by its folder, or by --group.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

from .config import MAX_SIDE_PX
from .store import ImageRecord, Workspace

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def _normalize(data: bytes, rotate_clockwise: bool) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if rotate_clockwise:
        image = image.transpose(Image.Transpose.ROTATE_270)  # 90° clockwise
    scale = MAX_SIDE_PX / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return image


def _add(ws: Workspace, data: bytes, *, group: str, source: str, kind: str, rotate: bool = False,
         meta: dict | None = None) -> ImageRecord | None:
    image_id = hashlib.sha256(data + (b"r" if rotate else b"")).hexdigest()[:24]
    path = ws.image_path(image_id)
    try:
        image = Image.open(path) if path.exists() else _normalize(data, rotate)
    except Exception as exc:  # Unreadable or unsupported file: skip, keep going.
        print(f"skip {source}: {exc}")
        return None
    if not path.exists():
        image.save(path, quality=90)
    return ImageRecord(id=image_id, group=group, source=source, width=image.width, height=image.height,
                       kind=kind, meta=meta or {})


def _forward(camera_to_world: list[float]) -> tuple[float, float, float]:
    # Column-major 4x4; the camera looks along -Z (ARKit axes).
    return (-camera_to_world[8], -camera_to_world[9], -camera_to_world[10])


def select_frames(frames: list[dict], count: int) -> list[dict]:
    """Most overhead frame first, then greedily the frame whose view direction is
    farthest from everything chosen (diverse angles, few near-duplicates)."""
    usable = [f for f in frames if f.get("tracking", "normal") == "normal" and len(f.get("camera_to_world", [])) == 16]
    if not usable:
        return []
    dirs = [_forward(f["camera_to_world"]) for f in usable]
    norm = [math.sqrt(sum(c * c for c in d)) or 1 for d in dirs]
    dirs = [tuple(c / n for c in d) for d, n in zip(dirs, norm)]
    chosen = [min(range(len(usable)), key=lambda i: dirs[i][1])]  # Most downward (-Y).
    while len(chosen) < min(count, len(usable)):
        def gap(i: int) -> float:
            return min(math.acos(max(-1, min(1, sum(a * b for a, b in zip(dirs[i], dirs[j]))))) for j in chosen)
        chosen.append(max((i for i in range(len(usable)) if i not in chosen), key=gap))
    return [usable[i] for i in chosen]


def _recording(ws: Workspace, root: Path, read, group: str | None, frames_per_capture: int) -> list[ImageRecord]:
    manifest = json.loads(read("manifest.json"))
    capture = group or manifest.get("dish_id") or root.stem
    records = []
    for frame in select_frames(manifest.get("frames", []), frames_per_capture):
        rotate = manifest.get("image_orientation", "native_sensor_unrotated") == "native_sensor_unrotated"
        record = _add(ws, read(frame["image"]), group=capture, source=f"{root}#{frame['image']}",
                      kind="recording_frame", rotate=rotate, meta={"timestamp": frame.get("timestamp_seconds")})
        if record:
            records.append(record)
    return records


def ingest(ws: Workspace, paths: list[Path], group: str | None = None, frames_per_capture: int = 8) -> int:
    records = _collect(ws, paths, group, frames_per_capture)
    added = ws.add_records(records)
    print(f"ingested {added} new image(s) ({len(records) - added} already present)")
    return added


def _collect(ws: Workspace, paths: list[Path], group: str | None, frames_per_capture: int) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                names = {Path(n).as_posix(): n for n in archive.namelist()}
                manifest = next((n for n in names if n.endswith("manifest.json")), None)
                if manifest is None:
                    print(f"skip {path}: no manifest.json")
                    continue
                prefix = manifest[: -len("manifest.json")]
                records += _recording(ws, path, lambda name: archive.read(names[prefix + name]), group, frames_per_capture)
        elif path.is_dir() and (path / "manifest.json").exists():
            records += _recording(ws, path, lambda name, p=path: (p / name).read_bytes(), group, frames_per_capture)
        elif path.is_dir() and (path / "rim.json").exists():
            for view in json.loads((path / "rim.json").read_text()).get("views", []):
                record = _add(ws, (path / view["image"]).read_bytes(), group=group or f"rim-{path.name}",
                              source=str(path / view["image"]), kind="rim_test")
                if record:
                    records.append(record)
        elif path.is_dir():
            special = [p for p in sorted(path.iterdir()) if p.is_dir() or p.suffix.lower() == ".zip"]
            records += _collect(ws, special, group, frames_per_capture)
            for file in sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS):
                sidecar = file.with_suffix(".json")
                upload = sidecar.exists() and "items" in json.loads(sidecar.read_text())
                # Each backend upload is its own plate, so its own group.
                record = _add(ws, file.read_bytes(), group=group or (f"upload-{file.stem}" if upload else path.name),
                              source=str(file), kind="food_upload" if upload else "image")
                if record:
                    records.append(record)
        elif path.suffix.lower() in IMAGE_EXTS:
            record = _add(ws, path.read_bytes(), group=group or path.parent.name, source=str(path), kind="image")
            if record:
                records.append(record)
        else:
            print(f"skip {path}: not an image, folder, recording or zip")
    return records
