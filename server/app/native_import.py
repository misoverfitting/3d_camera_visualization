"""Bounded ZIP ingestion into a new, immutable-input session."""
from __future__ import annotations

import json
import math
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from . import config, sessions
from .recording_format import validate

MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_EXPANDED_BYTES = 1536 * 1024 * 1024


def import_recording(stream, name: str = 'iPhone recording') -> sessions.Session:
    stream.seek(0, 2)
    if stream.tell() > MAX_ARCHIVE_BYTES:
        raise ValueError('Recording ZIP exceeds 1 GiB')
    stream.seek(0)
    with tempfile.TemporaryDirectory(dir=config.DATA_DIR) as staging:
        root = Path(staging)
        with zipfile.ZipFile(stream) as archive:
            members = archive.infolist()
            if len(members) > 500 or sum(m.file_size for m in members) > MAX_EXPANDED_BYTES:
                raise ValueError('Recording archive exceeds file-count or expanded-size limits')
            seen = set()
            for member in members:
                path = PurePosixPath(member.filename)
                if (path.is_absolute() or '..' in path.parts or '\\' in member.filename
                    or ':' in member.filename or stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError('Unsafe archive member')
                key = str(path).casefold()
                if key in seen:
                    raise ValueError('Duplicate archive member')
                seen.add(key)
                target = root.joinpath(*path.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open('wb') as dest:
                        shutil.copyfileobj(source, dest)
        manifests = list(root.rglob('manifest.json'))
        if len(manifests) != 1:
            raise ValueError('ZIP must contain exactly one recording manifest')
        recording_root = manifests[0].parent
        data = validate(recording_root)
        if len(data['frames']) < 8:
            raise ValueError('At least 8 frames are required for reconstruction')
        region = data.get('region')
        if region is not None:
            origin, size = region['origin'], region['size_m']
            if (len(origin) != 3 or not all(math.isfinite(x) for x in origin)
                or not math.isfinite(size) or not 0 < size <= 3
                or region.get('convention') != 'world aligned; origin at bottom center'):
                raise ValueError('Invalid scan region')
        session = sessions.create_session(name[:200])
        try:
            for i, frame in enumerate(data['frames']):
                dest = f'{i:05d}.jpg'
                shutil.copyfile(recording_root / frame['image'], session.photos_dir / dest)
                frame['image'] = dest
            (session.root / 'arkit.json').write_text(json.dumps(data), encoding='utf-8')
            return session
        except Exception:
            sessions.delete_session(session.id)
            raise
