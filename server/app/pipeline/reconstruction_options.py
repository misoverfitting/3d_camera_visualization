"""Explicit reconstruction settings and content-addressed stage checkpoints."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time


@dataclass(frozen=True)
class ReconstructionOptions:
    max_image_size: int = -1
    source_views: int = 20
    iterations: int = 5
    max_frames: int = 0
    feature_device: str = 'cpu'
    reuse: bool = True

    def __post_init__(self):
        if self.max_image_size != -1 and self.max_image_size < 64:
            raise ValueError('max_image_size must be -1 or at least 64')
        if self.source_views < 2 or self.iterations < 1:
            raise ValueError('At least two source views and one iteration required')
        if self.max_frames and self.max_frames < 8:
            raise ValueError('max_frames must be 0 or at least 8')
        if self.feature_device not in ('cpu', 'auto', 'cuda'):
            raise ValueError('feature_device must be cpu, auto or cuda')

    @classmethod
    def from_env(cls):
        return cls(max_image_size=int(os.environ.get('RECON_MAX_IMAGE_SIZE', '-1')),
                   source_views=int(os.environ.get('RECON_SOURCE_VIEWS', '20')),
                   iterations=int(os.environ.get('RECON_ITERATIONS', '5')),
                   max_frames=int(os.environ.get('RECON_MAX_FRAMES', '0')),
                   feature_device=os.environ.get('RECON_FEATURE_DEVICE', 'cpu'),
                   reuse=os.environ.get('RECON_REUSE', '1') != '0')

    def to_dict(self):
        return asdict(self)

    def stereo_args(self):
        return ['--PatchMatchStereo.geom_consistency', 'true',
                '--PatchMatchStereo.max_image_size', str(self.max_image_size),
                '--PatchMatchStereo.num_iterations', str(self.iterations)]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def input_fingerprint(session):
    images = [(p.name, file_digest(p)) for p in sorted(session.photos_dir.iterdir()) if p.is_file()]
    manifest = session.root / 'arkit.json'
    # The crop is downstream of fusion; changing it must not invalidate MVS.
    frames = json.loads(manifest.read_text(encoding='utf-8'))['frames'] if manifest.exists() else None
    return digest({'images': images, 'calibration_and_poses': frames})


def software_identity(binary):
    try:
        result = subprocess.run([binary, '-h'], capture_output=True, text=True)
        cli_help = result.stdout + result.stderr
    except FileNotFoundError:
        cli_help = 'unavailable: native CPU preview uses pycolmap'

    try:
        version = importlib.metadata.version('pycolmap')
    except importlib.metadata.PackageNotFoundError:
        version = None
    # Hash implementation sources so code changes cannot silently reuse old stages.
    sources = [Path(__file__), Path(__file__).with_name('colmap_common.py'),
               Path(__file__).with_name('keyframes.py'), Path(__file__).with_name('reconstruction_cache.py'),
               Path(__file__).with_name('arkit_pipeline.py'),
               Path(__file__).parents[1] / 'recording_format.py']
    return dict(binary=str(Path(shutil.which(binary) or binary).resolve()),
                cli_help='\n'.join(re.sub(r'^[IWEF]\d{8} [^\]]+\]\s*', '', line) for line in cli_help.splitlines()), pycolmap=version,
                implementation=digest([(p.name, file_digest(p)) for p in sources]))


class WorkSession:
    """Redirect scratch work only; originals, output paths and IDs remain intact."""
    def __init__(self, session, work):
        self._session, self.work_dir = session, Path(work)

    def __getattr__(self, name):
        return getattr(self._session, name)


def cached_stage(marker, artifacts, operation, reuse=True):
    """Only successful complete stages are reusable; failed artifacts are ignored."""
    record = read_record(marker)
    if reuse and valid_artifacts(record, artifacts):
        return {'reused': True, 'seconds': 0, 'original_seconds': record['seconds']}
    marker.unlink(missing_ok=True)
    started = time.perf_counter()
    operation()
    if not all(p.exists() and p.stat().st_size > 0 for p in artifacts):
        raise RuntimeError('Stage completed without expected artifacts')
    record = {'seconds': time.perf_counter() - started,
              'artifacts': {str(p): file_digest(p) for p in artifacts}}
    atomic_json(marker, record)
    return {'reused': False, **record}


def set_source_views(path, count):
    lines = path.read_text().splitlines()
    meaningful = [i for i, line in enumerate(lines) if line.strip() and not line.lstrip().startswith('#')]
    if not meaningful or len(meaningful) % 2:
        raise ValueError('Invalid PatchMatch source-view configuration')
    for index in meaningful[1::2]:
        lines[index] = f'__auto__, {count}'
    path.write_text('\n'.join(lines) + '\n')


def read_record(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def valid_artifacts(record, artifacts):
    return bool(artifacts) and 'seconds' in record and all(
        p.is_file() and p.stat().st_size > 0 and
        record.get('artifacts', {}).get(str(p)) == file_digest(p) for p in artifacts)
